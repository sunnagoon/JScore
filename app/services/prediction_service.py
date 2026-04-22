from __future__ import annotations

import datetime as dt
import joblib
import json
import math
import os
import threading
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import requests
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

from app.config import CACHE_PATH, PREDICTION_ARCHIVE_DIR
from app.services.mlb_service import BASE_API_URL, REQUEST_TIMEOUT_SECONDS, fetch_team_api_metrics, fetch_team_name_lookup

FINAL_STATES = {"Final", "Game Over", "Completed Early"}
REGULAR_SEASON_GAME_TYPE = "R"
MODEL_VERSION = 5
MODEL_VARIANTS = ("v2", "v3", "v4")
V3_MULTI_SEASON_LOOKBACK = 3
V4_MULTI_SEASON_LOOKBACK = 3
V4_TREE_MAX_ITER = 420

FEATURE_NAMES = [
    "home_advantage",
    "win_pct_diff",
    "run_diff_pg_diff",
    "runs_pg_diff",
    "runs_allowed_pg_diff",
    "recent10_diff",
    "home_away_split_diff",
    "rest_days_diff",
    "bullpen_recent_diff",
    "pythag_diff",
    "elo_diff_per_100",
    "recent7_diff",
    "recent14_diff",
    "recent30_diff",
    "run_diff_pg14_diff",
    "run_diff_pg30_diff",
    "runs_pg14_diff",
    "runs_allowed_pg14_diff",
    "form_trend_diff",
    "form_volatility_edge",
    "sample_reliability_diff",
]

V3_FEATURE_INTERACTIONS = [
    "win_pct_x_elo",
    "run_diff_x_elo",
    "recent10_x_rest",
    "bullpen_x_rest",
    "pythag_x_elo",
    "abs_rest_diff",
    "abs_elo_diff",
    "run_diff_sq",
    "elo_sq",
    "win_pct_sq",
    "recent7_x_elo",
    "form_trend_x_elo",
    "run_diff14_x_rest",
    "sample_rel_x_elo",
    "volatility_x_rest",
]

DEFAULT_PRIOR_GAMES = 10.0
DEFAULT_RUNS_PER_GAME = 4.5
ELO_BASE = 1500.0
ELO_K = 20.0
ELO_HOME_FIELD_ADV = 35.0


def _read_env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


HOME_PRIOR_NEUTRALIZE_STRENGTH = max(0.0, min(1.0, _read_env_float("MSCORE_HOME_PRIOR_NEUTRALIZE_STRENGTH", 0.35)))
DEFAULT_HOME_BASE_RATE = 0.5318
MARKET_FETCH_SCHEMA_VERSION = "v3"
MARKET_BLEND_MIN_WEIGHT = 0.16
MARKET_BLEND_MAX_WEIGHT = 0.56
PREGAME_CONTEXT_MIN_MULTIPLIER = 0.55

MODEL_CACHE_LOCK = threading.Lock()
MODEL_CACHE: dict[Any, dict[str, Any]] = {}
PREDICTION_CACHE: dict[str, Any] = {}
PITCHER_CACHE: dict[tuple[int, int], dict[str, Any]] = {}
MODEL_OBJECT_CACHE: dict[str, Any] = {}
TEAM_CONTEXT_CACHE: dict[int, dict[int, dict[str, Any]]] = {}
LINEUP_CACHE: dict[int, dict[str, Any]] = {}
TEAM_HITTING_BASELINE_CACHE: dict[int, dict[int, float]] = {}
ADVANCED_TEAM_CACHE: dict[int, dict[str, dict[str, float]]] = {}

XWOBA_CON_CANDIDATE_KEYS = (
    "mlb::seasonAdvanced::hitting::xwobaCon",
    "mlb::seasonAdvanced::hitting::xwobacon",
    "mlb::seasonAdvanced::hitting::xwOBAcon",
)
XFIP_CANDIDATE_KEYS = (
    "mlb::seasonAdvanced::pitching::xFip",
    "mlb::seasonAdvanced::pitching::xfip",
    "mlb::season::pitching::xFip",
)


class TeamState:
    def __init__(self) -> None:
        self.wins = 0
        self.losses = 0
        self.runs_for = 0
        self.runs_against = 0
        self.recent = deque(maxlen=30)
        self.runs_for_recent = deque(maxlen=30)
        self.runs_against_recent = deque(maxlen=30)
        self.home_wins = 0
        self.home_losses = 0
        self.away_wins = 0
        self.away_losses = 0
        self.runs_allowed_recent = deque(maxlen=5)
        self.bullpen_outs_recent = deque(maxlen=7)
        self.game_dates_recent = deque(maxlen=12)
        self.last_timezone_offset: float | None = None
        self.last_game_date: dt.date | None = None
        self.elo = ELO_BASE

    @property
    def games(self) -> int:
        return self.wins + self.losses


def _report_path(season: int, model_variant: str = "v4", seasons_back: int = V4_MULTI_SEASON_LOOKBACK) -> Path:
    variant = str(model_variant).lower()
    if variant == "v2":
        return CACHE_PATH.parent / f"backtest_report_{season}_v2.json"
    if variant == "v3":
        return CACHE_PATH.parent / f"backtest_report_{season}_v3_s{max(1, int(seasons_back))}.json"

    return CACHE_PATH.parent / f"backtest_report_{season}_v4_s{max(1, int(seasons_back))}.json"


def _model_object_path(season: int, model_variant: str, seasons_back: int) -> Path:
    variant = str(model_variant).lower()
    return CACHE_PATH.parent / f"backtest_model_{season}_{variant}_s{max(1, int(seasons_back))}.joblib"


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_rounded(value: Any, digits: int) -> float | None:
    num = _to_float(value, float("nan"))
    if math.isnan(num) or math.isinf(num):
        return None
    return round(float(num), digits)


def _clamp_prob(prob: float, lower: float = 0.02, upper: float = 0.98) -> float:
    return float(max(lower, min(upper, prob)))


def _pick_team_from_home_prob(home_prob: float, home_team: str, away_team: str) -> str:
    return home_team if float(home_prob) >= 0.5 else away_team


def _pick_probability(home_prob: float, pick_team: str, home_team: str, away_team: str) -> float:
    if pick_team == home_team:
        return float(home_prob)
    if pick_team == away_team:
        return float(1.0 - home_prob)
    return float(max(home_prob, 1.0 - home_prob))


def _edge_tier(edge_pick: float | None) -> str:
    if edge_pick is None:
        return "none"

    magnitude = abs(float(edge_pick))
    if magnitude >= 0.08:
        return "elite"
    if magnitude >= 0.05:
        return "strong"
    if magnitude >= 0.03:
        return "actionable"
    if magnitude >= 0.015:
        return "thin"
    return "none"


def _market_edge_fields(
    *,
    home_team: str,
    away_team: str,
    home_win_prob: float,
    pre_market_home_prob: float,
    market_home_prob: float | None,
) -> dict[str, Any]:
    model_pick_team = _pick_team_from_home_prob(home_win_prob, home_team=home_team, away_team=away_team)

    if market_home_prob is None:
        return {
            "model_pick_team": model_pick_team,
            "market_favored_team": None,
            "market_favored_prob": None,
            "model_vs_market_edge_home": None,
            "model_vs_market_edge_pick": None,
            "value_tier": "none",
            "market_disagrees_with_model": False,
        }

    market_home = float(market_home_prob)
    market_favored_team = _pick_team_from_home_prob(market_home, home_team=home_team, away_team=away_team)
    market_favored_prob = max(market_home, 1.0 - market_home)

    model_pick_prob = _pick_probability(pre_market_home_prob, model_pick_team, home_team=home_team, away_team=away_team)
    market_pick_prob = _pick_probability(market_home, model_pick_team, home_team=home_team, away_team=away_team)

    edge_home = float(pre_market_home_prob - market_home)
    edge_pick = float(model_pick_prob - market_pick_prob)

    return {
        "model_pick_team": model_pick_team,
        "market_favored_team": market_favored_team,
        "market_favored_prob": market_favored_prob,
        "model_vs_market_edge_home": edge_home,
        "model_vs_market_edge_pick": edge_pick,
        "value_tier": _edge_tier(edge_pick),
        "market_disagrees_with_model": bool(model_pick_team != market_favored_team),
    }


def _sigmoid(z: np.ndarray) -> np.ndarray:
    clipped = np.clip(z, -35, 35)
    return 1.0 / (1.0 + np.exp(-clipped))


def _logit(p: np.ndarray) -> np.ndarray:
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def _neutralize_home_prior(
    home_prob: float,
    *,
    home_base_rate: float,
    strength: float = HOME_PRIOR_NEUTRALIZE_STRENGTH,
) -> tuple[float, float]:
    p = _clamp_prob(home_prob)
    s = max(0.0, min(1.0, float(strength)))
    if s <= 0:
        return p, 0.0

    base = _clamp_prob(max(0.5, min(0.65, float(home_base_rate))))
    shift = float(_logit(np.asarray([base], dtype=float))[0]) * s
    adjusted_logit = float(_logit(np.asarray([p], dtype=float))[0]) - shift
    neutralized = float(_sigmoid(np.asarray([adjusted_logit], dtype=float))[0])
    neutralized = _clamp_prob(neutralized)
    return neutralized, float(neutralized - p)


def _smoothed_rate(num: float, den: float, prior_mean: float = 0.5, prior_weight: float = 6.0) -> float:
    return (num + prior_mean * prior_weight) / (den + prior_weight)


def _window_smoothed(values: list[float], window: int, prior_mean: float, prior_weight: float) -> float:
    if window <= 0:
        return float(prior_mean)
    if not values:
        return float(prior_mean)
    window_values = values[-window:]
    n = float(len(window_values))
    return (float(sum(window_values)) + prior_mean * prior_weight) / (n + prior_weight)


def _team_features(state: TeamState) -> dict[str, float]:
    games = float(state.games)
    win_pct = _smoothed_rate(state.wins, games, prior_mean=0.5, prior_weight=DEFAULT_PRIOR_GAMES)

    run_diff_pg = (state.runs_for - state.runs_against) / (games + DEFAULT_PRIOR_GAMES)
    runs_pg = (state.runs_for + DEFAULT_PRIOR_GAMES * DEFAULT_RUNS_PER_GAME) / (games + DEFAULT_PRIOR_GAMES)
    runs_allowed_pg = (state.runs_against + DEFAULT_PRIOR_GAMES * DEFAULT_RUNS_PER_GAME) / (games + DEFAULT_PRIOR_GAMES)

    recent_values = list(state.recent)
    runs_for_recent = list(state.runs_for_recent)
    runs_against_recent = list(state.runs_against_recent)

    recent10 = _window_smoothed(recent_values, window=10, prior_mean=0.5, prior_weight=5.0)
    recent7 = _window_smoothed(recent_values, window=7, prior_mean=0.5, prior_weight=6.0)
    recent14 = _window_smoothed(recent_values, window=14, prior_mean=0.5, prior_weight=7.0)
    recent30 = _window_smoothed(recent_values, window=30, prior_mean=0.5, prior_weight=9.0)

    runs_pg_14 = _window_smoothed(runs_for_recent, window=14, prior_mean=DEFAULT_RUNS_PER_GAME, prior_weight=7.0)
    runs_allowed_pg_14 = _window_smoothed(runs_against_recent, window=14, prior_mean=DEFAULT_RUNS_PER_GAME, prior_weight=7.0)
    run_diff_pg_14 = runs_pg_14 - runs_allowed_pg_14

    runs_pg_30 = _window_smoothed(runs_for_recent, window=30, prior_mean=DEFAULT_RUNS_PER_GAME, prior_weight=10.0)
    runs_allowed_pg_30 = _window_smoothed(runs_against_recent, window=30, prior_mean=DEFAULT_RUNS_PER_GAME, prior_weight=10.0)
    run_diff_pg_30 = runs_pg_30 - runs_allowed_pg_30

    last14 = recent_values[-14:]
    last14_n = float(len(last14))
    if last14:
        mu = float(sum(last14)) / last14_n
        var = float(sum((v - mu) ** 2 for v in last14) / last14_n)
    else:
        var = 0.25
    form_volatility = ((var * last14_n) + (0.25 * 6.0)) / (last14_n + 6.0)

    form_trend = (recent7 - recent30)
    sample_reliability = games / (games + 30.0)

    home_games = state.home_wins + state.home_losses
    away_games = state.away_wins + state.away_losses
    home_win_pct = _smoothed_rate(state.home_wins, home_games, prior_mean=0.54, prior_weight=4.0)
    away_win_pct = _smoothed_rate(state.away_wins, away_games, prior_mean=0.46, prior_weight=4.0)

    rf = state.runs_for + DEFAULT_RUNS_PER_GAME * DEFAULT_PRIOR_GAMES
    ra = state.runs_against + DEFAULT_RUNS_PER_GAME * DEFAULT_PRIOR_GAMES
    exponent = 1.83
    pythag_win_pct = (rf**exponent) / ((rf**exponent) + (ra**exponent)) if (rf + ra) > 0 else 0.5

    recent_ra_games = len(state.runs_allowed_recent)
    bullpen_ra_recent = (sum(state.runs_allowed_recent) + DEFAULT_RUNS_PER_GAME * 3.0) / (recent_ra_games + 3.0)

    return {
        "win_pct": win_pct,
        "run_diff_pg": run_diff_pg,
        "runs_pg": runs_pg,
        "runs_allowed_pg": runs_allowed_pg,
        "recent10": recent10,
        "recent7": recent7,
        "recent14": recent14,
        "recent30": recent30,
        "run_diff_pg_14": run_diff_pg_14,
        "run_diff_pg_30": run_diff_pg_30,
        "runs_pg_14": runs_pg_14,
        "runs_allowed_pg_14": runs_allowed_pg_14,
        "form_trend": form_trend,
        "form_volatility": form_volatility,
        "sample_reliability": sample_reliability,
        "home_win_pct": home_win_pct,
        "away_win_pct": away_win_pct,
        "pythag_win_pct": pythag_win_pct,
        "bullpen_ra_recent": bullpen_ra_recent,
    }


def _rest_days(state: TeamState, game_date: dt.date) -> float:
    if state.last_game_date is None:
        return 2.0

    delta = (game_date - state.last_game_date).days - 1
    return float(max(-1, min(5, delta)))


def _elo_home_win_prob(home_elo: float, away_elo: float) -> float:
    diff = (home_elo + ELO_HOME_FIELD_ADV) - away_elo
    return float(1.0 / (1.0 + (10.0 ** (-diff / 400.0))))


def _build_feature_vector(
    home_state: TeamState,
    away_state: TeamState,
    home_rest: float,
    away_rest: float,
) -> list[float]:
    home = _team_features(home_state)
    away = _team_features(away_state)

    return [
        1.0,
        home["win_pct"] - away["win_pct"],
        home["run_diff_pg"] - away["run_diff_pg"],
        home["runs_pg"] - away["runs_pg"],
        away["runs_allowed_pg"] - home["runs_allowed_pg"],
        home["recent10"] - away["recent10"],
        home["home_win_pct"] - away["away_win_pct"],
        home_rest - away_rest,
        away["bullpen_ra_recent"] - home["bullpen_ra_recent"],
        home["pythag_win_pct"] - away["pythag_win_pct"],
        (home_state.elo - away_state.elo) / 100.0,
        home["recent7"] - away["recent7"],
        home["recent14"] - away["recent14"],
        home["recent30"] - away["recent30"],
        home["run_diff_pg_14"] - away["run_diff_pg_14"],
        home["run_diff_pg_30"] - away["run_diff_pg_30"],
        home["runs_pg_14"] - away["runs_pg_14"],
        away["runs_allowed_pg_14"] - home["runs_allowed_pg_14"],
        home["form_trend"] - away["form_trend"],
        away["form_volatility"] - home["form_volatility"],
        home["sample_reliability"] - away["sample_reliability"],
    ]


def _recent_games_count(state: TeamState, game_date: dt.date, lookback_days: int = 3) -> int:
    count = 0
    for d in state.game_dates_recent:
        delta = (game_date - d).days
        if 0 <= delta <= lookback_days:
            count += 1
    return count


def _recent_bullpen_outs(state: TeamState, games_window: int = 3) -> float:
    if not state.bullpen_outs_recent:
        return 0.0
    values = list(state.bullpen_outs_recent)[-max(1, games_window) :]
    return float(sum(values))


def _prepare_features_for_model_variant(
    base_features: np.ndarray,
    model_variant: str,
) -> np.ndarray:
    variant = _resolve_model_variant(model_variant)
    if variant == "v2":
        return np.asarray(base_features, dtype=float)

    return _expand_feature_vector_v3(np.asarray(base_features, dtype=float))


def _bullpen_context_adjustment(
    home_state: TeamState,
    away_state: TeamState,
    game_date: dt.date,
) -> tuple[float, dict[str, float]]:
    home_outs = _recent_bullpen_outs(home_state, games_window=3)
    away_outs = _recent_bullpen_outs(away_state, games_window=3)

    home_games = _recent_games_count(home_state, game_date=game_date, lookback_days=3)
    away_games = _recent_games_count(away_state, game_date=game_date, lookback_days=3)

    outs_edge = (away_outs - home_outs) * 0.0015
    games_edge = float(away_games - home_games) * 0.01

    total = max(-0.06, min(0.06, outs_edge + games_edge))
    return total, {
        "bullpen_adjustment": round(total, 4),
        "bullpen_outs_edge": round(outs_edge, 4),
        "bullpen_games_edge": round(games_edge, 4),
    }



def _bullpen_health_adjustment(
    home_state: TeamState,
    away_state: TeamState,
    game_date: dt.date,
) -> tuple[float, dict[str, float]]:
    home_outs = _recent_bullpen_outs(home_state, games_window=3)
    away_outs = _recent_bullpen_outs(away_state, games_window=3)

    home_games = _recent_games_count(home_state, game_date=game_date, lookback_days=3)
    away_games = _recent_games_count(away_state, game_date=game_date, lookback_days=3)

    home_ra = (sum(home_state.runs_allowed_recent) + DEFAULT_RUNS_PER_GAME * 3.0) / (len(home_state.runs_allowed_recent) + 3.0)
    away_ra = (sum(away_state.runs_allowed_recent) + DEFAULT_RUNS_PER_GAME * 3.0) / (len(away_state.runs_allowed_recent) + 3.0)

    home_fatigue = min(1.0, (home_outs / 36.0) + (float(home_games) * 0.08))
    away_fatigue = min(1.0, (away_outs / 36.0) + (float(away_games) * 0.08))

    home_effectiveness = max(0.72, min(1.28, DEFAULT_RUNS_PER_GAME / max(2.2, home_ra)))
    away_effectiveness = max(0.72, min(1.28, DEFAULT_RUNS_PER_GAME / max(2.2, away_ra)))

    home_health = max(0.6, min(1.4, 1.0 + (home_effectiveness - 1.0) * 0.65 - (home_fatigue - 0.4) * 0.55))
    away_health = max(0.6, min(1.4, 1.0 + (away_effectiveness - 1.0) * 0.65 - (away_fatigue - 0.4) * 0.55))

    health_edge = (home_health - away_health) * 0.085
    total = max(-0.05, min(0.05, health_edge))

    return total, {
        "bullpen_health_adjustment": round(total, 4),
        "bullpen_health_edge": round(health_edge, 4),
        "bullpen_health_home_score": round(home_health * 100.0, 1),
        "bullpen_health_away_score": round(away_health * 100.0, 1),
        "bullpen_home_fatigue": round(home_fatigue, 4),
        "bullpen_away_fatigue": round(away_fatigue, 4),
        "bullpen_home_recent_ra": round(home_ra, 3),
        "bullpen_away_recent_ra": round(away_ra, 3),
    }


def _fetch_team_context_lookup(season: int) -> dict[int, dict[str, Any]]:
    cached = TEAM_CONTEXT_CACHE.get(int(season))
    if cached is not None:
        return cached

    try:
        response = requests.get(
            f"{BASE_API_URL}/teams",
            params={"sportId": 1, "season": season, "hydrate": "venue(timezone)"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        TEAM_CONTEXT_CACHE[int(season)] = {}
        return {}

    context: dict[int, dict[str, Any]] = {}
    for team in payload.get("teams", []):
        team_id = team.get("id")
        if team_id is None:
            continue

        venue = team.get("venue", {}) or {}
        tz = venue.get("timeZone", {}) or {}
        context[int(team_id)] = {
            "name": team.get("name"),
            "tz_offset": _to_float(tz.get("offset"), 0.0),
            "tz_id": tz.get("id"),
            "venue_id": venue.get("id"),
        }

    TEAM_CONTEXT_CACHE[int(season)] = context
    return context



def _fetch_team_hitting_baseline_lookup(season: int) -> dict[int, float]:
    cached = TEAM_HITTING_BASELINE_CACHE.get(int(season))
    if cached is not None:
        return cached

    output: dict[int, float] = {}
    try:
        response = requests.get(
            f"{BASE_API_URL}/teams/stats",
            params={"sportId": 1, "season": season, "stats": "season", "group": "hitting"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()

        stats_blocks = payload.get("stats") or []
        splits = (stats_blocks[0].get("splits") if stats_blocks else []) or []
        for row in splits:
            team_obj = row.get("team", {}) or {}
            team_id = team_obj.get("id")
            if team_id is None:
                continue

            stat = row.get("stat", {}) or {}
            ops = _to_float(stat.get("ops"), float("nan"))
            if math.isnan(ops) or ops <= 0:
                continue

            output[int(team_id)] = float(ops)
    except Exception:
        output = {}

    TEAM_HITTING_BASELINE_CACHE[int(season)] = output
    return output


def _first_metric_value(stats: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        raw = stats.get(key)
        if raw is None or raw == "" or raw == "-" or raw == "--":
            continue
        value = _to_float(raw, float("nan"))
        if not math.isnan(value):
            return float(value)
    return None


def _ratio_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    ratio = float(value)
    if math.isnan(ratio):
        return None
    if abs(ratio) > 1.5:
        ratio = ratio / 100.0
    if abs(ratio) > 1.5:
        ratio = ratio / 100.0
    return ratio


def _clip(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def _fetch_advanced_team_lookup(season: int) -> dict[str, dict[str, float]]:
    cached = ADVANCED_TEAM_CACHE.get(int(season))
    if cached is not None:
        return cached

    try:
        team_metrics, _ = fetch_team_api_metrics(season=season)
    except Exception:
        ADVANCED_TEAM_CACHE[int(season)] = {}
        return {}

    if not team_metrics:
        ADVANCED_TEAM_CACHE[int(season)] = {}
        return {}

    rows: dict[str, dict[str, float]] = {}
    for team_name, stats in team_metrics.items():
        if not team_name:
            continue

        ops = _to_float(stats.get("mlb::season::hitting::ops"), 0.720)
        games_played = _to_float(stats.get("mlb::season::hitting::gamesPlayed"), 0.0)
        leverage_conf = _clip((games_played / 95.0) if games_played > 0 else 0.25, 0.25, 1.0)

        hitting_ops_risp = _to_float(stats.get("mlb::situational::hitting::risp::ops"), ops)
        hitting_ops_risp2 = _to_float(stats.get("mlb::situational::hitting::risp2::ops"), hitting_ops_risp)
        hitting_ops_late_close = _to_float(stats.get("mlb::situational::hitting::lc::ops"), ops)

        pitching_ops_allowed = _to_float(stats.get("mlb::season::pitching::ops"), 0.720)
        pitching_ops_risp_allowed = _to_float(stats.get("mlb::situational::pitching::risp::ops"), pitching_ops_allowed)
        pitching_ops_risp2_allowed = _to_float(stats.get("mlb::situational::pitching::risp2::ops"), pitching_ops_risp_allowed)
        pitching_ops_late_close_allowed = _to_float(stats.get("mlb::situational::pitching::lc::ops"), pitching_ops_allowed)

        hitting_ops_risp = _clip(hitting_ops_risp, 0.450, 1.220)
        hitting_ops_risp2 = _clip(hitting_ops_risp2, 0.420, 1.240)
        hitting_ops_late_close = _clip(hitting_ops_late_close, 0.420, 1.220)
        pitching_ops_allowed = _clip(pitching_ops_allowed, 0.450, 1.120)
        pitching_ops_risp_allowed = _clip(pitching_ops_risp_allowed, 0.420, 1.240)
        pitching_ops_risp2_allowed = _clip(pitching_ops_risp2_allowed, 0.420, 1.260)
        pitching_ops_late_close_allowed = _clip(pitching_ops_late_close_allowed, 0.420, 1.240)

        leverage_offense_raw = _clip(
            0.50 * ((hitting_ops_late_close - 0.450) / 0.650)
            + 0.30 * ((hitting_ops_risp2 - 0.450) / 0.650)
            + 0.20 * ((hitting_ops_risp - 0.450) / 0.650),
            0.0,
            1.0,
        )
        leverage_pitching_raw = _clip(
            0.50 * ((1.100 - pitching_ops_late_close_allowed) / 0.650)
            + 0.30 * ((1.100 - pitching_ops_risp2_allowed) / 0.650)
            + 0.20 * ((1.100 - pitching_ops_risp_allowed) / 0.650),
            0.0,
            1.0,
        )
        leverage_offense_quality = _clip(leverage_conf * leverage_offense_raw + (1.0 - leverage_conf) * 0.5, 0.0, 1.0)
        leverage_pitching_quality = _clip(leverage_conf * leverage_pitching_raw + (1.0 - leverage_conf) * 0.5, 0.0, 1.0)
        leverage_net_quality = _clip(0.52 * leverage_offense_quality + 0.48 * leverage_pitching_quality, 0.0, 1.0)

        clutch_off_delta = _clip(
            leverage_conf * (0.55 * (hitting_ops_late_close - ops) + 0.45 * (hitting_ops_risp2 - ops)),
            -0.250,
            0.250,
        )
        clutch_pitch_delta = _clip(
            leverage_conf * (pitching_ops_allowed - (0.55 * pitching_ops_late_close_allowed + 0.45 * pitching_ops_risp2_allowed)),
            -0.250,
            0.250,
        )
        clutch_index = _clip(0.5 + (0.55 * clutch_off_delta + 0.45 * clutch_pitch_delta) / 0.30, 0.0, 1.0)

        k_pct = _ratio_or_none(
            _first_metric_value(
                stats,
                (
                    "mlb::seasonAdvanced::hitting::strikeoutPercentage",
                    "mlb::seasonAdvanced::hitting::strikeOutPercentage",
                ),
            )
        )
        bb_pct = _ratio_or_none(
            _first_metric_value(
                stats,
                (
                    "mlb::seasonAdvanced::hitting::walksPercentage",
                    "mlb::seasonAdvanced::hitting::baseOnBallsPercentage",
                ),
            )
        )
        hard_hit_pct = _ratio_or_none(_first_metric_value(stats, ("mlb::seasonAdvanced::hitting::hardHitPercentage",)))
        line_drive_pct = _ratio_or_none(_first_metric_value(stats, ("mlb::seasonAdvanced::hitting::lineDrivePercentage",)))

        k_pct = _clip(k_pct if k_pct is not None else 0.225, 0.14, 0.33)
        bb_pct = _clip(bb_pct if bb_pct is not None else 0.085, 0.05, 0.14)
        hard_hit_pct = _clip(hard_hit_pct if hard_hit_pct is not None else 0.375, 0.28, 0.50)
        line_drive_pct = _clip(line_drive_pct if line_drive_pct is not None else 0.210, 0.15, 0.30)

        xwobacon_raw = _first_metric_value(stats, XWOBA_CON_CANDIDATE_KEYS)
        if xwobacon_raw is not None:
            if xwobacon_raw > 1.5:
                xwobacon_raw = xwobacon_raw / 100.0
            if xwobacon_raw > 1.0:
                xwobacon_raw = xwobacon_raw / 1000.0

        xwobacon_formula = (
            0.33
            + 0.40 * (ops - 0.720)
            + 0.12 * (hard_hit_pct - 0.375)
            + 0.08 * (line_drive_pct - 0.210)
            - 0.06 * (k_pct - 0.225)
            + 0.06 * (bb_pct - 0.085)
        )
        xwobacon_proxy = _clip(float(xwobacon_raw) if xwobacon_raw is not None else xwobacon_formula, 0.220, 0.460)

        k9 = _first_metric_value(
            stats,
            (
                "mlb::seasonAdvanced::pitching::strikeoutsPer9",
                "mlb::season::pitching::strikeoutsPer9Inn",
            ),
        )
        bb9 = _first_metric_value(
            stats,
            (
                "mlb::seasonAdvanced::pitching::baseOnBallsPer9",
                "mlb::seasonAdvanced::pitching::walksPer9Inn",
            ),
        )
        hr9 = _first_metric_value(
            stats,
            (
                "mlb::seasonAdvanced::pitching::homeRunsPer9",
                "mlb::season::pitching::homeRunsPer9",
            ),
        )
        fly_ball_pct = _ratio_or_none(_first_metric_value(stats, ("mlb::seasonAdvanced::pitching::flyBallPercentage",)))
        whiff_pct = _ratio_or_none(_first_metric_value(stats, ("mlb::seasonAdvanced::pitching::whiffPercentage",)))
        k_minus_bb_pct = _ratio_or_none(_first_metric_value(stats, ("mlb::seasonAdvanced::pitching::strikeoutsMinusWalksPercentage",)))

        k9 = _clip(k9 if k9 is not None else 8.6, 5.0, 12.5)
        bb9 = _clip(bb9 if bb9 is not None else 3.1, 1.5, 5.6)
        hr9 = _clip(hr9 if hr9 is not None else 1.12, 0.45, 2.20)
        fly_ball_pct = _clip(fly_ball_pct if fly_ball_pct is not None else 0.35, 0.20, 0.55)
        whiff_pct = _clip(whiff_pct if whiff_pct is not None else 0.255, 0.15, 0.40)
        k_minus_bb_pct = _clip(k_minus_bb_pct if k_minus_bb_pct is not None else 0.125, 0.03, 0.25)

        expected_hr9 = _clip(0.62 * hr9 + 0.38 * 1.12 + 0.60 * (fly_ball_pct - 0.35), 0.55, 2.20)

        xfip_raw = _first_metric_value(stats, XFIP_CANDIDATE_KEYS)
        if xfip_raw is not None and xfip_raw > 12:
            xfip_raw = xfip_raw / 10.0

        xfip_formula = 3.20 + 0.30 * (bb9 - 3.1) - 0.18 * (k9 - 8.6) + 0.30 * (expected_hr9 - 1.12)
        xfip_proxy = _clip(float(xfip_raw) if xfip_raw is not None else xfip_formula, 2.20, 6.80)

        xwobacon_quality = _clip((xwobacon_proxy - 0.220) / (0.460 - 0.220), 0.0, 1.0)
        xfip_quality = _clip((6.80 - xfip_proxy) / (6.80 - 2.20), 0.0, 1.0)
        kbb_quality = _clip((k_minus_bb_pct - 0.05) / 0.18, 0.0, 1.0)
        hr9_quality = _clip((1.55 - hr9) / 0.90, 0.0, 1.0)

        offense_expected_quality = _clip(
            0.50 * xwobacon_quality
            + 0.14 * hard_hit_pct
            + 0.10 * line_drive_pct
            + 0.08 * bb_pct
            + 0.08 * (1.0 - k_pct)
            + 0.10 * leverage_offense_quality,
            0.0,
            1.0,
        )

        pitching_expected_quality = _clip(
            0.47 * xfip_quality
            + 0.18 * kbb_quality
            + 0.13 * hr9_quality
            + 0.12 * whiff_pct
            + 0.10 * leverage_pitching_quality,
            0.0,
            1.0,
        )

        rows[str(team_name)] = {
            "xwobacon_proxy": xwobacon_proxy,
            "xfip_proxy": xfip_proxy,
            "offense_expected_quality": offense_expected_quality,
            "pitching_expected_quality": pitching_expected_quality,
            "hitting_ops_risp": hitting_ops_risp,
            "hitting_ops_risp2": hitting_ops_risp2,
            "hitting_ops_late_close": hitting_ops_late_close,
            "pitching_ops_allowed": pitching_ops_allowed,
            "pitching_ops_risp_allowed": pitching_ops_risp_allowed,
            "pitching_ops_risp2_allowed": pitching_ops_risp2_allowed,
            "pitching_ops_late_close_allowed": pitching_ops_late_close_allowed,
            "leverage_offense_quality": leverage_offense_quality,
            "leverage_pitching_quality": leverage_pitching_quality,
            "leverage_net_quality": leverage_net_quality,
            "clutch_index": clutch_index,
        }

    if rows:
        offense_mean = float(np.mean([r["offense_expected_quality"] for r in rows.values()]))
        pitching_mean = float(np.mean([r["pitching_expected_quality"] for r in rows.values()]))
    else:
        offense_mean = 0.5
        pitching_mean = 0.5

    for row in rows.values():
        row["offense_expected_vs_league"] = row["offense_expected_quality"] - offense_mean
        row["pitching_expected_vs_league"] = row["pitching_expected_quality"] - pitching_mean

    ADVANCED_TEAM_CACHE[int(season)] = rows
    return rows


def _travel_context_adjustment(
    home_state: TeamState,
    away_state: TeamState,
    target_tz_offset: float,
    home_rest: float,
    away_rest: float,
) -> tuple[float, dict[str, float]]:
    home_prev = home_state.last_timezone_offset if home_state.last_timezone_offset is not None else target_tz_offset
    away_prev = away_state.last_timezone_offset if away_state.last_timezone_offset is not None else target_tz_offset

    home_shift = abs(float(target_tz_offset) - float(home_prev))
    away_shift = abs(float(target_tz_offset) - float(away_prev))

    shift_edge = (away_shift - home_shift) * 0.012
    rest_edge = max(-2.0, min(2.0, home_rest - away_rest)) * 0.005

    total = max(-0.07, min(0.07, shift_edge + rest_edge))
    return total, {
        "travel_adjustment": round(total, 4),
        "travel_tz_shift_edge": round(shift_edge, 4),
        "travel_rest_edge": round(rest_edge, 4),
        "home_tz_shift": round(home_shift, 3),
        "away_tz_shift": round(away_shift, 3),
    }


def _lineup_ops_value(player: dict[str, Any]) -> float | None:
    season_stats = player.get("seasonStats", {}) or {}
    batting = season_stats.get("batting", {}) or {}
    ops = batting.get("ops")
    if ops is None:
        return None
    value = _to_float(ops, float("nan"))
    if math.isnan(value):
        return None
    return value


def _fetch_game_lineup_context(game_pk: int | None) -> dict[str, Any]:
    if game_pk is None:
        return {}

    key = int(game_pk)
    cached = LINEUP_CACHE.get(key)
    if cached is not None:
        return cached

    output = {
        "home_confirmed": False,
        "away_confirmed": False,
        "home_lineup_ops": None,
        "away_lineup_ops": None,
        "home_batting_order_count": 0,
        "away_batting_order_count": 0,
    }

    try:
        response = requests.get(
            f"{BASE_API_URL}/game/{key}/boxscore",
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()

        for side in ("home", "away"):
            team_block = payload.get("teams", {}).get(side, {}) or {}
            order = team_block.get("battingOrder") or []
            players = team_block.get("players", {}) or {}

            ops_values: list[float] = []
            for player_id in order[:9]:
                pid = str(player_id)
                player = players.get(f"ID{pid}") or players.get(pid) or {}
                ops_val = _lineup_ops_value(player)
                if ops_val is not None:
                    ops_values.append(ops_val)

            output[f"{side}_batting_order_count"] = len(order)
            output[f"{side}_confirmed"] = len(order) >= 9
            output[f"{side}_lineup_ops"] = float(np.mean(ops_values)) if ops_values else None
    except Exception:
        pass

    LINEUP_CACHE[key] = output
    return output


def _lineup_context_adjustment(lineup_context: dict[str, Any]) -> tuple[float, dict[str, float]]:
    home_ops = lineup_context.get("home_lineup_ops")
    away_ops = lineup_context.get("away_lineup_ops")
    home_confirmed = bool(lineup_context.get("home_confirmed"))
    away_confirmed = bool(lineup_context.get("away_confirmed"))

    ops_edge = 0.0
    if home_ops is not None and away_ops is not None:
        ops_edge = (float(home_ops) - float(away_ops)) * 0.08

    confirmed_edge = (1 if home_confirmed else 0) - (1 if away_confirmed else 0)
    confirmed_adj = 0.01 * float(confirmed_edge)

    total = max(-0.05, min(0.05, ops_edge + confirmed_adj))
    return total, {
        "lineup_adjustment": round(total, 4),
        "lineup_ops_edge": round(ops_edge, 4),
        "lineup_confirmed_edge": round(confirmed_adj, 4),
        "home_lineup_confirmed": home_confirmed,
        "away_lineup_confirmed": away_confirmed,
        "home_lineup_ops": round(float(home_ops), 4) if home_ops is not None else None,
        "away_lineup_ops": round(float(away_ops), 4) if away_ops is not None else None,
    }



def _lineup_health_adjustment(
    lineup_context: dict[str, Any],
    home_baseline_ops: float | None,
    away_baseline_ops: float | None,
) -> tuple[float, dict[str, float | None]]:
    home_ops = lineup_context.get("home_lineup_ops")
    away_ops = lineup_context.get("away_lineup_ops")
    home_confirmed = bool(lineup_context.get("home_confirmed"))
    away_confirmed = bool(lineup_context.get("away_confirmed"))

    home_health = 1.0
    away_health = 1.0

    if home_ops is not None and home_baseline_ops is not None and float(home_baseline_ops) > 0:
        home_health = max(0.84, min(1.16, float(home_ops) / float(home_baseline_ops)))
    if away_ops is not None and away_baseline_ops is not None and float(away_baseline_ops) > 0:
        away_health = max(0.84, min(1.16, float(away_ops) / float(away_baseline_ops)))

    health_edge = (home_health - away_health) * 0.09
    confirmed_edge = ((1.0 if home_confirmed else 0.0) - (1.0 if away_confirmed else 0.0)) * 0.004

    total = max(-0.04, min(0.04, health_edge + confirmed_edge))
    return total, {
        "lineup_health_adjustment": round(total, 4),
        "lineup_health_edge": round(health_edge, 4),
        "lineup_health_confirmed_edge": round(confirmed_edge, 4),
        "lineup_health_home_score": round(home_health * 100.0, 1),
        "lineup_health_away_score": round(away_health * 100.0, 1),
        "lineup_baseline_home_ops": round(float(home_baseline_ops), 4) if home_baseline_ops is not None else None,
        "lineup_baseline_away_ops": round(float(away_baseline_ops), 4) if away_baseline_ops is not None else None,
    }


def _parse_date(text: str | None) -> dt.date | None:
    if not text:
        return None
    try:
        return dt.date.fromisoformat(str(text)[:10])
    except ValueError:
        return None


def _fetch_schedule_range(start_date: dt.date, end_date: dt.date) -> list[dict[str, Any]]:
    response = requests.get(
        f"{BASE_API_URL}/schedule",
        params={
            "sportId": 1,
            "startDate": start_date.isoformat(),
            "endDate": end_date.isoformat(),
            "hydrate": "team,linescore,probablePitcher",
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    payload = response.json()
    games: list[dict[str, Any]] = []

    for day in payload.get("dates", []):
        official_date = day.get("date")
        for game in day.get("games", []):
            if game.get("gameType") != REGULAR_SEASON_GAME_TYPE:
                continue

            home = game.get("teams", {}).get("home", {})
            away = game.get("teams", {}).get("away", {})

            home_pitcher = home.get("probablePitcher", {}) or {}
            away_pitcher = away.get("probablePitcher", {}) or {}
            linescore = game.get("linescore", {}) or {}
            innings_count = len(linescore.get("innings") or [])

            games.append(
                {
                    "game_pk": game.get("gamePk"),
                    "official_date": official_date,
                    "game_date": game.get("gameDate"),
                    "state": game.get("status", {}).get("abstractGameState", "Unknown"),
                    "status": game.get("status", {}).get("detailedState", "Unknown"),
                    "home_id": home.get("team", {}).get("id"),
                    "away_id": away.get("team", {}).get("id"),
                    "home_score": home.get("score"),
                    "away_score": away.get("score"),
                    "home_probable_pitcher_id": home_pitcher.get("id"),
                    "home_probable_pitcher_name": home_pitcher.get("fullName"),
                    "away_probable_pitcher_id": away_pitcher.get("id"),
                    "away_probable_pitcher_name": away_pitcher.get("fullName"),
                    "innings_count": innings_count,
                    "venue_id": game.get("venue", {}).get("id"),
                }
            )

    games.sort(key=lambda g: (g.get("official_date") or "", g.get("game_date") or "", g.get("game_pk") or 0))
    return games

def _update_states_after_game(
    home_state: TeamState,
    away_state: TeamState,
    home_score: int,
    away_score: int,
    game_date: dt.date,
    innings_count: int | None = None,
    venue_tz_offset: float | None = None,
) -> None:
    home_win = 1.0 if home_score > away_score else 0.0

    expected_home = _elo_home_win_prob(home_state.elo, away_state.elo)
    margin = abs(home_score - away_score)
    mov_multiplier = ((margin + 1.0) ** 0.8) / (7.5 + 0.006 * abs(home_state.elo - away_state.elo))
    k_factor = ELO_K * mov_multiplier
    delta = k_factor * (home_win - expected_home)

    home_state.elo += delta
    away_state.elo -= delta

    home_state.runs_for += home_score
    home_state.runs_against += away_score
    away_state.runs_for += away_score
    away_state.runs_against += home_score

    home_state.runs_allowed_recent.append(away_score)
    away_state.runs_allowed_recent.append(home_score)
    home_state.runs_for_recent.append(home_score)
    home_state.runs_against_recent.append(away_score)
    away_state.runs_for_recent.append(away_score)
    away_state.runs_against_recent.append(home_score)

    total_innings = float(max(9, int(innings_count or 9)))
    bullpen_outs = max(0.0, (total_innings - 5.0) * 3.0)
    home_state.bullpen_outs_recent.append(bullpen_outs)
    away_state.bullpen_outs_recent.append(bullpen_outs)
    home_state.game_dates_recent.append(game_date)
    away_state.game_dates_recent.append(game_date)

    if home_win == 1.0:
        home_state.wins += 1
        away_state.losses += 1
        home_state.home_wins += 1
        away_state.away_losses += 1
        home_state.recent.append(1)
        away_state.recent.append(0)
    else:
        away_state.wins += 1
        home_state.losses += 1
        away_state.away_wins += 1
        home_state.home_losses += 1
        away_state.recent.append(1)
        home_state.recent.append(0)

    if venue_tz_offset is not None:
        home_state.last_timezone_offset = float(venue_tz_offset)
        away_state.last_timezone_offset = float(venue_tz_offset)

    home_state.last_game_date = game_date
    away_state.last_game_date = game_date


def _build_backtest_dataset(season: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    team_lookup = fetch_team_name_lookup(season)
    team_context = _fetch_team_context_lookup(season)
    games = _fetch_schedule_range(dt.date(season, 3, 1), dt.date(season, 11, 15))

    states: dict[str, TeamState] = {team_name: TeamState() for team_name in team_lookup.values()}
    X_rows: list[list[float]] = []
    y_rows: list[int] = []
    elo_rows: list[float] = []
    meta_rows: list[dict[str, Any]] = []

    for game in games:
        if game.get("state") not in FINAL_STATES:
            continue

        game_date = _parse_date(game.get("official_date"))
        if game_date is None:
            continue

        home_id = game.get("home_id")
        away_id = game.get("away_id")
        if home_id is None or away_id is None:
            continue

        home_team = team_lookup.get(int(home_id))
        away_team = team_lookup.get(int(away_id))
        if not home_team or not away_team:
            continue

        home_state = states.setdefault(home_team, TeamState())
        away_state = states.setdefault(away_team, TeamState())

        home_rest = _rest_days(home_state, game_date)
        away_rest = _rest_days(away_state, game_date)
        features = _build_feature_vector(home_state, away_state, home_rest=home_rest, away_rest=away_rest)
        elo_home_prob = _elo_home_win_prob(home_state.elo, away_state.elo)

        home_score = game.get("home_score")
        away_score = game.get("away_score")
        if home_score is None or away_score is None:
            continue

        home_score = int(home_score)
        away_score = int(away_score)
        if home_score == away_score:
            continue

        home_win = 1 if home_score > away_score else 0

        X_rows.append(features)
        y_rows.append(home_win)
        elo_rows.append(elo_home_prob)
        meta_rows.append(
            {
                "game_pk": game.get("game_pk"),
                "official_date": game.get("official_date"),
                "away_team": away_team,
                "home_team": home_team,
                "away_score": away_score,
                "home_score": home_score,
            }
        )

        home_context = team_context.get(int(home_id), {})
        venue_tz_offset = _to_float(home_context.get("tz_offset"), 0.0)

        _update_states_after_game(
            home_state,
            away_state,
            home_score=home_score,
            away_score=away_score,
            game_date=game_date,
            innings_count=game.get("innings_count"),
            venue_tz_offset=venue_tz_offset,
        )

    if not X_rows:
        raise RuntimeError(f"No finalized regular-season games found for season {season}")

    return (
        np.asarray(X_rows, dtype=float),
        np.asarray(y_rows, dtype=float),
        np.asarray(elo_rows, dtype=float),
        meta_rows,
    )


def _fit_standardization(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if X.shape[1] <= 1:
        return np.asarray([]), np.asarray([])

    mean = X[:, 1:].mean(axis=0)
    std = X[:, 1:].std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def _apply_standardization(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    X_scaled = X.copy()
    if X.shape[1] > 1 and len(mean) == X.shape[1] - 1 and len(std) == X.shape[1] - 1:
        X_scaled[:, 1:] = (X_scaled[:, 1:] - mean) / std
    return X_scaled


def _fit_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.05,
    epochs: int = 3500,
    regularization: float = 2e-4,
) -> np.ndarray:
    n_samples, n_features = X.shape
    weights = np.zeros(n_features, dtype=float)

    reg_mask = np.ones(n_features, dtype=float)
    reg_mask[0] = 0.0

    for _ in range(epochs):
        preds = _sigmoid(X @ weights)
        gradient = (X.T @ (preds - y)) / max(n_samples, 1)
        gradient += regularization * weights * reg_mask
        weights -= learning_rate * gradient

    return weights


def _metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    probs = np.clip(probs, 1e-6, 1 - 1e-6)
    preds = (probs >= 0.5).astype(float)

    accuracy = float(np.mean(preds == y_true))
    log_loss = float(-np.mean(y_true * np.log(probs) + (1.0 - y_true) * np.log(1.0 - probs)))
    brier = float(np.mean((probs - y_true) ** 2))

    return {
        "accuracy": round(accuracy, 4),
        "log_loss": round(log_loss, 4),
        "brier_score": round(brier, 4),
    }


def _build_calibration_diagnostics(
    y_true: np.ndarray,
    probs: np.ndarray,
    num_bins: int = 10,
) -> dict[str, Any]:
    probs_arr = np.asarray(probs, dtype=float)
    y_arr = np.asarray(y_true, dtype=float)

    if len(probs_arr) == 0 or len(y_arr) == 0:
        return {
            "sample_size": 0,
            "base_rate": None,
            "ece": None,
            "mce": None,
            "reliability_curve": [],
            "brier_decomposition": {},
        }

    clipped = np.clip(probs_arr, 1e-6, 1 - 1e-6)
    edges = np.linspace(0.0, 1.0, max(2, int(num_bins)) + 1)

    curve: list[dict[str, Any]] = []
    weighted_abs_gap = 0.0
    max_gap = 0.0

    base_rate = float(np.mean(y_arr))
    total = float(len(clipped))
    reliability = 0.0
    resolution = 0.0

    for idx in range(len(edges) - 1):
        p_low = float(edges[idx])
        p_high = float(edges[idx + 1])

        if idx == len(edges) - 2:
            mask = (clipped >= p_low) & (clipped <= p_high)
        else:
            mask = (clipped >= p_low) & (clipped < p_high)

        count = int(np.sum(mask))
        if count <= 0:
            continue

        bin_probs = clipped[mask]
        bin_true = y_arr[mask]

        avg_pred = float(np.mean(bin_probs))
        observed = float(np.mean(bin_true))
        gap = float(avg_pred - observed)
        abs_gap = abs(gap)
        weight = count / total
        std_error = math.sqrt(max(1e-9, observed * (1.0 - observed)) / max(1, count))
        band_half = min(0.22, max(0.03, abs_gap, 1.96 * std_error))

        weighted_abs_gap += weight * abs_gap
        max_gap = max(max_gap, abs_gap)
        reliability += weight * ((avg_pred - observed) ** 2)
        resolution += weight * ((observed - base_rate) ** 2)

        curve.append(
            {
                "bin_index": idx,
                "p_low": round(p_low, 3),
                "p_high": round(p_high, 3),
                "p_center": round((p_low + p_high) / 2.0, 3),
                "count": count,
                "avg_pred": round(avg_pred, 4),
                "observed": round(observed, 4),
                "gap": round(gap, 4),
                "abs_gap": round(abs_gap, 4),
                "std_error": round(std_error, 4),
                "band_half": round(float(band_half), 4),
            }
        )

    brier_score = float(np.mean((clipped - y_arr) ** 2))
    uncertainty = float(base_rate * (1.0 - base_rate))
    brier_from_components = float(reliability - resolution + uncertainty)

    return {
        "sample_size": int(len(clipped)),
        "base_rate": round(base_rate, 4),
        "ece": round(float(weighted_abs_gap), 4),
        "mce": round(float(max_gap), 4),
        "reliability_curve": curve,
        "brier_decomposition": {
            "brier_score": round(brier_score, 4),
            "reliability": round(float(reliability), 4),
            "resolution": round(float(resolution), 4),
            "uncertainty": round(float(uncertainty), 4),
            "brier_from_components": round(float(brier_from_components), 4),
        },
    }


def _estimate_prediction_uncertainty(
    home_prob: float,
    calibration_test: dict[str, Any] | None,
    market_weight: float = 0.0,
) -> dict[str, Any]:
    p = _clamp_prob(float(home_prob), lower=0.01, upper=0.99)
    curve = list((calibration_test or {}).get("reliability_curve") or [])

    selected = None
    if curve:
        for row in curve:
            lo = _to_float(row.get("p_low"), 0.0)
            hi = _to_float(row.get("p_high"), 1.0)
            if p >= lo and p <= hi + 1e-9:
                selected = row
                break
        if selected is None:
            selected = min(
                curve,
                key=lambda row: abs(_to_float(row.get("p_center"), 0.5) - p),
            )

    if selected is None:
        count = 0
        abs_gap = 0.06
        half = 0.095
    else:
        count = _to_int(selected.get("count"), 0)
        abs_gap = abs(_to_float(selected.get("gap"), 0.0))
        half = max(
            _to_float(selected.get("band_half"), 0.0),
            _to_float(selected.get("std_error"), 0.0) * 1.96,
            abs_gap,
            0.03,
        )

    edge_distance = abs(p - 0.5)
    coinflip_pressure = max(0.0, (0.10 - edge_distance) / 0.10)
    sample_penalty = 0.018 if count < 40 else (0.008 if count < 90 else 0.0)

    half = half + (0.02 * coinflip_pressure) + sample_penalty
    market_shrink = max(0.78, 1.0 - (0.45 * max(0.0, min(0.5, float(market_weight)))))
    half = float(np.clip(half * market_shrink, 0.03, 0.22))

    low = _clamp_prob(p - half, lower=0.01, upper=0.99)
    high = _clamp_prob(p + half, lower=0.01, upper=0.99)

    if half >= 0.11 or (edge_distance < 0.055 and count < 90):
        level = "high"
    elif half >= 0.075 or edge_distance < 0.085:
        level = "medium"
    else:
        level = "low"

    high_uncertainty = bool(level == "high")
    if high_uncertainty:
        edge_multiplier = 0.55
    elif level == "medium":
        edge_multiplier = 0.8
    else:
        edge_multiplier = 1.0

    if count < 40:
        note = "Thin calibration sample"
    elif edge_distance < 0.06:
        note = "Near coin-flip range"
    elif abs_gap >= 0.08:
        note = "Large historical calibration gap"
    else:
        note = "Stable confidence region"

    return {
        "band_low": round(float(low), 4),
        "band_high": round(float(high), 4),
        "band_half": round(float(half), 4),
        "level": level,
        "high_uncertainty": high_uncertainty,
        "bin_count": int(count),
        "calibration_gap": round(float(abs_gap), 4),
        "edge_multiplier": float(edge_multiplier),
        "note": note,
    }


def _scale_adjustment_fields(parts: dict[str, Any], multiplier: float, keys: tuple[str, ...]) -> dict[str, Any]:
    out = dict(parts or {})
    m = float(np.clip(float(multiplier), PREGAME_CONTEXT_MIN_MULTIPLIER, 1.0))

    for key in keys:
        value = out.get(key)
        if isinstance(value, (int, float, np.floating)):
            out[key] = round(float(value) * m, 4)

    return out


def _pregame_context_gates(
    lineup_context: dict[str, Any],
    home_pitcher: dict[str, Any],
    away_pitcher: dict[str, Any],
) -> dict[str, Any]:
    home_lineup_count = _to_int(lineup_context.get("home_batting_order_count"), 0)
    away_lineup_count = _to_int(lineup_context.get("away_batting_order_count"), 0)
    home_lineup_confirmed = bool(lineup_context.get("home_confirmed"))
    away_lineup_confirmed = bool(lineup_context.get("away_confirmed"))

    home_lineup_score = 1.0 if home_lineup_confirmed else max(0.0, min(0.9, home_lineup_count / 9.0))
    away_lineup_score = 1.0 if away_lineup_confirmed else max(0.0, min(0.9, away_lineup_count / 9.0))
    lineup_readiness = max(0.0, min(1.0, 0.5 * (home_lineup_score + away_lineup_score)))

    home_starter_known = bool(home_pitcher and home_pitcher.get("id"))
    away_starter_known = bool(away_pitcher and away_pitcher.get("id"))
    home_rel = max(0.0, min(1.0, _to_float((home_pitcher or {}).get("reliability"), 0.0)))
    away_rel = max(0.0, min(1.0, _to_float((away_pitcher or {}).get("reliability"), 0.0)))

    home_starter_score = (0.25 + (0.75 * home_rel)) if home_starter_known else 0.2
    away_starter_score = (0.25 + (0.75 * away_rel)) if away_starter_known else 0.2
    starter_readiness = max(0.0, min(1.0, 0.5 * (home_starter_score + away_starter_score)))

    overall_readiness = max(0.0, min(1.0, (0.58 * starter_readiness) + (0.42 * lineup_readiness)))

    lineup_multiplier = float(np.clip(0.45 + (0.55 * lineup_readiness), PREGAME_CONTEXT_MIN_MULTIPLIER, 1.0))
    starter_multiplier = float(np.clip(0.5 + (0.5 * starter_readiness), PREGAME_CONTEXT_MIN_MULTIPLIER, 1.0))
    split_multiplier = float(np.clip(0.55 + (0.45 * min(lineup_readiness, starter_readiness)), PREGAME_CONTEXT_MIN_MULTIPLIER, 1.0))
    overall_multiplier = float(np.clip(0.5 + (0.5 * overall_readiness), PREGAME_CONTEXT_MIN_MULTIPLIER, 1.0))

    return {
        "overall_multiplier": overall_multiplier,
        "lineup_multiplier": lineup_multiplier,
        "starter_multiplier": starter_multiplier,
        "split_multiplier": split_multiplier,
        "home_lineup_ready": bool(home_lineup_confirmed or home_lineup_count >= 7),
        "away_lineup_ready": bool(away_lineup_confirmed or away_lineup_count >= 7),
        "home_starter_ready": bool(home_starter_known),
        "away_starter_ready": bool(away_starter_known),
        "lineup_readiness": round(float(lineup_readiness), 4),
        "starter_readiness": round(float(starter_readiness), 4),
        "overall_readiness": round(float(overall_readiness), 4),
    }


def _compute_market_blend_weight(
    *,
    pre_market_home_prob: float,
    market_home_prob: float | None,
    market_book_count: int,
    market_last_update: str | None,
    reference_date: dt.date,
    pre_market_uncertainty_level: str,
) -> tuple[float, dict[str, Any]]:
    if market_home_prob is None:
        return 0.0, {
            "policy": "dynamic_v1",
            "reason": "market_missing",
            "base": 0.0,
            "book_factor": 0.0,
            "freshness_factor": 0.0,
            "uncertainty_bonus": 0.0,
            "disagreement_bonus": 0.0,
            "line_age_hours": None,
        }

    confidence = abs(float(pre_market_home_prob) - 0.5)
    disagreement = abs(float(pre_market_home_prob) - float(market_home_prob))

    if confidence < 0.04:
        base = 0.34
    elif confidence < 0.08:
        base = 0.29
    else:
        base = 0.24

    uncertainty_bonus = 0.0
    if str(pre_market_uncertainty_level).lower() == "high":
        uncertainty_bonus = 0.08
    elif str(pre_market_uncertainty_level).lower() == "medium":
        uncertainty_bonus = 0.04

    disagreement_bonus = min(0.08, disagreement * 0.5)

    books = max(0, int(market_book_count))
    book_factor = float(np.clip(0.6 + (min(books, 14) / 20.0), 0.6, 1.25))

    age_hours = None
    freshness_factor = 0.9
    last_update_dt = _parse_iso_utc_timestamp(market_last_update)
    if last_update_dt is not None:
        now_utc = dt.datetime.now(dt.timezone.utc)
        age_hours = max(0.0, (now_utc - last_update_dt).total_seconds() / 3600.0)
        if age_hours <= 2.0:
            freshness_factor = 1.0
        elif age_hours <= 6.0:
            freshness_factor = 0.94
        elif age_hours <= 12.0:
            freshness_factor = 0.85
        else:
            freshness_factor = 0.74
    else:
        ref_cutoff = dt.datetime.combine(reference_date, dt.time(hour=8, tzinfo=dt.timezone.utc))
        if dt.datetime.now(dt.timezone.utc) - ref_cutoff > dt.timedelta(hours=18):
            freshness_factor = 0.8

    weight_raw = (base + uncertainty_bonus + disagreement_bonus) * book_factor * freshness_factor
    min_weight = MARKET_BLEND_MIN_WEIGHT if books >= 3 else max(0.10, MARKET_BLEND_MIN_WEIGHT - 0.04)
    market_weight = float(np.clip(weight_raw, min_weight, MARKET_BLEND_MAX_WEIGHT))

    reason = "balanced"
    if uncertainty_bonus >= 0.08:
        reason = "high_uncertainty"
    elif disagreement_bonus >= 0.05:
        reason = "strong_disagreement"
    elif books < 3:
        reason = "thin_market"
    elif freshness_factor < 0.82:
        reason = "stale_market"

    return market_weight, {
        "policy": "dynamic_v1",
        "reason": reason,
        "base": round(float(base), 4),
        "book_factor": round(float(book_factor), 4),
        "freshness_factor": round(float(freshness_factor), 4),
        "uncertainty_bonus": round(float(uncertainty_bonus), 4),
        "disagreement_bonus": round(float(disagreement_bonus), 4),
        "line_age_hours": round(float(age_hours), 2) if age_hours is not None else None,
    }


def _build_drift_monitor(
    train_metrics: dict[str, Any],
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    rolling_cv: dict[str, Any],
) -> dict[str, Any]:
    def _metric(metrics: dict[str, Any], key: str) -> float | None:
        value = _to_float(metrics.get(key), float("nan"))
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)

    train_acc = _metric(train_metrics, "accuracy")
    val_acc = _metric(validation_metrics, "accuracy")
    test_acc = _metric(test_metrics, "accuracy")
    train_log_loss = _metric(train_metrics, "log_loss")
    val_log_loss = _metric(validation_metrics, "log_loss")
    test_log_loss = _metric(test_metrics, "log_loss")

    folds = list((rolling_cv or {}).get("folds") or [])
    fold_acc = [
        _to_float(row.get("accuracy"), float("nan"))
        for row in folds
        if not math.isnan(_to_float(row.get("accuracy"), float("nan")))
    ]
    fold_log = [
        _to_float(row.get("log_loss"), float("nan"))
        for row in folds
        if not math.isnan(_to_float(row.get("log_loss"), float("nan")))
    ]

    cv_mean = (rolling_cv or {}).get("mean") or {}
    cv_acc = _metric(cv_mean, "accuracy")
    cv_log_loss = _metric(cv_mean, "log_loss")

    acc_gap_train_test = (train_acc - test_acc) if train_acc is not None and test_acc is not None else None
    acc_gap_val_test = (val_acc - test_acc) if val_acc is not None and test_acc is not None else None
    log_gap_train_test = (test_log_loss - train_log_loss) if train_log_loss is not None and test_log_loss is not None else None
    log_gap_val_test = (test_log_loss - val_log_loss) if val_log_loss is not None and test_log_loss is not None else None
    cv_gap_log_loss = (test_log_loss - cv_log_loss) if cv_log_loss is not None and test_log_loss is not None else None

    overfit_index = 0.0
    if log_gap_train_test is not None and log_gap_train_test > 0:
        overfit_index += min(0.45, log_gap_train_test / 0.12)
    if log_gap_val_test is not None and log_gap_val_test > 0:
        overfit_index += min(0.35, log_gap_val_test / 0.08)
    if acc_gap_train_test is not None and acc_gap_train_test > 0:
        overfit_index += min(0.20, acc_gap_train_test / 0.12)

    if len(fold_acc) >= 2:
        overfit_index += min(0.15, float(np.std(fold_acc, ddof=0)) / 0.05)

    overfit_index = float(np.clip(overfit_index, 0.0, 1.0))

    status = "stable"
    recommendation = "Generalization looks stable."
    if overfit_index >= 0.68:
        status = "high_risk"
        recommendation = "High drift risk: increase regularization and reduce feature/adjustment sensitivity."
    elif overfit_index >= 0.4:
        status = "watch"
        recommendation = "Monitor drift: keep retraining cadence and avoid adding high-variance features."

    return {
        "status": status,
        "overfit_index": round(float(overfit_index), 4),
        "recommendation": recommendation,
        "acc_gap_train_test": round(float(acc_gap_train_test), 4) if acc_gap_train_test is not None else None,
        "acc_gap_val_test": round(float(acc_gap_val_test), 4) if acc_gap_val_test is not None else None,
        "log_loss_gap_train_test": round(float(log_gap_train_test), 4) if log_gap_train_test is not None else None,
        "log_loss_gap_val_test": round(float(log_gap_val_test), 4) if log_gap_val_test is not None else None,
        "cv_gap_log_loss": round(float(cv_gap_log_loss), 4) if cv_gap_log_loss is not None else None,
        "cv_accuracy_std": round(float(np.std(fold_acc, ddof=0)), 4) if len(fold_acc) >= 2 else None,
        "cv_log_loss_std": round(float(np.std(fold_log, ddof=0)), 4) if len(fold_log) >= 2 else None,
        "cv_fold_count": len(folds),
        "cv_mean_accuracy": round(float(cv_acc), 4) if cv_acc is not None else None,
        "cv_mean_log_loss": round(float(cv_log_loss), 4) if cv_log_loss is not None else None,
    }


def _optimize_blend_weight(model_probs: np.ndarray, elo_probs: np.ndarray, y_true: np.ndarray) -> float:
    best_weight = 0.75
    best_loss = float("inf")

    for weight in np.linspace(0.5, 0.9, 9):
        blended = (weight * model_probs) + ((1.0 - weight) * elo_probs)
        loss = _metrics(y_true, blended)["log_loss"]
        if loss < best_loss:
            best_loss = loss
            best_weight = float(weight)

    return best_weight


def _fit_platt_scaler(base_probs: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    if len(base_probs) < 30:
        return 0.0, 1.0

    z = _logit(base_probs)
    X_cal = np.column_stack([np.ones(len(z), dtype=float), z])
    weights = _fit_logistic_regression(
        X_cal,
        y_true,
        learning_rate=0.03,
        epochs=2500,
        regularization=8e-4,
    )
    return float(weights[0]), float(weights[1])


def _apply_platt_scaler(base_probs: np.ndarray, intercept: float, slope: float) -> np.ndarray:
    z = _logit(base_probs)
    return _sigmoid(intercept + slope * z)


def _predict_with_model_bundle(bundle: dict[str, Any], X: np.ndarray, elo_probs: np.ndarray) -> np.ndarray:
    mean = np.asarray(bundle.get("feature_mean", []), dtype=float)
    std = np.asarray(bundle.get("feature_std", []), dtype=float)
    weights = np.asarray(bundle.get("weights", []), dtype=float)

    X_scaled = _apply_standardization(X, mean=mean, std=std)
    model_probs = _sigmoid(X_scaled @ weights)

    blend_weight = float(bundle.get("blend_weight", 0.75))
    blended = (blend_weight * model_probs) + ((1.0 - blend_weight) * elo_probs)

    cal = bundle.get("calibration", {})
    intercept = float(cal.get("intercept", 0.0))
    slope = float(cal.get("slope", 1.0))
    calibrated = _apply_platt_scaler(blended, intercept=intercept, slope=slope)

    return np.clip(calibrated, 0.01, 0.99)


def _train_bundle(
    X_train: np.ndarray,
    y_train: np.ndarray,
    elo_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    elo_val: np.ndarray,
) -> dict[str, Any]:
    mean, std = _fit_standardization(X_train)
    X_train_scaled = _apply_standardization(X_train, mean=mean, std=std)
    X_val_scaled = _apply_standardization(X_val, mean=mean, std=std)

    weights = _fit_logistic_regression(X_train_scaled, y_train)

    train_model_probs = _sigmoid(X_train_scaled @ weights)
    val_model_probs = _sigmoid(X_val_scaled @ weights)

    blend_weight = _optimize_blend_weight(val_model_probs, elo_val, y_val)

    train_blended = (blend_weight * train_model_probs) + ((1.0 - blend_weight) * elo_train)
    val_blended = (blend_weight * val_model_probs) + ((1.0 - blend_weight) * elo_val)

    cal_intercept, cal_slope = _fit_platt_scaler(val_blended, y_val)

    bundle = {
        "weights": [float(w) for w in weights.tolist()],
        "feature_mean": [float(v) for v in mean.tolist()],
        "feature_std": [float(v) for v in std.tolist()],
        "blend_weight": float(blend_weight),
        "calibration": {
            "intercept": float(cal_intercept),
            "slope": float(cal_slope),
        },
        "train_probs": _apply_platt_scaler(train_blended, intercept=cal_intercept, slope=cal_slope),
        "val_probs": _apply_platt_scaler(val_blended, intercept=cal_intercept, slope=cal_slope),
    }
    return bundle


def _predict_with_tree_bundle(bundle: dict[str, Any], X: np.ndarray, elo_probs: np.ndarray) -> np.ndarray:
    model = bundle.get("tree_model")
    if model is None:
        return np.clip(np.asarray(elo_probs, dtype=float), 0.01, 0.99)

    model_probs = model.predict_proba(X)[:, 1]

    blend_weight = float(bundle.get("blend_weight", 0.7))
    blended = (blend_weight * model_probs) + ((1.0 - blend_weight) * elo_probs)

    cal = bundle.get("calibration", {})
    intercept = float(cal.get("intercept", 0.0))
    slope = float(cal.get("slope", 1.0))
    calibrated = _apply_platt_scaler(blended, intercept=intercept, slope=slope)

    return np.clip(calibrated, 0.01, 0.99)


def _train_tree_bundle(
    X_train: np.ndarray,
    y_train: np.ndarray,
    elo_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    elo_val: np.ndarray,
    compute_importance: bool = True,
) -> dict[str, Any]:
    tree_model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.045,
        max_iter=V4_TREE_MAX_ITER,
        max_leaf_nodes=31,
        max_depth=5,
        min_samples_leaf=20,
        l2_regularization=0.15,
        random_state=42,
    )
    tree_model.fit(X_train, y_train)

    train_model_probs = tree_model.predict_proba(X_train)[:, 1]
    val_model_probs = tree_model.predict_proba(X_val)[:, 1]

    blend_weight = _optimize_blend_weight(val_model_probs, elo_val, y_val)

    train_blended = (blend_weight * train_model_probs) + ((1.0 - blend_weight) * elo_train)
    val_blended = (blend_weight * val_model_probs) + ((1.0 - blend_weight) * elo_val)

    cal_intercept, cal_slope = _fit_platt_scaler(val_blended, y_val)

    sample_size = min(len(X_val), 420)
    sample_x = X_val[:sample_size]
    sample_y = y_val[:sample_size]
    if compute_importance and sample_size >= 60:
        try:
            perm = permutation_importance(
                tree_model,
                sample_x,
                sample_y,
                n_repeats=3,
                random_state=42,
                scoring="neg_log_loss",
            )
            importance_raw = [float(max(0.0, v)) for v in perm.importances_mean.tolist()]
        except Exception:
            importance_raw = [0.0] * X_train.shape[1]
    else:
        importance_raw = [0.0] * X_train.shape[1]

    bundle = {
        "tree_model": tree_model,
        "blend_weight": float(blend_weight),
        "calibration": {
            "intercept": float(cal_intercept),
            "slope": float(cal_slope),
        },
        "feature_importance_raw": importance_raw,
        "feature_mean": [float(v) for v in np.mean(X_train, axis=0).tolist()],
        "train_probs": _apply_platt_scaler(train_blended, intercept=cal_intercept, slope=cal_slope),
        "val_probs": _apply_platt_scaler(val_blended, intercept=cal_intercept, slope=cal_slope),
    }
    return bundle


def _feature_importance_tree_summary(bundle: dict[str, Any], feature_names: list[str]) -> list[dict[str, Any]]:
    raw = np.asarray(bundle.get("feature_importance_raw", []), dtype=float)
    if len(raw) != len(feature_names):
        return []

    total = float(np.sum(raw))
    if total <= 0:
        total = 1.0

    rows: list[dict[str, Any]] = []
    for i, feature_name in enumerate(feature_names):
        score = float(raw[i])
        rows.append(
            {
                "feature": feature_name,
                "coefficient": round(score, 6),
                "abs_coefficient": round(score, 6),
                "importance_pct": round((score / total) * 100.0, 2),
                "direction": "impact",
            }
        )

    rows.sort(key=lambda row: row["abs_coefficient"], reverse=True)
    return rows


def _feature_ablation_tree_summary(
    bundle: dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    elo_test: np.ndarray,
    feature_names: list[str],
) -> dict[str, Any]:
    if len(X_test) == 0:
        return {"baseline": {}, "rows": []}

    baseline_probs = _predict_with_tree_bundle(bundle, X_test, elo_test)
    baseline_metrics = _metrics(y_test, baseline_probs)

    feature_mean = np.asarray(bundle.get("feature_mean", []), dtype=float)
    rows: list[dict[str, Any]] = []

    for index, feature_name in enumerate(feature_names):
        X_variant = X_test.copy()
        if index < len(feature_mean):
            X_variant[:, index] = feature_mean[index]
        else:
            X_variant[:, index] = float(np.mean(X_test[:, index]))

        variant_probs = _predict_with_tree_bundle(bundle, X_variant, elo_test)
        variant_metrics = _metrics(y_test, variant_probs)

        rows.append(
            {
                "feature": feature_name,
                "accuracy": variant_metrics["accuracy"],
                "log_loss": variant_metrics["log_loss"],
                "brier_score": variant_metrics["brier_score"],
                "delta_accuracy": round(variant_metrics["accuracy"] - baseline_metrics["accuracy"], 4),
                "delta_log_loss": round(variant_metrics["log_loss"] - baseline_metrics["log_loss"], 4),
                "delta_brier": round(variant_metrics["brier_score"] - baseline_metrics["brier_score"], 4),
            }
        )

    rows.sort(key=lambda row: row["delta_log_loss"], reverse=True)
    return {"baseline": baseline_metrics, "rows": rows}


def _load_tree_model(model_path: str) -> Any | None:
    if not model_path:
        return None

    cached = MODEL_OBJECT_CACHE.get(model_path)
    if cached is not None:
        return cached

    path = Path(model_path)
    if not path.exists():
        return None

    try:
        model = joblib.load(path)
    except Exception:
        return None

    MODEL_OBJECT_CACHE[model_path] = model
    return model


def _split_indices(n_samples: int) -> tuple[int, int]:
    if n_samples < 220:
        raise RuntimeError("Not enough games for robust train/validation/test split")

    train_end = max(140, int(n_samples * 0.62))
    val_end = max(train_end + 40, int(n_samples * 0.82))

    train_end = min(train_end, n_samples - 80)
    val_end = min(val_end, n_samples - 30)

    if val_end <= train_end:
        val_end = min(n_samples - 30, train_end + 40)

    return train_end, val_end


def _rolling_cv_summary(X: np.ndarray, y: np.ndarray, elo_probs: np.ndarray) -> dict[str, Any]:
    n_samples = len(X)
    if n_samples < 500:
        return {"folds": [], "mean": {}}

    folds: list[dict[str, Any]] = []
    min_train = max(220, int(n_samples * 0.45))
    horizon = max(70, int((n_samples - min_train) / 5))

    for fold_idx in range(1, 6):
        train_end = min_train + (fold_idx - 1) * horizon
        test_start = train_end
        test_end = min(n_samples, test_start + horizon)

        if test_end - test_start < 40 or train_end < 180:
            continue

        X_fold = X[:train_end]
        y_fold = y[:train_end]
        elo_fold = elo_probs[:train_end]

        val_size = max(35, int(len(X_fold) * 0.2))
        fit_end = len(X_fold) - val_size
        if fit_end < 120:
            continue

        bundle = _train_bundle(
            X_train=X_fold[:fit_end],
            y_train=y_fold[:fit_end],
            elo_train=elo_fold[:fit_end],
            X_val=X_fold[fit_end:],
            y_val=y_fold[fit_end:],
            elo_val=elo_fold[fit_end:],
        )

        test_probs = _predict_with_model_bundle(bundle, X[test_start:test_end], elo_probs[test_start:test_end])
        fold_metrics = _metrics(y[test_start:test_end], test_probs)
        fold_metrics["fold"] = fold_idx
        fold_metrics["train_games"] = int(train_end)
        fold_metrics["test_games"] = int(test_end - test_start)
        folds.append(fold_metrics)

    if not folds:
        return {"folds": [], "mean": {}}

    mean_metrics = {
        "accuracy": round(float(np.mean([f["accuracy"] for f in folds])), 4),
        "log_loss": round(float(np.mean([f["log_loss"] for f in folds])), 4),
        "brier_score": round(float(np.mean([f["brier_score"] for f in folds])), 4),
        "accuracy_std": round(float(np.std([f["accuracy"] for f in folds], ddof=0)), 4),
        "log_loss_std": round(float(np.std([f["log_loss"] for f in folds], ddof=0)), 4),
    }
    return {"folds": folds, "mean": mean_metrics}


def _rolling_cv_summary_tree(X: np.ndarray, y: np.ndarray, elo_probs: np.ndarray) -> dict[str, Any]:
    n_samples = len(X)
    if n_samples < 500:
        return {"folds": [], "mean": {}}

    folds: list[dict[str, Any]] = []
    min_train = max(240, int(n_samples * 0.45))
    horizon = max(75, int((n_samples - min_train) / 5))

    for fold_idx in range(1, 6):
        train_end = min_train + (fold_idx - 1) * horizon
        test_start = train_end
        test_end = min(n_samples, test_start + horizon)

        if test_end - test_start < 45 or train_end < 220:
            continue

        X_fold = X[:train_end]
        y_fold = y[:train_end]
        elo_fold = elo_probs[:train_end]

        val_size = max(45, int(len(X_fold) * 0.2))
        fit_end = len(X_fold) - val_size
        if fit_end < 160:
            continue

        bundle = _train_tree_bundle(
            X_train=X_fold[:fit_end],
            y_train=y_fold[:fit_end],
            elo_train=elo_fold[:fit_end],
            X_val=X_fold[fit_end:],
            y_val=y_fold[fit_end:],
            elo_val=elo_fold[fit_end:],
            compute_importance=False,
        )

        test_probs = _predict_with_tree_bundle(bundle, X[test_start:test_end], elo_probs[test_start:test_end])
        fold_metrics = _metrics(y[test_start:test_end], test_probs)
        fold_metrics["fold"] = fold_idx
        fold_metrics["train_games"] = int(train_end)
        fold_metrics["test_games"] = int(test_end - test_start)
        folds.append(fold_metrics)

    if not folds:
        return {"folds": [], "mean": {}}

    mean_metrics = {
        "accuracy": round(float(np.mean([f["accuracy"] for f in folds])), 4),
        "log_loss": round(float(np.mean([f["log_loss"] for f in folds])), 4),
        "brier_score": round(float(np.mean([f["brier_score"] for f in folds])), 4),
        "accuracy_std": round(float(np.std([f["accuracy"] for f in folds], ddof=0)), 4),
        "log_loss_std": round(float(np.std([f["log_loss"] for f in folds], ddof=0)), 4),
    }
    return {"folds": folds, "mean": mean_metrics}


def _resolve_model_variant(model_variant: str | None) -> str:
    variant = str(model_variant or "v4").strip().lower()
    return variant if variant in MODEL_VARIANTS else "v4"


def _feature_names_for_variant(model_variant: str) -> list[str]:
    variant = _resolve_model_variant(model_variant)
    if variant == "v2":
        return FEATURE_NAMES
    return FEATURE_NAMES + V3_FEATURE_INTERACTIONS


def _expand_feature_vector_v3(base_features: np.ndarray) -> np.ndarray:
    if len(base_features) < len(FEATURE_NAMES):
        return base_features

    win_pct_diff = float(base_features[1])
    run_diff_pg_diff = float(base_features[2])
    recent10_diff = float(base_features[5])
    rest_days_diff = float(base_features[7])
    bullpen_recent_diff = float(base_features[8])
    pythag_diff = float(base_features[9])
    elo_diff = float(base_features[10])
    recent7_diff = float(base_features[11])
    run_diff_pg14_diff = float(base_features[14])
    form_trend_diff = float(base_features[18])
    form_volatility_edge = float(base_features[19])
    sample_reliability_diff = float(base_features[20])

    extras = np.asarray(
        [
            win_pct_diff * elo_diff,
            run_diff_pg_diff * elo_diff,
            recent10_diff * rest_days_diff,
            bullpen_recent_diff * rest_days_diff,
            pythag_diff * elo_diff,
            abs(rest_days_diff),
            abs(elo_diff),
            run_diff_pg_diff * run_diff_pg_diff,
            elo_diff * elo_diff,
            win_pct_diff * win_pct_diff,
            recent7_diff * elo_diff,
            form_trend_diff * elo_diff,
            run_diff_pg14_diff * rest_days_diff,
            sample_reliability_diff * elo_diff,
            form_volatility_edge * rest_days_diff,
        ],
        dtype=float,
    )

    return np.concatenate([base_features, extras])


def _expand_feature_matrix_v3(X_base: np.ndarray) -> np.ndarray:
    if len(X_base) == 0:
        return X_base

    rows = [_expand_feature_vector_v3(np.asarray(row, dtype=float)) for row in X_base]
    return np.asarray(rows, dtype=float)


def _build_multiseason_backtest_dataset(
    target_season: int,
    seasons_back: int = V3_MULTI_SEASON_LOOKBACK,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    start_season = max(2000, int(target_season) - max(1, int(seasons_back)) + 1)
    seasons = list(range(start_season, int(target_season) + 1))

    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    elo_parts: list[np.ndarray] = []
    meta_parts: list[dict[str, Any]] = []

    for season in seasons:
        X_s, y_s, elo_s, meta_s = _build_backtest_dataset(season)
        X_parts.append(X_s)
        y_parts.append(y_s)
        elo_parts.append(elo_s)

        for row in meta_s:
            copied = dict(row)
            copied["season"] = int(season)
            meta_parts.append(copied)

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    elo = np.concatenate(elo_parts)

    X = _expand_feature_matrix_v3(X)
    return X, y, elo, meta_parts


def _split_indices_v3(meta_rows: list[dict[str, Any]], target_season: int) -> tuple[int, int]:
    target_indices = [index for index, row in enumerate(meta_rows) if int(row.get("season", -1)) == int(target_season)]
    if len(target_indices) < 180:
        return _split_indices(len(meta_rows))

    start = target_indices[0]
    count = len(target_indices)

    train_end = start + max(120, int(count * 0.55))
    val_end = start + max(170, int(count * 0.80))

    train_end = min(train_end, len(meta_rows) - 80)
    val_end = min(val_end, len(meta_rows) - 30)

    if val_end <= train_end:
        val_end = min(len(meta_rows) - 30, train_end + 45)

    return train_end, val_end


def _feature_importance_summary(bundle: dict[str, Any], feature_names: list[str]) -> list[dict[str, Any]]:
    weights = np.asarray(bundle.get("weights", []), dtype=float)
    if len(weights) != len(feature_names):
        return []

    abs_weights = np.abs(weights)
    total_abs = float(abs_weights.sum())
    if total_abs <= 0:
        total_abs = 1.0

    rows: list[dict[str, Any]] = []
    for index, feature_name in enumerate(feature_names):
        coef = float(weights[index])
        abs_coef = float(abs_weights[index])
        rows.append(
            {
                "feature": feature_name,
                "coefficient": round(coef, 6),
                "abs_coefficient": round(abs_coef, 6),
                "importance_pct": round((abs_coef / total_abs) * 100.0, 2),
                "direction": "positive" if coef >= 0 else "negative",
            }
        )

    rows.sort(key=lambda row: row["abs_coefficient"], reverse=True)
    return rows


def _feature_ablation_summary(
    bundle: dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    elo_test: np.ndarray,
    feature_names: list[str],
) -> dict[str, Any]:
    if len(X_test) == 0:
        return {"baseline": {}, "rows": []}

    baseline_probs = _predict_with_model_bundle(bundle, X_test, elo_test)
    baseline_metrics = _metrics(y_test, baseline_probs)

    feature_mean = np.asarray(bundle.get("feature_mean", []), dtype=float)
    rows: list[dict[str, Any]] = []

    for index, feature_name in enumerate(feature_names):
        X_variant = X_test.copy()
        if index == 0:
            X_variant[:, 0] = 0.0
        elif index - 1 < len(feature_mean):
            X_variant[:, index] = feature_mean[index - 1]
        else:
            X_variant[:, index] = float(np.mean(X_test[:, index]))

        variant_probs = _predict_with_model_bundle(bundle, X_variant, elo_test)
        variant_metrics = _metrics(y_test, variant_probs)

        rows.append(
            {
                "feature": feature_name,
                "accuracy": variant_metrics["accuracy"],
                "log_loss": variant_metrics["log_loss"],
                "brier_score": variant_metrics["brier_score"],
                "delta_accuracy": round(variant_metrics["accuracy"] - baseline_metrics["accuracy"], 4),
                "delta_log_loss": round(variant_metrics["log_loss"] - baseline_metrics["log_loss"], 4),
                "delta_brier": round(variant_metrics["brier_score"] - baseline_metrics["brier_score"], 4),
            }
        )

    rows.sort(key=lambda row: row["delta_log_loss"], reverse=True)
    return {"baseline": baseline_metrics, "rows": rows}


def _is_valid_cached_report(
    report: dict[str, Any],
    model_variant: str = "v4",
    seasons_back: int = V4_MULTI_SEASON_LOOKBACK,
) -> bool:
    variant = _resolve_model_variant(model_variant)
    model_version = int(report.get("model_version") or 0)

    if "model" not in report or "metrics" not in report:
        return False

    calibration_test = report.get("metrics", {}).get("calibration_test", {})
    if not calibration_test or not list(calibration_test.get("reliability_curve") or []):
        return False

    if variant == "v2":
        return model_version == 2

    if variant == "v3":
        if model_version != 3:
            return False
        expected_back = max(1, int(seasons_back))
        return int(report.get("seasons_back") or expected_back) == expected_back

    if model_version != MODEL_VERSION:
        return False

    expected_back = max(1, int(seasons_back))
    if int(report.get("seasons_back") or expected_back) != expected_back:
        return False

    expected_features = _feature_names_for_variant("v4")
    report_features = list(report.get("feature_names") or [])
    if report_features != expected_features:
        return False

    model_block = report.get("model", {}) or {}
    model_features = list(model_block.get("feature_names") or [])
    if model_features != expected_features:
        return False

    model_path = str(model_block.get("model_path") or "")
    if not model_path:
        return False
    return Path(model_path).exists()


def _invalidate_prediction_cache_for_backtest_season(backtest_season: int, model_variant: str | None = None) -> None:
    target_variant = _resolve_model_variant(model_variant) if model_variant else None

    for cache_key in list(PREDICTION_CACHE.keys()):
        parts = str(cache_key).split("::")
        if len(parts) < 3:
            continue

        if parts[2] != str(backtest_season):
            continue

        if target_variant is not None:
            if len(parts) < 4 or _resolve_model_variant(parts[3]) != target_variant:
                continue

        PREDICTION_CACHE.pop(cache_key, None)


def _build_report_from_split(
    *,
    season: int,
    model_variant: str,
    seasons_back: int,
    feature_names: list[str],
    X: np.ndarray,
    y: np.ndarray,
    elo_probs: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    elo_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    elo_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    elo_test: np.ndarray,
    notes: str,
    training_seasons: list[int] | None = None,
) -> dict[str, Any]:
    bundle = _train_bundle(
        X_train=X_train,
        y_train=y_train,
        elo_train=elo_train,
        X_val=X_val,
        y_val=y_val,
        elo_val=elo_val,
    )

    train_probs = _predict_with_model_bundle(bundle, X_train, elo_train)
    val_probs = _predict_with_model_bundle(bundle, X_val, elo_val)
    test_probs = _predict_with_model_bundle(bundle, X_test, elo_test)
    full_probs = _predict_with_model_bundle(bundle, X, elo_probs)

    train_metrics = _metrics(y_train, train_probs)
    val_metrics = _metrics(y_val, val_probs)
    test_metrics = _metrics(y_test, test_probs)
    full_metrics = _metrics(y, full_probs)
    calibration_test = _build_calibration_diagnostics(y_test, test_probs)

    rolling_cv = _rolling_cv_summary(X, y, elo_probs)
    drift_monitor = _build_drift_monitor(train_metrics, val_metrics, test_metrics, rolling_cv)
    feature_importance = _feature_importance_summary(bundle, feature_names=feature_names)
    ablation = _feature_ablation_summary(
        bundle=bundle,
        X_test=X_test,
        y_test=y_test,
        elo_test=elo_test,
        feature_names=feature_names,
    )

    report = {
        "model_version": 2 if model_variant == "v2" else (3 if model_variant == "v3" else MODEL_VERSION),
        "model_variant": model_variant,
        "season": season,
        "seasons_back": int(max(1, seasons_back)),
        "training_seasons": training_seasons or [season],
        "trained_at": dt.datetime.now().isoformat(),
        "games_total": int(len(X)),
        "games_train": int(len(X_train)),
        "games_validation": int(len(X_val)),
        "games_test": int(len(X_test)),
        "feature_names": feature_names,
        "weights": [float(round(w, 6)) for w in bundle["weights"]],
        "model": {
            "feature_names": feature_names,
            "weights": [float(round(w, 8)) for w in bundle["weights"]],
            "feature_mean": [float(round(v, 8)) for v in bundle["feature_mean"]],
            "feature_std": [float(round(v, 8)) for v in bundle["feature_std"]],
            "blend_weight": float(round(bundle["blend_weight"], 4)),
            "calibration": {
                "intercept": float(round(bundle["calibration"]["intercept"], 8)),
                "slope": float(round(bundle["calibration"]["slope"], 8)),
            },
            "feature_importance": feature_importance,
            "ablation": ablation,
            "calibration_test": calibration_test,
            "drift_monitor": drift_monitor,
        },
        "metrics": {
            "train": train_metrics,
            "validation": val_metrics,
            "test": test_metrics,
            "full": full_metrics,
            "rolling_cv": rolling_cv,
            "calibration_test": calibration_test,
            "drift_monitor": drift_monitor,
        },
        "feature_importance": feature_importance,
        "ablation": ablation,
        "notes": notes,
    }

    return report


def _run_backtest_for_season_v2(season: int, force_retrain: bool = False) -> dict[str, Any]:
    cache_key = (season, "v2", 1)

    with MODEL_CACHE_LOCK:
        if force_retrain:
            _invalidate_prediction_cache_for_backtest_season(season, model_variant="v2")

        if not force_retrain and cache_key in MODEL_CACHE and _is_valid_cached_report(MODEL_CACHE[cache_key], "v2", 1):
            return MODEL_CACHE[cache_key]

        path = _report_path(season, model_variant="v2", seasons_back=1)
        if path.exists() and not force_retrain:
            cached = json.loads(path.read_text(encoding="utf-8"))
            if _is_valid_cached_report(cached, "v2", 1):
                MODEL_CACHE[cache_key] = cached
                return cached

        X, y, elo_probs, _meta = _build_backtest_dataset(season)
        train_end, val_end = _split_indices(len(X))

        X_train, y_train, elo_train = X[:train_end], y[:train_end], elo_probs[:train_end]
        X_val, y_val, elo_val = X[train_end:val_end], y[train_end:val_end], elo_probs[train_end:val_end]
        X_test, y_test, elo_test = X[val_end:], y[val_end:], elo_probs[val_end:]

        report = _build_report_from_split(
            season=season,
            model_variant="v2",
            seasons_back=1,
            feature_names=FEATURE_NAMES,
            X=X,
            y=y,
            elo_probs=elo_probs,
            X_train=X_train,
            y_train=y_train,
            elo_train=elo_train,
            X_val=X_val,
            y_val=y_val,
            elo_val=elo_val,
            X_test=X_test,
            y_test=y_test,
            elo_test=elo_test,
            notes="V2 baseline: single-season linear features with logistic+Elo blend and calibration.",
            training_seasons=[season],
        )

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        _invalidate_prediction_cache_for_backtest_season(season, model_variant="v2")
        MODEL_CACHE[cache_key] = report
        return report


def _run_backtest_for_season_v3(
    season: int,
    force_retrain: bool = False,
    seasons_back: int = V3_MULTI_SEASON_LOOKBACK,
) -> dict[str, Any]:
    back = max(1, int(seasons_back))
    cache_key = (season, "v3", back)

    with MODEL_CACHE_LOCK:
        if force_retrain:
            _invalidate_prediction_cache_for_backtest_season(season, model_variant="v3")

        if not force_retrain and cache_key in MODEL_CACHE and _is_valid_cached_report(MODEL_CACHE[cache_key], "v3", back):
            return MODEL_CACHE[cache_key]

        path = _report_path(season, model_variant="v3", seasons_back=back)
        if path.exists() and not force_retrain:
            cached = json.loads(path.read_text(encoding="utf-8"))
            if _is_valid_cached_report(cached, "v3", back):
                MODEL_CACHE[cache_key] = cached
                return cached

        X, y, elo_probs, meta = _build_multiseason_backtest_dataset(season, seasons_back=back)
        train_end, val_end = _split_indices_v3(meta, target_season=season)

        X_train, y_train, elo_train = X[:train_end], y[:train_end], elo_probs[:train_end]
        X_val, y_val, elo_val = X[train_end:val_end], y[train_end:val_end], elo_probs[train_end:val_end]
        X_test, y_test, elo_test = X[val_end:], y[val_end:], elo_probs[val_end:]

        training_seasons = sorted({int(row.get("season", season)) for row in meta if int(row.get("season", season)) <= season})

        report = _build_report_from_split(
            season=season,
            model_variant="v3",
            seasons_back=back,
            feature_names=_feature_names_for_variant("v3"),
            X=X,
            y=y,
            elo_probs=elo_probs,
            X_train=X_train,
            y_train=y_train,
            elo_train=elo_train,
            X_val=X_val,
            y_val=y_val,
            elo_val=elo_val,
            X_test=X_test,
            y_test=y_test,
            elo_test=elo_test,
            notes="V3 candidate: multi-season walk-forward training set with nonlinear interaction expansion.",
            training_seasons=training_seasons,
        )

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        _invalidate_prediction_cache_for_backtest_season(season, model_variant="v3")
        MODEL_CACHE[cache_key] = report
        return report


def _run_backtest_for_season_v4(
    season: int,
    force_retrain: bool = False,
    seasons_back: int = V4_MULTI_SEASON_LOOKBACK,
) -> dict[str, Any]:
    back = max(1, int(seasons_back))
    cache_key = (season, "v4", back)

    with MODEL_CACHE_LOCK:
        if force_retrain:
            _invalidate_prediction_cache_for_backtest_season(season, model_variant="v4")

        if not force_retrain and cache_key in MODEL_CACHE and _is_valid_cached_report(MODEL_CACHE[cache_key], "v4", back):
            return MODEL_CACHE[cache_key]

        path = _report_path(season, model_variant="v4", seasons_back=back)
        if path.exists() and not force_retrain:
            cached = json.loads(path.read_text(encoding="utf-8"))
            if _is_valid_cached_report(cached, "v4", back):
                MODEL_CACHE[cache_key] = cached
                return cached

        X, y, elo_probs, meta = _build_multiseason_backtest_dataset(season, seasons_back=back)
        train_end, val_end = _split_indices_v3(meta, target_season=season)

        X_train, y_train, elo_train = X[:train_end], y[:train_end], elo_probs[:train_end]
        X_val, y_val, elo_val = X[train_end:val_end], y[train_end:val_end], elo_probs[train_end:val_end]
        X_test, y_test, elo_test = X[val_end:], y[val_end:], elo_probs[val_end:]

        bundle = _train_tree_bundle(
            X_train=X_train,
            y_train=y_train,
            elo_train=elo_train,
            X_val=X_val,
            y_val=y_val,
            elo_val=elo_val,
        )

        train_probs = _predict_with_tree_bundle(bundle, X_train, elo_train)
        val_probs = _predict_with_tree_bundle(bundle, X_val, elo_val)
        test_probs = _predict_with_tree_bundle(bundle, X_test, elo_test)
        full_probs = _predict_with_tree_bundle(bundle, X, elo_probs)

        train_metrics = _metrics(y_train, train_probs)
        val_metrics = _metrics(y_val, val_probs)
        test_metrics = _metrics(y_test, test_probs)
        full_metrics = _metrics(y, full_probs)
        calibration_test = _build_calibration_diagnostics(y_test, test_probs)
        rolling_cv = _rolling_cv_summary_tree(X, y, elo_probs)
        drift_monitor = _build_drift_monitor(train_metrics, val_metrics, test_metrics, rolling_cv)

        feature_names = _feature_names_for_variant("v4")
        feature_importance = _feature_importance_tree_summary(bundle, feature_names=feature_names)
        ablation = _feature_ablation_tree_summary(
            bundle=bundle,
            X_test=X_test,
            y_test=y_test,
            elo_test=elo_test,
            feature_names=feature_names,
        )

        training_seasons = sorted({int(row.get("season", season)) for row in meta if int(row.get("season", season)) <= season})
        model_obj_path = _model_object_path(season, model_variant="v4", seasons_back=back)
        model_obj_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle.get("tree_model"), model_obj_path)
        MODEL_OBJECT_CACHE[str(model_obj_path)] = bundle.get("tree_model")

        report = {
            "model_version": MODEL_VERSION,
            "model_variant": "v4",
            "model_type": "tree_gbdt",
            "season": season,
            "seasons_back": back,
            "training_seasons": training_seasons,
            "trained_at": dt.datetime.now().isoformat(),
            "games_total": int(len(X)),
            "games_train": int(len(X_train)),
            "games_validation": int(len(X_val)),
            "games_test": int(len(X_test)),
            "feature_names": feature_names,
            "weights": [],
            "model": {
                "model_type": "tree_gbdt",
                "model_path": str(model_obj_path),
                "feature_names": feature_names,
                "weights": [],
                "feature_mean": bundle.get("feature_mean", []),
                "feature_std": [],
                "blend_weight": float(round(bundle.get("blend_weight", 0.75), 4)),
                "calibration": {
                    "intercept": float(round(bundle.get("calibration", {}).get("intercept", 0.0), 8)),
                    "slope": float(round(bundle.get("calibration", {}).get("slope", 1.0), 8)),
                },
                "feature_importance": feature_importance,
                "ablation": ablation,
                "calibration_test": calibration_test,
                "drift_monitor": drift_monitor,
            },
            "metrics": {
                "train": train_metrics,
                "validation": val_metrics,
                "test": test_metrics,
                "full": full_metrics,
                "rolling_cv": rolling_cv,
                "calibration_test": calibration_test,
                "drift_monitor": drift_monitor,
            },
            "feature_importance": feature_importance,
            "ablation": ablation,
            "notes": "V5 production: multi-season walk-forward tree ensemble with rolling-window + shrinkage features, Elo blend, and Platt calibration.",
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        _invalidate_prediction_cache_for_backtest_season(season, model_variant="v4")
        MODEL_CACHE[cache_key] = report
        return report


def run_backtest_for_season(
    season: int,
    force_retrain: bool = False,
    model_variant: str = "v4",
    seasons_back: int = V4_MULTI_SEASON_LOOKBACK,
) -> dict[str, Any]:
    variant = _resolve_model_variant(model_variant)
    if variant == "v2":
        return _run_backtest_for_season_v2(season=season, force_retrain=force_retrain)
    if variant == "v3":
        return _run_backtest_for_season_v3(
            season=season,
            force_retrain=force_retrain,
            seasons_back=seasons_back,
        )

    return _run_backtest_for_season_v4(
        season=season,
        force_retrain=force_retrain,
        seasons_back=seasons_back,
    )


def run_backtest_model_comparison(
    season: int,
    force_retrain: bool = False,
    seasons_back: int = V4_MULTI_SEASON_LOOKBACK,
) -> dict[str, Any]:
    v2 = run_backtest_for_season(season=season, force_retrain=force_retrain, model_variant="v2", seasons_back=1)
    v3 = run_backtest_for_season(season=season, force_retrain=force_retrain, model_variant="v3", seasons_back=seasons_back)
    v4 = run_backtest_for_season(season=season, force_retrain=force_retrain, model_variant="v4", seasons_back=seasons_back)

    test_v2 = v2.get("metrics", {}).get("test", {})
    test_v3 = v3.get("metrics", {}).get("test", {})
    test_v4 = v4.get("metrics", {}).get("test", {})

    delta_v3_minus_v2 = {
        "accuracy": round(float(test_v3.get("accuracy", 0.0) - test_v2.get("accuracy", 0.0)), 4),
        "log_loss": round(float(test_v3.get("log_loss", 0.0) - test_v2.get("log_loss", 0.0)), 4),
        "brier_score": round(float(test_v3.get("brier_score", 0.0) - test_v2.get("brier_score", 0.0)), 4),
    }
    delta_v4_minus_v3 = {
        "accuracy": round(float(test_v4.get("accuracy", 0.0) - test_v3.get("accuracy", 0.0)), 4),
        "log_loss": round(float(test_v4.get("log_loss", 0.0) - test_v3.get("log_loss", 0.0)), 4),
        "brier_score": round(float(test_v4.get("brier_score", 0.0) - test_v3.get("brier_score", 0.0)), 4),
    }
    delta_v4_minus_v2 = {
        "accuracy": round(float(test_v4.get("accuracy", 0.0) - test_v2.get("accuracy", 0.0)), 4),
        "log_loss": round(float(test_v4.get("log_loss", 0.0) - test_v2.get("log_loss", 0.0)), 4),
        "brier_score": round(float(test_v4.get("brier_score", 0.0) - test_v2.get("brier_score", 0.0)), 4),
    }

    return {
        "season": season,
        "seasons_back": max(1, int(seasons_back)),
        "v2": v2,
        "v3": v3,
        "v4": v4,
        "delta_v3_minus_v2": delta_v3_minus_v2,
        "delta_v4_minus_v3": delta_v4_minus_v3,
        "delta_v4_minus_v2": delta_v4_minus_v2,
    }


def _empty_team_state_map(team_lookup: dict[int, str]) -> dict[str, TeamState]:
    return {team_name: TeamState() for team_name in team_lookup.values()}


def _apply_final_game_to_states(
    game: dict[str, Any],
    states: dict[str, TeamState],
    team_lookup: dict[int, str],
    team_context: dict[int, dict[str, Any]],
) -> None:
    game_date = _parse_date(game.get("official_date"))
    if game_date is None:
        return

    home_id = game.get("home_id")
    away_id = game.get("away_id")
    if home_id is None or away_id is None:
        return

    home_name = team_lookup.get(int(home_id))
    away_name = team_lookup.get(int(away_id))
    if not home_name or not away_name:
        return

    home_state = states.setdefault(home_name, TeamState())
    away_state = states.setdefault(away_name, TeamState())

    home_score = game.get("home_score")
    away_score = game.get("away_score")
    if home_score is None or away_score is None:
        return

    home_score = int(home_score)
    away_score = int(away_score)
    if home_score == away_score:
        return

    home_context = team_context.get(int(home_id), {})
    venue_tz_offset = _to_float(home_context.get("tz_offset"), 0.0)

    _update_states_after_game(
        home_state,
        away_state,
        home_score=home_score,
        away_score=away_score,
        game_date=game_date,
        innings_count=game.get("innings_count"),
        venue_tz_offset=venue_tz_offset,
    )


def _build_states_before_date(reference_date: dt.date, season: int) -> tuple[dict[str, TeamState], dict[int, str]]:
    team_lookup = fetch_team_name_lookup(season)
    team_context = _fetch_team_context_lookup(season)
    states = _empty_team_state_map(team_lookup)

    games = _fetch_schedule_range(dt.date(season, 3, 1), reference_date)
    for game in games:
        if game.get("state") not in FINAL_STATES:
            continue

        game_date = _parse_date(game.get("official_date"))
        if game_date is None or game_date >= reference_date:
            continue

        _apply_final_game_to_states(game, states, team_lookup, team_context)

    return states, team_lookup


def _parse_ip_to_float(value: Any) -> float:
    if value is None:
        return 0.0

    raw = str(value).strip()
    if not raw:
        return 0.0

    if "." not in raw:
        return _to_float(raw, 0.0)

    whole, frac = raw.split(".", 1)
    innings = _to_float(whole, 0.0)
    outs = _to_int(frac[:1], 0)
    outs = max(0, min(outs, 2))
    return innings + (outs / 3.0)


def _blend_with_prior(value: float | None, sample: float, prior: float, prior_weight: float) -> float | None:
    if value is None:
        return None
    s = max(0.0, float(sample))
    return ((float(value) * s) + (prior * prior_weight)) / (s + prior_weight)


def _fetch_pitcher_profile(pitcher_id: int | None, season: int) -> dict[str, Any]:
    if pitcher_id is None:
        return {}

    key = (season, int(pitcher_id))
    with MODEL_CACHE_LOCK:
        if key in PITCHER_CACHE:
            return PITCHER_CACHE[key]

    profile: dict[str, Any] = {
        "id": int(pitcher_id),
        "name": None,
        "hand": None,
        "era": None,
        "whip": None,
        "fip": None,
        "xfip": None,
        "k_minus_bb_per9": None,
        "innings_pitched": 0.0,
        "games_started": 0,
        "quality_starts": 0,
        "reliability": 0.0,
    }

    try:
        bio = requests.get(
            f"{BASE_API_URL}/people/{pitcher_id}",
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        bio.raise_for_status()
        person = (bio.json().get("people") or [{}])[0]
        profile["name"] = person.get("fullName")
        profile["hand"] = str(person.get("pitchHand", {}).get("code") or "").upper() or None
    except Exception:
        pass

    try:
        stat_resp = requests.get(
            f"{BASE_API_URL}/people/{pitcher_id}/stats",
            params={"stats": "season", "group": "pitching", "season": season},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        stat_resp.raise_for_status()
        stats_payload = stat_resp.json()
        blocks = stats_payload.get("stats", [])
        split = None
        if blocks:
            split = (blocks[0].get("splits") or [None])[0]
        stat = split.get("stat", {}) if split else {}

        era = _to_float(stat.get("era"), float("nan"))
        whip = _to_float(stat.get("whip"), float("nan"))
        fip = _to_float(stat.get("fip"), float("nan"))
        xfip = _to_float(stat.get("xfip") or stat.get("xFip") or stat.get("xFIP"), float("nan"))
        so = _to_float(stat.get("strikeOuts"), 0.0)
        bb = _to_float(stat.get("baseOnBalls"), 0.0)
        ip = _parse_ip_to_float(stat.get("inningsPitched"))
        gs = _to_int(stat.get("gamesStarted"), 0)
        qs = _to_int(stat.get("qualityStarts"), 0)

        raw_era = None if math.isnan(era) else era
        raw_whip = None if math.isnan(whip) else whip
        raw_fip = None if math.isnan(fip) else fip
        raw_xfip = None if math.isnan(xfip) else xfip

        # Stabilize early-season volatility with league-ish priors weighted by IP sample.
        blended_era = _blend_with_prior(raw_era, sample=ip, prior=4.25, prior_weight=35.0)
        blended_whip = _blend_with_prior(raw_whip, sample=ip, prior=1.32, prior_weight=35.0)
        blended_fip = _blend_with_prior(raw_fip, sample=ip, prior=4.20, prior_weight=35.0)
        blended_xfip = _blend_with_prior(raw_xfip, sample=ip, prior=4.20, prior_weight=35.0)

        profile["era"] = blended_era
        profile["whip"] = blended_whip
        profile["fip"] = blended_fip
        profile["xfip"] = blended_xfip if blended_xfip is not None else blended_fip
        profile["k_minus_bb_per9"] = ((so - bb) * 9.0 / ip) if ip > 0 else None
        profile["innings_pitched"] = ip
        profile["games_started"] = gs
        profile["quality_starts"] = qs
        profile["reliability"] = min(1.0, ip / 60.0)
    except Exception:
        pass

    with MODEL_CACHE_LOCK:
        PITCHER_CACHE[key] = profile

    return profile


def _extract_split_win_pct(split_records: list[dict[str, Any]], split_type: str, fallback: float) -> float:
    for row in split_records:
        label = str(row.get("type") or "").strip().lower()
        if label != split_type:
            continue

        wins = _to_int(row.get("wins"), 0)
        losses = _to_int(row.get("losses"), 0)
        return _smoothed_rate(wins, wins + losses, prior_mean=fallback, prior_weight=4.0)

    return fallback


def _fetch_split_lookup(season: int) -> dict[str, dict[str, float]]:
    try:
        team_lookup = fetch_team_name_lookup(season)
        response = requests.get(
            f"{BASE_API_URL}/standings",
            params={
                "leagueId": "103,104",
                "season": season,
                "standingsTypes": "regularSeason",
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return {}

    output: dict[str, dict[str, float]] = {}

    for division in payload.get("records", []):
        for row in division.get("teamRecords", []):
            team_obj = row.get("team", {})
            team_id = team_obj.get("id")
            canonical = team_lookup.get(int(team_id)) if team_id is not None else team_obj.get("name")
            if not canonical:
                continue

            wins = _to_int(row.get("wins"), 0)
            losses = _to_int(row.get("losses"), 0)
            overall = _smoothed_rate(wins, wins + losses, prior_mean=0.5, prior_weight=8.0)

            split_records = row.get("records", {}).get("splitRecords", []) or []
            output[canonical] = {
                "overall": overall,
                "home": _extract_split_win_pct(split_records, "home", fallback=0.54),
                "away": _extract_split_win_pct(split_records, "away", fallback=0.46),
                "vs_left": _extract_split_win_pct(split_records, "left", fallback=overall),
                "vs_right": _extract_split_win_pct(split_records, "right", fallback=overall),
                "last10": _extract_split_win_pct(split_records, "lastten", fallback=overall),
            }

    return output


def _starter_adjustment(home_pitcher: dict[str, Any], away_pitcher: dict[str, Any]) -> tuple[float, dict[str, float]]:
    if not home_pitcher or not away_pitcher:
        return 0.0, {"starter_adjustment": 0.0}

    home_era = home_pitcher.get("era")
    away_era = away_pitcher.get("era")
    home_whip = home_pitcher.get("whip")
    away_whip = away_pitcher.get("whip")
    home_fip = home_pitcher.get("fip")
    away_fip = away_pitcher.get("fip")
    home_kbb = home_pitcher.get("k_minus_bb_per9")
    away_kbb = away_pitcher.get("k_minus_bb_per9")

    home_rel = float(home_pitcher.get("reliability") or 0.0)
    away_rel = float(away_pitcher.get("reliability") or 0.0)
    rel_factor = 0.5 + 0.5 * min(home_rel, away_rel)

    era_edge = 0.0
    whip_edge = 0.0
    fip_edge = 0.0
    kbb_edge = 0.0
    qs_edge = 0.0

    if home_era is not None and away_era is not None:
        era_edge = (float(away_era) - float(home_era)) * 0.012

    if home_whip is not None and away_whip is not None:
        whip_edge = (float(away_whip) - float(home_whip)) * 0.03

    if home_fip is not None and away_fip is not None:
        fip_edge = (float(away_fip) - float(home_fip)) * 0.01

    if home_kbb is not None and away_kbb is not None:
        kbb_edge = (float(home_kbb) - float(away_kbb)) * 0.01

    home_gs = max(1, int(home_pitcher.get("games_started") or 1))
    away_gs = max(1, int(away_pitcher.get("games_started") or 1))
    home_qs_rate = float(home_pitcher.get("quality_starts") or 0) / home_gs
    away_qs_rate = float(away_pitcher.get("quality_starts") or 0) / away_gs
    qs_edge = (home_qs_rate - away_qs_rate) * 0.025

    total_raw = era_edge + whip_edge + fip_edge + kbb_edge + qs_edge
    total = max(-0.09, min(0.09, total_raw * rel_factor))

    return total, {
        "starter_adjustment": round(total, 4),
        "starter_era_edge": round(era_edge * rel_factor, 4),
        "starter_whip_edge": round(whip_edge * rel_factor, 4),
        "starter_fip_edge": round(fip_edge * rel_factor, 4),
        "starter_kbb_edge": round(kbb_edge * rel_factor, 4),
        "starter_qs_edge": round(qs_edge * rel_factor, 4),
        "starter_reliability_factor": round(rel_factor, 4),
    }


def _starter_public_metrics(pitcher: dict[str, Any]) -> dict[str, Any]:
    if not pitcher:
        return {
            "starter_id": None,
            "starter_name": None,
            "starter_hand": None,
            "starter_era": None,
            "starter_whip": None,
            "starter_fip": None,
            "starter_xfip": None,
            "starter_k_minus_bb_per9": None,
            "starter_innings_pitched": None,
            "starter_games_started": None,
            "starter_quality_starts": None,
            "starter_qs_rate": None,
            "starter_reliability": None,
        }

    games_started = _to_int(pitcher.get("games_started"), 0)
    quality_starts = _to_int(pitcher.get("quality_starts"), 0)
    qs_rate = (float(quality_starts) / float(games_started)) if games_started > 0 else None

    xfip_value = pitcher.get("xfip")
    if xfip_value is None:
        xfip_value = pitcher.get("fip")

    reliability = _to_float(pitcher.get("reliability"), 0.0)
    reliability = max(0.0, min(1.0, reliability))

    return {
        "starter_id": _to_int(pitcher.get("id"), 0) or None,
        "starter_name": pitcher.get("name"),
        "starter_hand": pitcher.get("hand"),
        "starter_era": _as_rounded(pitcher.get("era"), 2),
        "starter_whip": _as_rounded(pitcher.get("whip"), 3),
        "starter_fip": _as_rounded(pitcher.get("fip"), 2),
        "starter_xfip": _as_rounded(xfip_value, 2),
        "starter_k_minus_bb_per9": _as_rounded(pitcher.get("k_minus_bb_per9"), 2),
        "starter_innings_pitched": _as_rounded(pitcher.get("innings_pitched"), 1),
        "starter_games_started": games_started if games_started > 0 else None,
        "starter_quality_starts": quality_starts if games_started > 0 else None,
        "starter_qs_rate": round(float(qs_rate), 3) if qs_rate is not None else None,
        "starter_reliability": round(float(reliability), 3),
    }


def _split_adjustment(
    home_team: str,
    away_team: str,
    split_lookup: dict[str, dict[str, float]],
    home_starter_hand: str | None,
    away_starter_hand: str | None,
) -> tuple[float, dict[str, float]]:
    home_split = split_lookup.get(home_team, {})
    away_split = split_lookup.get(away_team, {})

    if not home_split or not away_split:
        return 0.0, {"split_adjustment": 0.0}

    home_field_edge = (home_split.get("home", 0.54) - away_split.get("away", 0.46)) * 0.1

    away_hand_key = "vs_left" if (away_starter_hand or "").upper() == "L" else "vs_right"
    home_hand_key = "vs_left" if (home_starter_hand or "").upper() == "L" else "vs_right"

    hand_edge = (home_split.get(away_hand_key, home_split.get("overall", 0.5)) - away_split.get(home_hand_key, away_split.get("overall", 0.5))) * 0.07
    form_edge = (home_split.get("last10", home_split.get("overall", 0.5)) - away_split.get("last10", away_split.get("overall", 0.5))) * 0.04

    total = max(-0.07, min(0.07, home_field_edge + hand_edge + form_edge))
    return total, {
        "split_adjustment": round(total, 4),
        "split_home_away_edge": round(home_field_edge, 4),
        "split_handedness_edge": round(hand_edge, 4),
        "split_last10_edge": round(form_edge, 4),
    }



def _luck_factor_adjustment(home_state: TeamState, away_state: TeamState) -> tuple[float, dict[str, float]]:
    home = _team_features(home_state)
    away = _team_features(away_state)

    home_luck = float(home.get("win_pct", 0.5) - home.get("pythag_win_pct", 0.5))
    away_luck = float(away.get("win_pct", 0.5) - away.get("pythag_win_pct", 0.5))

    regression_edge = (away_luck - home_luck) * 0.32
    total = max(-0.045, min(0.045, regression_edge))

    return total, {
        "luck_adjustment": round(total, 4),
        "luck_regression_edge": round(regression_edge, 4),
        "home_luck_index": round(home_luck, 4),
        "away_luck_index": round(away_luck, 4),
    }


def _advanced_team_quality_adjustment(
    home_team: str,
    away_team: str,
    advanced_lookup: dict[str, dict[str, float]],
) -> tuple[float, dict[str, float | None]]:
    home = advanced_lookup.get(home_team)
    away = advanced_lookup.get(away_team)

    if not home or not away:
        return 0.0, {
            "advanced_adjustment": 0.0,
            "advanced_offense_edge": 0.0,
            "advanced_pitching_edge": 0.0,
            "advanced_xwobacon_edge": 0.0,
            "advanced_xfip_edge": 0.0,
            "advanced_leverage_edge": 0.0,
            "advanced_clutch_edge": 0.0,
            "home_xwobacon_proxy": round(float(home.get("xwobacon_proxy")), 4) if home else None,
            "away_xwobacon_proxy": round(float(away.get("xwobacon_proxy")), 4) if away else None,
            "home_xfip_proxy": round(float(home.get("xfip_proxy")), 3) if home else None,
            "away_xfip_proxy": round(float(away.get("xfip_proxy")), 3) if away else None,
            "home_expected_offense_quality": round(float(home.get("offense_expected_quality") * 100.0), 1) if home else None,
            "away_expected_offense_quality": round(float(away.get("offense_expected_quality") * 100.0), 1) if away else None,
            "home_expected_pitching_quality": round(float(home.get("pitching_expected_quality") * 100.0), 1) if home else None,
            "away_expected_pitching_quality": round(float(away.get("pitching_expected_quality") * 100.0), 1) if away else None,
            "home_leverage_net_quality": round(float(home.get("leverage_net_quality") * 100.0), 1) if home else None,
            "away_leverage_net_quality": round(float(away.get("leverage_net_quality") * 100.0), 1) if away else None,
            "home_clutch_index": round(float(home.get("clutch_index") * 100.0), 1) if home else None,
            "away_clutch_index": round(float(away.get("clutch_index") * 100.0), 1) if away else None,
        }

    offense_edge = (float(home["offense_expected_quality"]) - float(away["offense_expected_quality"])) * 0.06
    pitching_edge = (float(home["pitching_expected_quality"]) - float(away["pitching_expected_quality"])) * 0.07
    xwobacon_edge = (float(home["xwobacon_proxy"]) - float(away["xwobacon_proxy"])) * 0.10
    xfip_edge = (float(away["xfip_proxy"]) - float(home["xfip_proxy"])) * 0.018
    leverage_edge = (float(home.get("leverage_net_quality", 0.5)) - float(away.get("leverage_net_quality", 0.5))) * 0.03
    clutch_edge = (float(home.get("clutch_index", 0.5)) - float(away.get("clutch_index", 0.5))) * 0.02

    total = max(-0.05, min(0.05, offense_edge + pitching_edge + xwobacon_edge + xfip_edge + leverage_edge + clutch_edge))

    return total, {
        "advanced_adjustment": round(total, 4),
        "advanced_offense_edge": round(offense_edge, 4),
        "advanced_pitching_edge": round(pitching_edge, 4),
        "advanced_xwobacon_edge": round(xwobacon_edge, 4),
        "advanced_xfip_edge": round(xfip_edge, 4),
        "advanced_leverage_edge": round(leverage_edge, 4),
        "advanced_clutch_edge": round(clutch_edge, 4),
        "home_xwobacon_proxy": round(float(home["xwobacon_proxy"]), 4),
        "away_xwobacon_proxy": round(float(away["xwobacon_proxy"]), 4),
        "home_xfip_proxy": round(float(home["xfip_proxy"]), 3),
        "away_xfip_proxy": round(float(away["xfip_proxy"]), 3),
        "home_expected_offense_quality": round(float(home["offense_expected_quality"]) * 100.0, 1),
        "away_expected_offense_quality": round(float(away["offense_expected_quality"]) * 100.0, 1),
        "home_expected_pitching_quality": round(float(home["pitching_expected_quality"]) * 100.0, 1),
        "away_expected_pitching_quality": round(float(away["pitching_expected_quality"]) * 100.0, 1),
        "home_leverage_net_quality": round(float(home.get("leverage_net_quality", 0.5)) * 100.0, 1),
        "away_leverage_net_quality": round(float(away.get("leverage_net_quality", 0.5)) * 100.0, 1),
        "home_clutch_index": round(float(home.get("clutch_index", 0.5)) * 100.0, 1),
        "away_clutch_index": round(float(away.get("clutch_index", 0.5)) * 100.0, 1),
    }


TEAM_ALIASES = {
    "la dodgers": "los angeles dodgers",
    "la angels": "los angeles angels",
    "ny yankees": "new york yankees",
    "ny mets": "new york mets",
    "chi cubs": "chicago cubs",
    "chi white sox": "chicago white sox",
    "kc royals": "kansas city royals",
    "sd padres": "san diego padres",
    "sf giants": "san francisco giants",
    "tb rays": "tampa bay rays",
    "d backs": "arizona diamondbacks",
    "diamondbacks": "arizona diamondbacks",
    "guardians": "cleveland guardians",
}


def _normalize_team_name(name: str | None) -> str:
    raw = str(name or "").strip().lower().replace(".", "")
    raw = " ".join(raw.split())
    return TEAM_ALIASES.get(raw, raw)


def _parse_iso_utc_timestamp(value: Any) -> dt.datetime | None:
    text = str(value or "").strip()
    if not text:
        return None

    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"

    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)

    return parsed


def _fetch_market_probabilities(reference_date: dt.date, canonical_teams: set[str]) -> dict[tuple[str, str], dict[str, Any]]:
    api_key = os.getenv("MSCORE_ODDS_API_KEY", "").strip()
    if not api_key:
        return {}

    normalized_to_canonical = {_normalize_team_name(team): team for team in canonical_teams}

    window_start = dt.datetime.combine(reference_date, dt.time(hour=8, minute=0, tzinfo=dt.timezone.utc))
    window_end = window_start + dt.timedelta(hours=23, minutes=59, seconds=59)

    def _window_params() -> dict[str, str]:
        return {
            "apiKey": api_key,
            "regions": "us",
            "markets": "h2h",
            "oddsFormat": "decimal",
            "dateFormat": "iso",
            "commenceTimeFrom": window_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "commenceTimeTo": window_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

    try:
        response = requests.get(
            "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds",
            params=_window_params(),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return {}

    # Fallback: some feed states return sparse/empty rows with strict commence filters.
    if not isinstance(payload, list) or not payload:
        try:
            response = requests.get(
                "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds",
                params={
                    "apiKey": api_key,
                    "regions": "us",
                    "markets": "h2h",
                    "oddsFormat": "decimal",
                    "dateFormat": "iso",
                },
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception:
            return {}

    output: dict[tuple[str, str], dict[str, Any]] = {}

    for event in payload if isinstance(payload, list) else []:
        event_time = _parse_iso_utc_timestamp(event.get("commence_time"))
        if event_time is not None and (event_time < window_start or event_time > window_end):
            continue

        home_raw = event.get("home_team")
        away_raw = event.get("away_team")

        if not away_raw:
            teams = event.get("teams") or []
            for team_name in teams:
                if team_name != home_raw:
                    away_raw = team_name
                    break

        # Newer odds payloads can omit `teams`; derive away from h2h outcomes.
        if not away_raw:
            for book in event.get("bookmakers", []):
                for market in book.get("markets", []):
                    if market.get("key") != "h2h":
                        continue
                    for outcome in market.get("outcomes", []):
                        name = outcome.get("name")
                        if name and name != home_raw:
                            away_raw = name
                            break
                    if away_raw:
                        break
                if away_raw:
                    break

        home_name = normalized_to_canonical.get(_normalize_team_name(home_raw))
        away_name = normalized_to_canonical.get(_normalize_team_name(away_raw))
        if not home_name or not away_name:
            continue

        home_probs: list[float] = []
        away_probs: list[float] = []
        book_updates: list[dt.datetime] = []

        for book in event.get("bookmakers", []):
            updated_at = _parse_iso_utc_timestamp(book.get("last_update"))
            if updated_at is not None:
                book_updates.append(updated_at)

            for market in book.get("markets", []):
                if market.get("key") != "h2h":
                    continue

                market_update = _parse_iso_utc_timestamp(market.get("last_update"))
                if market_update is not None:
                    book_updates.append(market_update)

                home_price = None
                away_price = None
                for outcome in market.get("outcomes", []):
                    name = outcome.get("name")
                    price = _to_float(outcome.get("price"), 0.0)

                    if name == home_raw:
                        home_price = price
                    elif name == away_raw:
                        away_price = price
                    elif away_raw is None and name and name != home_raw:
                        away_price = price

                if home_price <= 1.0 or away_price <= 1.0:
                    continue

                implied_home = 1.0 / home_price
                implied_away = 1.0 / away_price
                total = implied_home + implied_away
                if total <= 0:
                    continue

                home_probs.append(implied_home / total)
                away_probs.append(implied_away / total)

        if not home_probs or not away_probs:
            continue

        first_update = None
        last_update = None
        if book_updates:
            book_updates.sort()
            first_update = book_updates[0].isoformat()
            last_update = book_updates[-1].isoformat()

        output[(away_name, home_name)] = {
            "home_prob": float(np.mean(home_probs)),
            "away_prob": float(np.mean(away_probs)),
            "book_count": float(len(home_probs)),
            "first_update": first_update,
            "last_update": last_update,
        }

    return output


def _predict_home_prob_from_report(report: dict[str, Any], features: np.ndarray, elo_home_prob: float) -> float:
    model = report.get("model", {})
    if not model:
        weights = np.asarray(report.get("weights", []), dtype=float)
        if len(weights) != len(features):
            return float(elo_home_prob)
        return float(_sigmoid(np.asarray([features @ weights]))[0])

    model_type = str(model.get("model_type") or report.get("model_type") or "").strip().lower()
    if model_type == "tree_gbdt":
        model_path = str(model.get("model_path") or "")
        tree_model = _load_tree_model(model_path)
        if tree_model is None:
            return float(elo_home_prob)

        feature_names = model.get("feature_names") or []
        feature_count = len(feature_names)
        f = np.asarray(features, dtype=float)
        if feature_count > 0 and len(f) != feature_count:
            if len(f) > feature_count:
                f = f[:feature_count]
            else:
                f = np.concatenate([f, np.zeros(feature_count - len(f), dtype=float)])

        bundle = {
            "tree_model": tree_model,
            "blend_weight": model.get("blend_weight", 0.75),
            "calibration": model.get("calibration", {}),
        }
        probs = _predict_with_tree_bundle(bundle, X=np.asarray([f], dtype=float), elo_probs=np.asarray([elo_home_prob], dtype=float))
        return float(probs[0])

    feature_count = len(model.get("feature_names") or []) or len(model.get("weights") or [])
    f = np.asarray(features, dtype=float)
    if feature_count > 0 and len(f) != feature_count:
        if len(f) > feature_count:
            f = f[:feature_count]
        else:
            f = np.concatenate([f, np.zeros(feature_count - len(f), dtype=float)])

    bundle = {
        "weights": model.get("weights", []),
        "feature_mean": model.get("feature_mean", []),
        "feature_std": model.get("feature_std", []),
        "blend_weight": model.get("blend_weight", 0.75),
        "calibration": model.get("calibration", {}),
    }

    probs = _predict_with_model_bundle(bundle, X=np.asarray([f], dtype=float), elo_probs=np.asarray([elo_home_prob], dtype=float))
    return float(probs[0])


def get_today_matchup_predictions(
    reference_date: dt.date,
    current_season: int,
    backtest_season: int,
    model_variant: str = "v4",
    seasons_back: int = V4_MULTI_SEASON_LOOKBACK,
) -> dict[str, Any]:
    variant = _resolve_model_variant(model_variant)
    back = max(1, int(seasons_back))
    market_key = os.getenv("MSCORE_ODDS_API_KEY", "").strip()
    market_key_fingerprint = market_key[-6:] if market_key else "none"
    cache_key = f"{reference_date.isoformat()}::{current_season}::{backtest_season}::{variant}::{back}::{market_key_fingerprint}::{MARKET_FETCH_SCHEMA_VERSION}"
    with MODEL_CACHE_LOCK:
        cached = PREDICTION_CACHE.get(cache_key)
        if cached:
            return cached

    backtest = run_backtest_for_season(
        backtest_season,
        model_variant=variant,
        seasons_back=back,
    )
    model_block = backtest.get("model", {})
    rolling_mean = backtest.get("metrics", {}).get("rolling_cv", {}).get("mean", {})
    calibration_test = model_block.get("calibration_test") or backtest.get("metrics", {}).get("calibration_test", {})
    drift_monitor = model_block.get("drift_monitor") or backtest.get("metrics", {}).get("drift_monitor")
    if not drift_monitor:
        drift_monitor = _build_drift_monitor(
            backtest.get("metrics", {}).get("train", {}),
            backtest.get("metrics", {}).get("validation", {}),
            backtest.get("metrics", {}).get("test", {}),
            backtest.get("metrics", {}).get("rolling_cv", {}),
        )
    home_base_rate = _to_float(calibration_test.get("base_rate"), DEFAULT_HOME_BASE_RATE)
    if not 0.45 <= home_base_rate <= 0.65:
        home_base_rate = DEFAULT_HOME_BASE_RATE

    states, team_lookup = _build_states_before_date(reference_date=reference_date, season=current_season)
    team_context = _fetch_team_context_lookup(current_season)
    team_hitting_baseline = _fetch_team_hitting_baseline_lookup(current_season)
    advanced_team_lookup = _fetch_advanced_team_lookup(current_season)

    split_lookup = _fetch_split_lookup(current_season)
    market_lookup = _fetch_market_probabilities(reference_date, canonical_teams=set(team_lookup.values()))

    daily_games = _fetch_schedule_range(reference_date, reference_date)
    predictions: list[dict[str, Any]] = []
    pregame_overall_multipliers: list[float] = []
    market_weights_used: list[float] = []

    for game in daily_games:
        home_id = game.get("home_id")
        away_id = game.get("away_id")
        if home_id is None or away_id is None:
            continue

        home_team = team_lookup.get(int(home_id))
        away_team = team_lookup.get(int(away_id))
        if not home_team or not away_team:
            continue

        home_state = states.setdefault(home_team, TeamState())
        away_state = states.setdefault(away_team, TeamState())

        game_date = _parse_date(game.get("official_date")) or reference_date
        home_rest = _rest_days(home_state, game_date)
        away_rest = _rest_days(away_state, game_date)

        features = np.asarray(
            _build_feature_vector(home_state, away_state, home_rest=home_rest, away_rest=away_rest),
            dtype=float,
        )
        elo_home_prob = _elo_home_win_prob(home_state.elo, away_state.elo)

        model_variant_report = _resolve_model_variant(backtest.get("model_variant", variant))
        model_features = _prepare_features_for_model_variant(features, model_variant=model_variant_report)
        model_home_prob_raw = _predict_home_prob_from_report(backtest, features=model_features, elo_home_prob=elo_home_prob)
        model_home_prob, home_prior_neutralization = _neutralize_home_prior(
            model_home_prob_raw,
            home_base_rate=home_base_rate,
            strength=HOME_PRIOR_NEUTRALIZE_STRENGTH,
        )

        home_pitcher = _fetch_pitcher_profile(game.get("home_probable_pitcher_id"), season=current_season)
        away_pitcher = _fetch_pitcher_profile(game.get("away_probable_pitcher_id"), season=current_season)
        home_starter_metrics = _starter_public_metrics(home_pitcher)
        away_starter_metrics = _starter_public_metrics(away_pitcher)

        starter_adj, starter_parts = _starter_adjustment(home_pitcher, away_pitcher)
        bullpen_adj, bullpen_parts = _bullpen_context_adjustment(home_state, away_state, game_date=game_date)
        bullpen_health_adj, bullpen_health_parts = _bullpen_health_adjustment(home_state, away_state, game_date=game_date)
        luck_adj, luck_parts = _luck_factor_adjustment(home_state, away_state)
        advanced_adj, advanced_parts = _advanced_team_quality_adjustment(
            home_team=home_team,
            away_team=away_team,
            advanced_lookup=advanced_team_lookup,
        )
        split_adj, split_parts = _split_adjustment(
            home_team=home_team,
            away_team=away_team,
            split_lookup=split_lookup,
            home_starter_hand=home_pitcher.get("hand") if home_pitcher else None,
            away_starter_hand=away_pitcher.get("hand") if away_pitcher else None,
        )

        home_context = team_context.get(int(home_id), {})
        target_tz_offset = _to_float(home_context.get("tz_offset"), 0.0)
        travel_adj, travel_parts = _travel_context_adjustment(
            home_state=home_state,
            away_state=away_state,
            target_tz_offset=target_tz_offset,
            home_rest=home_rest,
            away_rest=away_rest,
        )

        lineup_context = _fetch_game_lineup_context(game.get("game_pk"))
        lineup_adj, lineup_parts = _lineup_context_adjustment(lineup_context)
        home_baseline_ops = team_hitting_baseline.get(int(home_id))
        away_baseline_ops = team_hitting_baseline.get(int(away_id))
        lineup_health_adj, lineup_health_parts = _lineup_health_adjustment(
            lineup_context=lineup_context,
            home_baseline_ops=home_baseline_ops,
            away_baseline_ops=away_baseline_ops,
        )

        pregame_gates = _pregame_context_gates(lineup_context=lineup_context, home_pitcher=home_pitcher, away_pitcher=away_pitcher)
        starter_multiplier = _to_float(pregame_gates.get("starter_multiplier"), 1.0)
        lineup_multiplier = _to_float(pregame_gates.get("lineup_multiplier"), 1.0)
        split_multiplier = _to_float(pregame_gates.get("split_multiplier"), 1.0)

        starter_adj *= starter_multiplier
        split_adj *= split_multiplier
        lineup_adj *= lineup_multiplier
        lineup_health_adj *= lineup_multiplier

        starter_parts = _scale_adjustment_fields(
            starter_parts,
            starter_multiplier,
            keys=(
                "starter_adjustment",
                "starter_era_edge",
                "starter_whip_edge",
                "starter_fip_edge",
                "starter_kbb_edge",
                "starter_qs_edge",
            ),
        )
        split_parts = _scale_adjustment_fields(
            split_parts,
            split_multiplier,
            keys=(
                "split_adjustment",
                "split_home_away_edge",
                "split_handedness_edge",
                "split_last10_edge",
            ),
        )
        lineup_parts = _scale_adjustment_fields(
            lineup_parts,
            lineup_multiplier,
            keys=("lineup_adjustment", "lineup_ops_edge", "lineup_confirmed_edge"),
        )
        lineup_health_parts = _scale_adjustment_fields(
            lineup_health_parts,
            lineup_multiplier,
            keys=("lineup_health_adjustment", "lineup_health_edge", "lineup_health_confirmed_edge"),
        )

        pregame_overall_multipliers.append(_to_float(pregame_gates.get("overall_multiplier"), 1.0))

        adjusted_home_prob = _clamp_prob(
            model_home_prob
            + starter_adj
            + bullpen_adj
            + bullpen_health_adj
            + split_adj
            + travel_adj
            + lineup_adj
            + lineup_health_adj
            + luck_adj
            + advanced_adj,
        )

        market_entry = market_lookup.get((away_team, home_team))
        market_home_prob = float(market_entry.get("home_prob")) if market_entry and market_entry.get("home_prob") is not None else None
        market_book_count = _to_int(market_entry.get("book_count"), 0) if market_entry else 0
        market_first_update = market_entry.get("first_update") if market_entry else None
        market_last_update = market_entry.get("last_update") if market_entry else None

        pre_market_uncertainty = _estimate_prediction_uncertainty(
            home_prob=adjusted_home_prob,
            calibration_test=calibration_test,
            market_weight=0.0,
        )

        market_weight, market_blend = _compute_market_blend_weight(
            pre_market_home_prob=adjusted_home_prob,
            market_home_prob=market_home_prob,
            market_book_count=market_book_count,
            market_last_update=market_last_update,
            reference_date=reference_date,
            pre_market_uncertainty_level=str(pre_market_uncertainty.get("level") or "low"),
        )

        final_home_prob = adjusted_home_prob
        if market_home_prob is not None:
            final_home_prob = _clamp_prob((1.0 - market_weight) * adjusted_home_prob + market_weight * market_home_prob)
            market_weights_used.append(float(market_weight))

        home_win_prob = float(final_home_prob)
        away_win_prob = float(1.0 - home_win_prob)
        favored = home_team if home_win_prob >= away_win_prob else away_team
        favored_prob = max(home_win_prob, away_win_prob)

        uncertainty_profile = _estimate_prediction_uncertainty(
            home_prob=home_win_prob,
            calibration_test=calibration_test,
            market_weight=market_weight,
        )

        edge_fields = _market_edge_fields(
            home_team=home_team,
            away_team=away_team,
            home_win_prob=home_win_prob,
            pre_market_home_prob=adjusted_home_prob,
            market_home_prob=market_home_prob,
        )

        row: dict[str, Any] = {
            "game_pk": game.get("game_pk"),
            "official_date": game.get("official_date"),
            "game_date": game.get("game_date"),
            "status": game.get("status"),
            "state": game.get("state"),
            "away_team": away_team,
            "home_team": home_team,
            "away_win_prob": round(away_win_prob, 4),
            "home_win_prob": round(home_win_prob, 4),
            "favored_team": favored,
            "favored_win_prob": round(favored_prob, 4),
            "confidence": round(abs(home_win_prob - 0.5) * 2, 4),
            "model_home_win_prob": round(model_home_prob, 4),
            "model_home_win_prob_raw": round(float(model_home_prob_raw), 4),
            "home_prior_base_rate": round(float(home_base_rate), 4),
            "home_prior_neutralize_strength": round(float(HOME_PRIOR_NEUTRALIZE_STRENGTH), 2),
            "home_prior_neutralization": round(float(home_prior_neutralization), 4),
            "pre_market_home_win_prob": round(float(adjusted_home_prob), 4),
            "elo_home_win_prob": round(float(elo_home_prob), 4),
            "starter_adjustment": round(float(starter_adj), 4),
            "split_adjustment": round(float(split_adj), 4),
            "bullpen_adjustment": round(float(bullpen_adj), 4),
            "bullpen_health_adjustment": round(float(bullpen_health_adj), 4),
            "travel_adjustment": round(float(travel_adj), 4),
            "lineup_adjustment": round(float(lineup_adj), 4),
            "lineup_health_adjustment": round(float(lineup_health_adj), 4),
            "luck_adjustment": round(float(luck_adj), 4),
            "advanced_adjustment": round(float(advanced_adj), 4),
            "market_home_win_prob": round(float(market_home_prob), 4) if market_home_prob is not None else None,
            "market_weight": round(float(market_weight), 2),
            "market_weight_policy": market_blend.get("policy"),
            "market_weight_reason": market_blend.get("reason"),
            "market_weight_base": market_blend.get("base"),
            "market_weight_book_factor": market_blend.get("book_factor"),
            "market_weight_freshness_factor": market_blend.get("freshness_factor"),
            "market_weight_uncertainty_bonus": market_blend.get("uncertainty_bonus"),
            "market_weight_disagreement_bonus": market_blend.get("disagreement_bonus"),
            "market_last_update_age_hours": market_blend.get("line_age_hours"),
            "market_book_count": int(market_book_count),
            "market_first_update": market_first_update,
            "market_last_update": market_last_update,
            "model_pick_team": edge_fields.get("model_pick_team"),
            "market_favored_team": edge_fields.get("market_favored_team"),
            "market_favored_prob": round(float(edge_fields.get("market_favored_prob")), 4) if edge_fields.get("market_favored_prob") is not None else None,
            "model_vs_market_edge_home": round(float(edge_fields.get("model_vs_market_edge_home")), 4) if edge_fields.get("model_vs_market_edge_home") is not None else None,
            "model_vs_market_edge_pick": round(float(edge_fields.get("model_vs_market_edge_pick")), 4) if edge_fields.get("model_vs_market_edge_pick") is not None else None,
            "value_tier": edge_fields.get("value_tier", "none"),
            "market_disagrees_with_model": bool(edge_fields.get("market_disagrees_with_model", False)),
            "pre_market_uncertainty_level": pre_market_uncertainty.get("level"),
            "uncertainty_level": uncertainty_profile.get("level"),
            "high_uncertainty": bool(uncertainty_profile.get("high_uncertainty")),
            "pregame_context_multiplier": round(float(pregame_gates.get("overall_multiplier", 1.0)), 4),
            "pregame_lineup_multiplier": round(float(pregame_gates.get("lineup_multiplier", 1.0)), 4),
            "pregame_starter_multiplier": round(float(pregame_gates.get("starter_multiplier", 1.0)), 4),
            "pregame_split_multiplier": round(float(pregame_gates.get("split_multiplier", 1.0)), 4),
            "pregame_lineup_readiness": pregame_gates.get("lineup_readiness"),
            "pregame_starter_readiness": pregame_gates.get("starter_readiness"),
            "pregame_overall_readiness": pregame_gates.get("overall_readiness"),
            "pregame_home_lineup_ready": bool(pregame_gates.get("home_lineup_ready")),
            "pregame_away_lineup_ready": bool(pregame_gates.get("away_lineup_ready")),
            "pregame_home_starter_ready": bool(pregame_gates.get("home_starter_ready")),
            "pregame_away_starter_ready": bool(pregame_gates.get("away_starter_ready")),
            "home_win_prob_band_low": uncertainty_profile.get("band_low"),
            "home_win_prob_band_high": uncertainty_profile.get("band_high"),
            "home_win_prob_band_half": uncertainty_profile.get("band_half"),
            "uncertainty_bin_count": uncertainty_profile.get("bin_count"),
            "uncertainty_calibration_gap": uncertainty_profile.get("calibration_gap"),
            "uncertainty_edge_multiplier": uncertainty_profile.get("edge_multiplier"),
            "uncertainty_note": uncertainty_profile.get("note"),
            "home_probable_pitcher": home_starter_metrics.get("starter_name") or game.get("home_probable_pitcher_name"),
            "away_probable_pitcher": away_starter_metrics.get("starter_name") or game.get("away_probable_pitcher_name"),
        }

        row.update(starter_parts)
        row.update(bullpen_parts)
        row.update(bullpen_health_parts)
        row.update(split_parts)
        row.update(travel_parts)
        row.update(lineup_parts)
        row.update(lineup_health_parts)
        row.update(luck_parts)
        row.update(advanced_parts)

        for key, value in home_starter_metrics.items():
            row[f"home_{key}"] = value
        for key, value in away_starter_metrics.items():
            row[f"away_{key}"] = value

        if game.get("away_score") is not None and game.get("home_score") is not None:
            away_score = int(game.get("away_score"))
            home_score = int(game.get("home_score"))
            row["away_score"] = away_score
            row["home_score"] = home_score
            actual_winner = home_team if home_score > away_score else away_team
            predicted_winner = home_team if home_win_prob >= 0.5 else away_team
            row["actual_winner"] = actual_winner
            row["predicted_winner"] = predicted_winner
            row["prediction_correct"] = bool(actual_winner == predicted_winner)

        predictions.append(row)

    predictions.sort(key=lambda item: (item.get("game_date") or "", item.get("game_pk") or 0))

    market_odds_enabled = bool(os.getenv("MSCORE_ODDS_API_KEY", "").strip())
    market_matchups_with_lines = sum(1 for row in predictions if row.get("market_home_win_prob") is not None)
    if not market_odds_enabled:
        market_status = "disabled_missing_api_key"
    elif market_matchups_with_lines <= 0:
        market_status = "enabled_no_lines_returned"
    else:
        market_status = "enabled_live"

    pregame_avg_multiplier = float(np.mean(pregame_overall_multipliers)) if pregame_overall_multipliers else None
    pregame_low_readiness_games = sum(1 for value in pregame_overall_multipliers if value < 0.74)
    market_avg_weight = float(np.mean(market_weights_used)) if market_weights_used else None

    result = {
        "reference_date": reference_date.isoformat(),
        "season": current_season,
        "backtest_season": backtest_season,
        "model_variant": _resolve_model_variant(model_variant),
        "seasons_back": max(1, int(seasons_back)),
        "model": {
            "model_version": backtest.get("model_version"),
            "model_variant": backtest.get("model_variant", variant),
            "seasons_back": backtest.get("seasons_back", back),
            "feature_names": model_block.get("feature_names", backtest.get("feature_names", FEATURE_NAMES)),
            "weights": model_block.get("weights", backtest.get("weights", [])),
            "blend_weight": model_block.get("blend_weight", 0.75),
            "calibration": model_block.get("calibration", {}),
            "metrics_test": backtest.get("metrics", {}).get("test", {}),
            "rolling_cv_mean": rolling_mean,
            "calibration_test": calibration_test,
            "home_prior_base_rate": round(float(home_base_rate), 4),
            "home_prior_neutralize_strength": round(float(HOME_PRIOR_NEUTRALIZE_STRENGTH), 2),
            "uncertainty_policy": {
                "high_band_threshold": 0.11,
                "medium_band_threshold": 0.075,
                "high_edge_multiplier": 0.55,
                "medium_edge_multiplier": 0.8,
            },
            "feature_importance": model_block.get("feature_importance", backtest.get("feature_importance", [])),
            "ablation": model_block.get("ablation", backtest.get("ablation", {})),
            "drift_monitor": drift_monitor or {},
            "market_blend_policy": {
                "policy": "dynamic_v1",
                "min_weight": MARKET_BLEND_MIN_WEIGHT,
                "max_weight": MARKET_BLEND_MAX_WEIGHT,
            },
            "market_blend_summary": {
                "avg_market_weight": round(float(market_avg_weight), 4) if market_avg_weight is not None else None,
                "games_with_market": len(market_weights_used),
            },
            "pregame_context_summary": {
                "avg_multiplier": round(float(pregame_avg_multiplier), 4) if pregame_avg_multiplier is not None else None,
                "low_readiness_games": int(pregame_low_readiness_games),
                "readiness_pct": round(float((pregame_avg_multiplier or 0.0) * 100.0), 1) if pregame_avg_multiplier is not None else None,
            },
            "trained_at": backtest.get("trained_at"),
        },
        "games": predictions,
        "market_odds_enabled": market_odds_enabled,
        "market_status": market_status,
        "market_matchups_with_lines": market_matchups_with_lines,
    }

    with MODEL_CACHE_LOCK:
        PREDICTION_CACHE[cache_key] = result

    return result


def _prediction_archive_path(reference_date: dt.date) -> Path:
    return PREDICTION_ARCHIVE_DIR / f"{reference_date.isoformat()}.json"


def _archive_summary(games: list[dict[str, Any]]) -> dict[str, Any]:
    finalized = [game for game in games if game.get("actual_winner") and game.get("predicted_winner")]
    correct = sum(1 for game in finalized if game.get("prediction_correct"))
    accuracy = (correct / len(finalized)) if finalized else None

    return {
        "games_total": len(games),
        "games_final": len(finalized),
        "correct_picks": correct,
        "accuracy": round(float(accuracy), 4) if accuracy is not None else None,
    }


def write_daily_prediction_archive(
    reference_date: dt.date,
    current_season: int,
    backtest_season: int,
    force_rebuild: bool = False,
    model_variant: str = "v4",
    seasons_back: int = V4_MULTI_SEASON_LOOKBACK,
) -> dict[str, Any]:
    path = _prediction_archive_path(reference_date)
    if path.exists() and not force_rebuild:
        return json.loads(path.read_text(encoding="utf-8"))

    bundle = get_today_matchup_predictions(
        reference_date=reference_date,
        current_season=current_season,
        backtest_season=backtest_season,
        model_variant=model_variant,
        seasons_back=seasons_back,
    )

    payload = {
        "archived_at": dt.datetime.now().isoformat(),
        "reference_date": reference_date.isoformat(),
        "season": current_season,
        "backtest_season": backtest_season,
        "model": bundle.get("model", {}),
        "market_odds_enabled": bool(bundle.get("market_odds_enabled")),
        "market_status": bundle.get("market_status"),
        "market_matchups_with_lines": bundle.get("market_matchups_with_lines"),
        "summary": _archive_summary(bundle.get("games", [])),
        "games": bundle.get("games", []),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def read_prediction_archive(reference_date: dt.date) -> dict[str, Any] | None:
    path = _prediction_archive_path(reference_date)
    if not path.exists():
        return None

    return json.loads(path.read_text(encoding="utf-8"))


def list_prediction_archives(limit: int = 30) -> list[dict[str, Any]]:
    if not PREDICTION_ARCHIVE_DIR.exists():
        return []

    files = sorted(PREDICTION_ARCHIVE_DIR.glob("*.json"), key=lambda item: item.name, reverse=True)
    rows: list[dict[str, Any]] = []

    for path in files[: max(1, limit)]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        summary = payload.get("summary", {})
        rows.append(
            {
                "reference_date": payload.get("reference_date"),
                "archived_at": payload.get("archived_at"),
                "games_total": summary.get("games_total"),
                "games_final": summary.get("games_final"),
                "accuracy": summary.get("accuracy"),
                "model_trained_at": payload.get("model", {}).get("trained_at"),
                "path": str(path),
            }
        )

    return rows


def get_prediction_archive_status(limit: int = 30) -> dict[str, Any]:
    entries = list_prediction_archives(limit=limit)
    latest = entries[0] if entries else None
    return {
        "archive_count": len(entries),
        "latest": latest,
        "entries": entries,
    }


def run_nightly_retrain_and_archive(
    reference_date: dt.date | None = None,
    model_variant: str = "v4",
    seasons_back: int = V4_MULTI_SEASON_LOOKBACK,
) -> dict[str, Any]:
    if reference_date is None:
        reference_date = dt.date.today()

    backtest_season = reference_date.year - 1
    retrain = run_backtest_for_season(
        season=backtest_season,
        force_retrain=True,
        model_variant=model_variant,
        seasons_back=seasons_back,
    )

    archive_date = reference_date - dt.timedelta(days=1)
    archive_payload = write_daily_prediction_archive(
        reference_date=archive_date,
        current_season=archive_date.year,
        backtest_season=archive_date.year - 1,
        force_rebuild=True,
        model_variant=model_variant,
        seasons_back=seasons_back,
    )

    return {
        "ran_at": dt.datetime.now().isoformat(),
        "retrained_season": backtest_season,
        "retrained_at": retrain.get("trained_at"),
        "model_variant": retrain.get("model_variant", _resolve_model_variant(model_variant)),
        "seasons_back": retrain.get("seasons_back", max(1, int(seasons_back))),
        "archived_date": archive_date.isoformat(),
        "archive_summary": archive_payload.get("summary", {}),
    }


def get_last_season_report(
    reference_date: dt.date,
    model_variant: str = "v4",
    seasons_back: int = V4_MULTI_SEASON_LOOKBACK,
) -> dict[str, Any]:
    backtest_season = reference_date.year - 1
    return run_backtest_for_season(
        backtest_season,
        model_variant=model_variant,
        seasons_back=seasons_back,
    )
