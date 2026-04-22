from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from app.config import CACHE_PATH, REFRESH_INTERVAL_MINUTES, SNAPSHOT_ARCHIVE_DIR, TIMEZONE, WORKBOOK_PATH
from app.services.mlb_service import fetch_games_window, fetch_standings, fetch_team_api_metrics
from app.services.prediction_service import (
    get_last_season_report,
    get_prediction_archive_status,
    get_today_matchup_predictions,
    read_prediction_archive,
)

OPS_KEY = "mlb::season::hitting::ops"
OBP_KEY = "mlb::season::hitting::obp"
SLG_KEY = "mlb::season::hitting::slg"
ISO_ADV_KEY = "mlb::seasonAdvanced::hitting::iso"
HITTING_WALKS_PER_STRIKEOUT_KEY = "mlb::seasonAdvanced::hitting::walksPerStrikeout"
PLATE_APPEARANCES_KEY = "mlb::season::hitting::plateAppearances"
HITTING_STOLEN_BASE_PCT_KEY = "mlb::season::hitting::stolenBasePercentage"
HITTING_STOLEN_BASES_KEY = "mlb::season::hitting::stolenBases"
HITTING_CAUGHT_STEALING_KEY = "mlb::season::hitting::caughtStealing"

RUNS_SCORED_KEY = "mlb::season::hitting::runs"
RUNS_ALLOWED_KEY = "mlb::season::pitching::runs"
ERA_KEY = "mlb::season::pitching::era"
WHIP_KEY = "mlb::season::pitching::whip"
PITCHING_KBB_KEY = "mlb::season::pitching::strikeoutWalkRatio"
PITCHING_HR9_KEY = "mlb::season::pitching::homeRunsPer9"
PITCHING_WHIFF_PCT_KEY = "mlb::seasonAdvanced::pitching::whiffPercentage"
PITCHING_GAMES_STARTED_KEY = "mlb::season::pitching::gamesStarted"
PITCHING_QUALITY_STARTS_KEY = "mlb::seasonAdvanced::pitching::qualityStarts"
PITCHING_K_MINUS_BB_PCT_KEY = "mlb::seasonAdvanced::pitching::strikeoutsMinusWalksPercentage"
SAVES_KEY = "mlb::season::pitching::saves"
BLOWN_SAVES_KEY = "mlb::season::pitching::blownSaves"
SAVE_OPPORTUNITIES_KEY = "mlb::season::pitching::saveOpportunities"
HOLDS_KEY = "mlb::season::pitching::holds"

FIELDING_KEY = "mlb::season::fielding::fielding"
FIELDING_CAUGHT_STEALING_PCT_KEY = "mlb::season::fielding::caughtStealingPercentage"
GAMES_PLAYED_KEY = "mlb::season::hitting::gamesPlayed"

HITS_KEY = "mlb::season::hitting::hits"
DOUBLES_KEY = "mlb::season::hitting::doubles"
TRIPLES_KEY = "mlb::season::hitting::triples"
HOME_RUNS_KEY = "mlb::season::hitting::homeRuns"
AT_BATS_KEY = "mlb::season::hitting::atBats"
SAC_FLIES_KEY = "mlb::season::hitting::sacFlies"
HITTING_STRIKEOUTS_KEY = "mlb::season::hitting::strikeOuts"
HITTING_BALLS_IN_PLAY_ADV_KEY = "mlb::seasonAdvanced::hitting::ballsInPlay"
HITTING_LINE_HITS_ADV_KEY = "mlb::seasonAdvanced::hitting::lineHits"
HITTING_FLY_HITS_ADV_KEY = "mlb::seasonAdvanced::hitting::flyHits"

PITCHING_STRIKEOUTS_PER9_ADV_KEY = "mlb::seasonAdvanced::pitching::strikeoutsPer9"
PITCHING_BB_PER9_ADV_KEY = "mlb::seasonAdvanced::pitching::baseOnBallsPer9"
PITCHING_HR_PER9_ADV_KEY = "mlb::seasonAdvanced::pitching::homeRunsPer9"
PITCHING_FLY_BALL_PCT_ADV_KEY = "mlb::seasonAdvanced::pitching::flyBallPercentage"
PITCHING_OPS_ALLOWED_KEY = "mlb::season::pitching::ops"

HITTING_OPS_RISP_KEY = "mlb::situational::hitting::risp::ops"
HITTING_OPS_RISP2_KEY = "mlb::situational::hitting::risp2::ops"
HITTING_OPS_LATE_CLOSE_KEY = "mlb::situational::hitting::lc::ops"
PITCHING_OPS_RISP_ALLOWED_KEY = "mlb::situational::pitching::risp::ops"
PITCHING_OPS_RISP2_ALLOWED_KEY = "mlb::situational::pitching::risp2::ops"
PITCHING_OPS_LATE_CLOSE_ALLOWED_KEY = "mlb::situational::pitching::lc::ops"

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

RANKING_TABLE_DEFINITIONS = [
    {"key": "live_rank", "label": "Live", "description": "Current live rank from MLB API metrics."},
    {"key": "workbook_rank", "label": "Template", "description": "Workbook template rank is disabled in API-first mode."},
    {"key": "team", "label": "Team", "description": "MLB team name."},
    {"key": "mscore", "label": "Mscore", "description": "Friend-style blend (improved): Base + Offense + Defense + Context buckets on an absolute sigmoid 0-100 scale (not league min-max)."},
    {"key": "live_power_score", "label": "Power", "description": "Context-adjusted formula: Mscore plus Talent/Form/Risk overlay with elite-team dampening, confidence scaling, capped adjustment range, and a 5% coaching/execution overlay when the signal is meaningful."},
    {"key": "d_plus_score", "label": "D+", "description": "Defense-plus score: near-equal Base/Offense/Defense/Context blend with extra pitching quality/health emphasis."},
    {"key": "joe_score", "label": "Joe", "description": "Joe Score v3: fixed 45/40/15 blend (xwOBAcon / xFIP-inverse / clutch execution) with sample-size shrinkage and uncertainty control (higher is better)."},
    {"key": "power_minus_d_plus", "label": "Power-D+", "description": "Power minus D+ score. Positive means context/timing boosts above defense-plus baseline; negative means caution vs D+ baseline."},
    {"key": "leverage_net_quality", "label": "LevQ", "description": "Combined pressure-situation quality from offense (RISP/late-close) and pitching run prevention in those same spots."},
    {"key": "clutch_index", "label": "Clutch", "description": "Net clutch execution index versus season baseline using late/close and RISP two-out performance."},
    {"key": "record", "label": "W-L", "description": "Current season wins and losses from MLB standings."},
    {"key": "win_pct", "label": "Win%", "description": "Current winning percentage from MLB standings."},
    {"key": "run_diff", "label": "RD", "description": "Run differential from MLB standings."},
    {"key": "streak", "label": "Streak", "description": "Current win/loss streak from MLB standings."},
    {"key": "projected_wins_162", "label": "Proj W", "description": "Projected wins over 162 games from API-based model."},
]

LIVE_STAT_CATALOG = [
    {"key": "live_rank", "label": "Live Rank", "description": "Current rank from API-derived composite score.", "group": "Live Model"},
    {"key": "workbook_rank", "label": "Template Rank", "description": "Workbook template rank disabled in API-first mode.", "group": "Live Model"},
    {"key": "rank_delta", "label": "Rank Delta", "description": "Rank delta is disabled in API-first mode.", "group": "Live Model"},
    {"key": "mscore", "label": "Mscore", "description": "Friend-style blend (improved): Base + Offense + Defense + Context buckets on an absolute sigmoid 0-100 scale (not league min-max).", "group": "Live Model"},
    {"key": "mscore_base_component", "label": "Mscore Base", "description": "Series/sweep-style base proxy from win profile and resilience signals.", "group": "Live Model"},
    {"key": "mscore_offense_component", "label": "Mscore Offense", "description": "Offense bucket proxy (wRC+/BsR/PA/Runs style blend) with friend-style weighting.", "group": "Live Model"},
    {"key": "mscore_defense_component", "label": "Mscore Defense", "description": "Defense/run-prevention bucket with starter/bullpen weighting and fielding/catching proxies.", "group": "Live Model"},
    {"key": "mscore_context_component", "label": "Mscore Context", "description": "Small context bucket (pythag/regression and pitching differential signals).", "group": "Live Model"},
    {"key": "xwobacon_proxy", "label": "xwOBAcon Proxy", "description": "Expected contact quality proxy: uses direct API xwOBAcon when available, otherwise derives from contact run value and batted-ball mix.", "group": "Live Model"},
    {"key": "xfip_proxy", "label": "xFIP Proxy", "description": "Expected run-prevention proxy: uses direct API xFIP when available, otherwise derives from regressed HR/9 with K/9, BB/9, and fly-ball profile. Lower is better.", "group": "Live Model"},
    {"key": "hitting_ops_risp", "label": "OPS (RISP)", "description": "Team OPS with runners in scoring position (MLB situation split: risp).", "group": "Leverage / Clutch"},
    {"key": "hitting_ops_risp2", "label": "OPS (RISP, 2 Out)", "description": "Team OPS with runners in scoring position and two outs (split: risp2).", "group": "Leverage / Clutch"},
    {"key": "hitting_ops_late_close", "label": "OPS (Late/Close)", "description": "Team OPS in late/close situations (split: lc).", "group": "Leverage / Clutch"},
    {"key": "pitching_ops_risp_allowed", "label": "Opp OPS (RISP)", "description": "Opponent OPS allowed by team pitching with runners in scoring position.", "group": "Leverage / Clutch"},
    {"key": "pitching_ops_risp2_allowed", "label": "Opp OPS (RISP, 2 Out)", "description": "Opponent OPS allowed by team pitching with RISP and two outs.", "group": "Leverage / Clutch"},
    {"key": "pitching_ops_late_close_allowed", "label": "Opp OPS (Late/Close)", "description": "Opponent OPS allowed by team pitching in late/close situations.", "group": "Leverage / Clutch"},
    {"key": "leverage_offense_quality", "label": "Leverage Offense", "description": "0-100 pressure offense quality blend from OPS in RISP/RISP2/Late-Close contexts.", "group": "Leverage / Clutch"},
    {"key": "leverage_pitching_quality", "label": "Leverage Pitching", "description": "0-100 pressure pitching quality blend from opponent OPS allowed in RISP/RISP2/Late-Close contexts.", "group": "Leverage / Clutch"},
    {"key": "leverage_net_quality", "label": "Leverage Net", "description": "Combined leverage quality (offense + pitching), higher is better.", "group": "Leverage / Clutch"},
    {"key": "clutch_index", "label": "Clutch Index", "description": "0-100 clutch execution index versus baseline in pressure situations.", "group": "Leverage / Clutch"},
    {"key": "clutch_offense_delta_ops", "label": "Clutch Off OPS Delta", "description": "Offense clutch delta: late/close and RISP2 OPS versus season OPS baseline.", "group": "Leverage / Clutch"},
    {"key": "clutch_pitching_delta_ops", "label": "Clutch Pitch OPS Delta", "description": "Pitching clutch delta: season OPS allowed minus late/close and RISP2 OPS allowed.", "group": "Leverage / Clutch"},
    {"key": "joe_score", "label": "Joe Score", "description": "Joe Score v3: fixed 45/40/15 blend (xwOBAcon / xFIP-inverse / clutch execution) with sample-size shrinkage toward league baseline.", "group": "Live Model"},
    {"key": "joe_rank", "label": "Joe Rank", "description": "Rank of Joe Score across all MLB teams (1 = best combined xwOBAcon + xFIP + clutch profile).", "group": "Live Model"},
    {"key": "joe_confidence", "label": "Joe Confidence", "description": "Confidence (0-100) in Joe Score after sample-size and signal-agreement checks.", "group": "Live Model"},
    {"key": "joe_band_low", "label": "Joe Band Low", "description": "Lower uncertainty band for Joe Score.", "group": "Live Model"},
    {"key": "joe_band_high", "label": "Joe Band High", "description": "Upper uncertainty band for Joe Score.", "group": "Live Model"},
    {"key": "joe_uncertainty_level", "label": "Joe Uncertainty", "description": "Qualitative Joe uncertainty bucket: low/medium/high.", "group": "Live Model"},
    {"key": "joe_weight_xwobacon_pct", "label": "Joe Weight xwOBAcon", "description": "Fixed Joe model weight (%) assigned to xwOBAcon side.", "group": "Live Model"},
    {"key": "joe_weight_xfip_pct", "label": "Joe Weight xFIP", "description": "Fixed Joe model weight (%) assigned to inverse xFIP side.", "group": "Live Model"},
    {"key": "joe_weight_clutch_pct", "label": "Joe Weight Clutch", "description": "Fixed Joe model weight (%) assigned to clutch execution side (leverage + clutch index).", "group": "Live Model"},
    {"key": "live_power_score", "label": "Live Power Score", "description": "Context-adjusted formula: Mscore plus Talent/Form/Risk overlay with elite-team dampening, confidence scaling, capped adjustment range, and a 5% coaching/execution overlay when the signal is meaningful.", "group": "Live Model"},
    {"key": "d_plus_score", "label": "D+ Score", "description": "Defense-plus score: near-equal Base/Offense/Defense/Context blend with extra pitching quality/health emphasis.", "group": "Live Model"},
    {"key": "power_minus_d_plus", "label": "Power-D+", "description": "Power minus D+ score. Positive means context/timing boosts above defense-plus baseline; negative means caution vs D+ baseline.", "group": "Live Model"},
    {"key": "talent_score", "label": "Talent Score", "description": "Underlying talent blend from baseline quality, offense, run prevention, and defense.", "group": "Live Model"},
    {"key": "form_score", "label": "Form Score", "description": "Recent form/momentum blend using streak, scoring trend, and near-term movement.", "group": "Live Model"},
    {"key": "risk_score", "label": "Risk Score", "description": "Caution index from bullpen fragility, overperformance risk, sample risk, and run prevention volatility. Higher = more risk.", "group": "Live Model"},
    {"key": "power_context_adjustment", "label": "Power Context Adj", "description": "Net adjustment (in points) applied to Mscore to produce Power after dampening/caps, including coaching overlay when meaningful.", "group": "Live Model"},
    {"key": "power_context_confidence", "label": "Power Confidence", "description": "Confidence in context overlay based on season sample size; low sample shrinks adjustments.", "group": "Live Model"},
    {"key": "coaching_execution_score", "label": "Coaching/Execution Score", "description": "Proxy score (0-100) for tactical execution quality using sequencing, bullpen closeout efficiency, baserunning, and discipline/defense edges.", "group": "Live Model"},
    {"key": "coaching_execution_adjustment", "label": "Coaching Adj", "description": "Coaching/execution contribution in points before the 5% blend into Power; positive supports current performance.", "group": "Live Model"},
    {"key": "coaching_execution_confidence", "label": "Coaching Confidence", "description": "Team-level confidence (0-100) after sample shrinkage and league signal-strength gating.", "group": "Live Model"},
    {"key": "coaching_execution_weight", "label": "Coaching Weight", "description": "Effective Power blend weight (%) for coaching factor, capped at 5 and set near 0 when signal is weak.", "group": "Live Model"},
    {"key": "wins", "label": "Live Wins", "description": "Official current MLB wins from standings.", "group": "Live Results"},
    {"key": "losses", "label": "Live Losses", "description": "Official current MLB losses from standings.", "group": "Live Results"},
    {"key": "win_pct", "label": "Live Win%", "description": "Current winning percentage from standings.", "group": "Live Results"},
    {"key": "run_diff", "label": "Run Differential", "description": "Runs scored minus runs allowed from standings.", "group": "Live Results"},
    {"key": "streak", "label": "Streak", "description": "Current streak code.", "group": "Live Results"},
    {"key": "games_back", "label": "Games Back", "description": "Games behind division leader.", "group": "Live Results"},
    {"key": "projected_win_pct", "label": "Projected Win%", "description": "Blended forecast based on win% and power score.", "group": "Projection"},
    {"key": "projected_wins_162", "label": "Projected Wins (162)", "description": "Projected full-season wins over 162 games.", "group": "Projection"},
]


def _safe_iso_date(value: Any) -> dt.date | None:
    if not value:
        return None
    text = str(value)
    if len(text) >= 10:
        text = text[:10]
    try:
        return dt.date.fromisoformat(text)
    except ValueError:
        return None



def _streak_score(value: Any) -> float:
    text = str(value or "").strip().upper()
    if len(text) < 2:
        return 0.0

    sign = 0.0
    if text.startswith("W"):
        sign = 1.0
    elif text.startswith("L"):
        sign = -1.0
    else:
        return 0.0

    digits = ""
    for ch in text[1:]:
        if ch.isdigit():
            digits += ch
        else:
            break

    length = float(int(digits)) if digits else 0.0
    return sign * min(10.0, length) / 10.0


def _select_relevant_games(games_window: list[dict[str, Any]], today: dt.date) -> list[dict[str, Any]]:
    today_games = [game for game in games_window if _safe_iso_date(game.get("official_date")) == today]
    live_spillover = [
        game
        for game in games_window
        if game.get("is_live") and _safe_iso_date(game.get("official_date")) != today
    ]

    ordered = live_spillover + today_games
    seen: set[Any] = set()
    selected: list[dict[str, Any]] = []
    for game in ordered:
        game_pk = game.get("game_pk")
        if game_pk in seen:
            continue
        seen.add(game_pk)
        selected.append(game)

    return selected[:20]


def _normalize_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().all():
        return pd.Series([0.5] * len(series), index=series.index)

    filled = numeric.fillna(numeric.median())
    min_value = filled.min()
    max_value = filled.max()

    if max_value - min_value == 0:
        return pd.Series([0.5] * len(series), index=series.index)

    return (filled - min_value) / (max_value - min_value)


def _abs_corr_series(left: pd.Series, right: pd.Series) -> float:
    left_num = pd.to_numeric(left, errors="coerce")
    right_num = pd.to_numeric(right, errors="coerce")
    joined = pd.concat([left_num, right_num], axis=1).dropna()

    if joined.empty:
        return 0.0
    if joined.iloc[:, 0].nunique(dropna=True) <= 1 or joined.iloc[:, 1].nunique(dropna=True) <= 1:
        return 0.0

    corr = joined.iloc[:, 0].corr(joined.iloc[:, 1])
    if pd.isna(corr):
        return 0.0

    return abs(float(corr))


def _signed_scale(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    if numeric.empty:
        return pd.Series([], dtype=float)

    # Winsorize to limit outlier distortion, then scale to [-1, 1] like workbook point terms.
    low = float(numeric.quantile(0.02))
    high = float(numeric.quantile(0.98))
    clipped = numeric.clip(lower=low, upper=high)

    positives = clipped[clipped > 0]
    negatives = clipped[clipped < 0]
    pos_max = float(positives.max()) if not positives.empty else 1.0
    neg_min = float(negatives.min()) if not negatives.empty else -1.0

    def _scale(value: float) -> float:
        if value > 0:
            return float(value / pos_max) if pos_max != 0 else 0.0
        if value < 0:
            return float(value / abs(neg_min)) if neg_min != 0 else 0.0
        return 0.0

    return clipped.apply(_scale).clip(-1.0, 1.0)


def _to_native(value: Any, digits: int | None = None) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        if math.isnan(float(value)):
            return None
        native = float(value)
        if digits is not None:
            return round(native, digits)
        if native.is_integer():
            return int(native)
        return round(native, 3)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if pd.isna(value):
        return None
    return value


def _first_api_metric(api_stats: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in api_stats and api_stats.get(key) not in {None, "", "-", "--"}:
            return api_stats.get(key)
    return None


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None

    if math.isnan(out) or math.isinf(out):
        return None

    return out


def _as_ratio(series: pd.Series) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce").astype(float)
    clean = clean.replace([np.inf, -np.inf], np.nan)
    return clean.where(clean <= 1.5, clean / 100.0)


def _clamp_prob(prob: float) -> float:
    return float(max(0.02, min(0.98, prob)))


def _pick_team_from_home_prob(home_prob: float, home_team: str, away_team: str) -> str:
    return home_team if float(home_prob) >= 0.5 else away_team


def _pick_probability(home_prob: float, pick_team: str, home_team: str, away_team: str) -> float:
    if pick_team == home_team:
        return float(home_prob)
    if pick_team == away_team:
        return float(1.0 - home_prob)
    return float(max(home_prob, 1.0 - home_prob))


def _value_tier(edge_pick: float | None) -> str:
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


def _derive_pre_market_home_prob(game: dict[str, Any]) -> float:
    existing = _safe_float(game.get("pre_market_home_win_prob"))
    if existing is not None:
        return _clamp_prob(existing)

    model_home = _safe_float(game.get("model_home_win_prob"))
    if model_home is None:
        model_home = _safe_float(game.get("home_win_prob"))
    if model_home is None:
        model_home = 0.5

    context_keys = [
        "starter_adjustment",
        "split_adjustment",
        "bullpen_adjustment",
        "bullpen_health_adjustment",
        "travel_adjustment",
        "lineup_adjustment",
        "lineup_health_adjustment",
        "luck_adjustment",
        "advanced_adjustment",
    ]
    context_adj = sum((_safe_float(game.get(key)) or 0.0) for key in context_keys)
    return _clamp_prob(model_home + context_adj)


def _derive_market_edge_pick(game: dict[str, Any]) -> float | None:
    home_team = str(game.get("home_team") or "")
    away_team = str(game.get("away_team") or "")
    if not home_team or not away_team:
        return None

    market_home = _safe_float(game.get("market_home_win_prob"))
    if market_home is None:
        return None

    home_prob = _safe_float(game.get("home_win_prob"))
    if home_prob is None:
        return None

    pre_market_home = _derive_pre_market_home_prob(game)
    pick_team = str(game.get("model_pick_team") or _pick_team_from_home_prob(home_prob, home_team=home_team, away_team=away_team))

    model_pick_prob = _pick_probability(pre_market_home, pick_team, home_team=home_team, away_team=away_team)
    market_pick_prob = _pick_probability(market_home, pick_team, home_team=home_team, away_team=away_team)
    return float(model_pick_prob - market_pick_prob)


def _matchup_id_from_game(game: dict[str, Any]) -> str:
    home_team = str(game.get("home_team") or "")
    away_team = str(game.get("away_team") or "")
    game_pk = game.get("game_pk")
    if game_pk is not None:
        return f"pk:{game_pk}"
    return f"alt:{game.get('official_date') or ''}|{away_team}|{home_team}"


def _bet_quality_grade(score: float) -> str:
    if score >= 75.0:
        return "A"
    if score >= 60.0:
        return "B"
    if score >= 45.0:
        return "C"
    return "Pass"


def _compute_bet_quality(
    *,
    pre_market_home: float,
    home_prob: float,
    market_edge_pick: float | None,
    market_edge_pick_raw: float | None,
    confidence: float,
    uncertainty_level: str,
    uncertainty_multiplier: float,
    band_half: float | None,
    market_available: bool,
) -> dict[str, Any]:
    if market_edge_pick_raw is not None:
        edge_basis = abs(float(market_edge_pick_raw))
    elif market_edge_pick is not None:
        edge_basis = abs(float(market_edge_pick))
    else:
        edge_basis = abs(float(pre_market_home) - 0.5) * 0.85

    edge_scale = 0.08 if market_available else 0.05
    edge_strength = max(0.0, min(1.0, edge_basis / edge_scale))
    confidence_clamped = max(0.0, min(1.0, float(confidence)))
    confidence_factor = 0.55 + (0.45 * confidence_clamped)

    if band_half is None:
        band_factor = 0.9
    else:
        band_ref = max(0.0, min(0.2, float(band_half)))
        band_factor = 1.0 - ((band_ref / 0.2) * 0.55)
    band_factor = max(0.35, min(1.0, band_factor))

    level = str(uncertainty_level or "low").strip().lower()
    if level == "high":
        level_factor = 0.58
    elif level == "medium":
        level_factor = 0.8
    else:
        level_factor = 1.0

    market_factor = 1.0 if market_available else 0.9
    unc_mult = max(0.35, min(1.0, float(uncertainty_multiplier)))

    score = 100.0 * edge_strength * confidence_factor * band_factor * level_factor * market_factor * unc_mult
    score = max(0.0, min(99.9, score))
    grade = _bet_quality_grade(score)

    edge_for_gate = abs(float(market_edge_pick)) if market_edge_pick is not None else abs(float(pre_market_home) - 0.5)
    actionable_cut = 60.0 if market_available else 42.0
    actionable = bool(score >= actionable_cut and level != "high" and edge_for_gate >= 0.02)

    return {
        "score": round(float(score), 1),
        "grade": grade,
        "actionable": actionable,
    }


_MATCHUP_DELTA_KEYS = [
    "home_win_prob",
    "pre_market_home_win_prob",
    "model_home_win_prob",
    "model_home_win_prob_raw",
    "starter_adjustment",
    "split_adjustment",
    "bullpen_adjustment",
    "bullpen_health_adjustment",
    "travel_adjustment",
    "lineup_adjustment",
    "lineup_health_adjustment",
    "luck_adjustment",
    "advanced_adjustment",
    "market_home_win_prob",
]


def _load_prior_matchup_lookup(
    reference_date: dt.date,
    archive_entries: list[dict[str, Any]],
    max_days: int = 21,
) -> dict[str, dict[str, Any]]:
    if not archive_entries:
        return {}

    lookup: dict[str, dict[str, Any]] = {}
    scanned_days = 0

    for entry in archive_entries:
        if scanned_days >= max(1, int(max_days)):
            break

        ref = _safe_iso_date(entry.get("reference_date"))
        if ref is None or ref >= reference_date:
            continue

        scanned_days += 1

        try:
            payload = read_prediction_archive(ref)
        except Exception:
            continue

        if not payload:
            continue

        for game in payload.get("games", []):
            if not isinstance(game, dict):
                continue

            matchup_id = _matchup_id_from_game(game)
            if matchup_id in lookup:
                continue

            lookup[matchup_id] = {
                "reference_date": ref,
                "game": game,
            }

    return lookup


def _attach_matchup_prediction_deltas(
    matchup_predictions: list[dict[str, Any]],
    reference_date: dt.date,
    archive_entries: list[dict[str, Any]],
) -> None:
    prior_lookup = _load_prior_matchup_lookup(reference_date, archive_entries=archive_entries, max_days=21)

    for game in matchup_predictions:
        matchup_id = _matchup_id_from_game(game)
        game["matchup_id"] = matchup_id

        prior = prior_lookup.get(matchup_id)
        if not prior:
            game["delta_source_date"] = None
            game["delta_days"] = None
            game["delta_favored_team_changed"] = None
            for key in _MATCHUP_DELTA_KEYS:
                game[f"delta_{key}"] = None
            continue

        prior_date = prior.get("reference_date")
        prior_game = prior.get("game", {})
        prior_date_iso = prior_date.isoformat() if isinstance(prior_date, dt.date) else None

        game["delta_source_date"] = prior_date_iso
        game["delta_days"] = int((reference_date - prior_date).days) if isinstance(prior_date, dt.date) else None

        prior_favored = str(prior_game.get("favored_team") or "")
        curr_favored = str(game.get("favored_team") or "")
        game["delta_favored_team_changed"] = bool(prior_favored and curr_favored and prior_favored != curr_favored)

        for key in _MATCHUP_DELTA_KEYS:
            current = _safe_float(game.get(key))
            previous = _safe_float(prior_game.get(key))
            delta_value = None if current is None or previous is None else float(current - previous)
            game[f"delta_{key}"] = round(float(delta_value), 4) if delta_value is not None else None


def _build_prediction_value_board(matchup_predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for game in matchup_predictions:
        home_team = str(game.get("home_team") or "")
        away_team = str(game.get("away_team") or "")
        if not home_team or not away_team:
            continue

        home_prob = _safe_float(game.get("home_win_prob"))
        away_prob = _safe_float(game.get("away_win_prob"))
        if home_prob is None:
            continue
        if away_prob is None:
            away_prob = 1.0 - home_prob

        pre_market_home = _derive_pre_market_home_prob(game)
        model_pick_team = str(game.get("model_pick_team") or _pick_team_from_home_prob(home_prob, home_team=home_team, away_team=away_team))
        model_pick_prob = _pick_probability(pre_market_home, model_pick_team, home_team=home_team, away_team=away_team)

        market_home = _safe_float(game.get("market_home_win_prob"))
        market_pick_prob = None
        market_favored_team = None
        is_market_upset = False
        if market_home is not None:
            market_pick_prob = _pick_probability(market_home, model_pick_team, home_team=home_team, away_team=away_team)
            market_favored_team = str(game.get("market_favored_team") or _pick_team_from_home_prob(market_home, home_team=home_team, away_team=away_team))
            is_market_upset = bool(market_favored_team and market_favored_team != model_pick_team)

        market_edge_pick_raw = _safe_float(game.get("model_vs_market_edge_pick"))
        if market_edge_pick_raw is None and market_pick_prob is not None:
            market_edge_pick_raw = float(model_pick_prob - market_pick_prob)

        uncertainty_multiplier = _safe_float(game.get("uncertainty_edge_multiplier"))
        if uncertainty_multiplier is None:
            uncertainty_multiplier = 1.0
        uncertainty_multiplier = float(max(0.35, min(1.0, uncertainty_multiplier)))

        market_edge_pick = None if market_edge_pick_raw is None else float(market_edge_pick_raw * uncertainty_multiplier)
        value_tier = _value_tier(market_edge_pick)

        confidence = round(abs(home_prob - 0.5) * 2.0, 4)
        uncertainty_level = str(game.get("uncertainty_level") or "low")
        band_half = _safe_float(game.get("home_win_prob_band_half"))
        bet_quality = _compute_bet_quality(
            pre_market_home=pre_market_home,
            home_prob=home_prob,
            market_edge_pick=market_edge_pick,
            market_edge_pick_raw=market_edge_pick_raw,
            confidence=confidence,
            uncertainty_level=uncertainty_level,
            uncertainty_multiplier=uncertainty_multiplier,
            band_half=band_half,
            market_available=market_pick_prob is not None,
        )

        matchup_id = _matchup_id_from_game(game)

        rows.append(
            {
                "matchup_id": matchup_id,
                "game_pk": game.get("game_pk"),
                "official_date": game.get("official_date"),
                "away_team": away_team,
                "home_team": home_team,
                "model_pick_team": model_pick_team,
                "model_pick_prob": round(float(model_pick_prob), 4),
                "market_pick_prob": round(float(market_pick_prob), 4) if market_pick_prob is not None else None,
                "market_home_win_prob": round(float(market_home), 4) if market_home is not None else None,
                "market_edge_pick": round(float(market_edge_pick), 4) if market_edge_pick is not None else None,
                "market_edge_pick_raw": round(float(market_edge_pick_raw), 4) if market_edge_pick_raw is not None else None,
                "uncertainty_edge_multiplier": round(float(uncertainty_multiplier), 3),
                "uncertainty_level": uncertainty_level,
                "high_uncertainty": bool(game.get("high_uncertainty")),
                "home_win_prob_band_low": _safe_float(game.get("home_win_prob_band_low")),
                "home_win_prob_band_high": _safe_float(game.get("home_win_prob_band_high")),
                "home_win_prob_band_half": band_half,
                "uncertainty_note": game.get("uncertainty_note"),
                "confidence": confidence,
                "value_tier": value_tier,
                "is_market_upset": is_market_upset,
                "market_favored_team": market_favored_team,
                "favored_team": game.get("favored_team"),
                "favored_win_prob": game.get("favored_win_prob"),
                "bet_quality_score": bet_quality.get("score"),
                "bet_quality_grade": bet_quality.get("grade"),
                "bet_quality_actionable": bool(bet_quality.get("actionable")),
                "delta_source_date": game.get("delta_source_date"),
                "delta_days": game.get("delta_days"),
                "delta_home_win_prob": game.get("delta_home_win_prob"),
                "delta_pre_market_home_win_prob": game.get("delta_pre_market_home_win_prob"),
                "delta_model_home_win_prob": game.get("delta_model_home_win_prob"),
                "delta_market_home_win_prob": game.get("delta_market_home_win_prob"),
                "delta_favored_team_changed": game.get("delta_favored_team_changed"),
            }
        )

    rows.sort(
        key=lambda row: (
            float(row.get("bet_quality_score") or 0.0),
            abs(float(row.get("market_edge_pick"))) if row.get("market_edge_pick") is not None else -1.0,
            float(row.get("confidence") or 0.0),
        ),
        reverse=True,
    )
    return rows


def _true_clv_pick_from_game(game: dict[str, Any], pick_team: str, home_team: str, away_team: str) -> float | None:
    open_home = _safe_float(game.get("market_home_open_prob"))
    close_home = _safe_float(game.get("market_home_close_prob"))
    if open_home is None or close_home is None:
        return None

    open_pick = _pick_probability(open_home, pick_team, home_team=home_team, away_team=away_team)
    close_pick = _pick_probability(close_home, pick_team, home_team=home_team, away_team=away_team)
    return float(close_pick - open_pick)


def _build_market_clv_tracker(archive_entries: list[dict[str, Any]], max_days: int = 30) -> dict[str, Any]:
    if not archive_entries:
        return {
            "mode": "proxy",
            "days_analyzed": 0,
            "games_with_market": 0,
            "finalized_with_market": 0,
            "avg_edge_pick": None,
            "median_edge_pick": None,
            "positive_edge_rate": None,
            "finalized_accuracy": None,
            "strong_edge_games": 0,
            "strong_edge_accuracy": None,
            "avg_true_clv": None,
            "daily": [],
        }

    proxy_edges: list[float] = []
    true_clvs: list[float] = []
    finalized_records: list[tuple[float, bool]] = []
    daily_rows: list[dict[str, Any]] = []

    for entry in archive_entries[: max(1, int(max_days))]:
        ref = _safe_iso_date(entry.get("reference_date"))
        if ref is None:
            continue

        try:
            payload = read_prediction_archive(ref)
        except Exception:
            continue

        if not payload:
            continue

        day_edges: list[float] = []
        day_true_clvs: list[float] = []
        day_finalized: list[bool] = []
        day_strong: list[bool] = []

        for game in payload.get("games", []):
            home_team = str(game.get("home_team") or "")
            away_team = str(game.get("away_team") or "")
            if not home_team or not away_team:
                continue

            home_prob = _safe_float(game.get("home_win_prob"))
            if home_prob is None:
                continue

            pick_team = str(game.get("model_pick_team") or _pick_team_from_home_prob(home_prob, home_team=home_team, away_team=away_team))

            edge_pick = _safe_float(game.get("model_vs_market_edge_pick"))
            if edge_pick is None:
                edge_pick = _derive_market_edge_pick(game)
            if edge_pick is None:
                continue

            day_edges.append(edge_pick)
            proxy_edges.append(edge_pick)

            true_clv = _true_clv_pick_from_game(game, pick_team=pick_team, home_team=home_team, away_team=away_team)
            if true_clv is not None:
                day_true_clvs.append(true_clv)
                true_clvs.append(true_clv)

            pred_correct = game.get("prediction_correct")
            if isinstance(pred_correct, bool):
                day_finalized.append(pred_correct)
                finalized_records.append((edge_pick, pred_correct))
                if abs(edge_pick) >= 0.05:
                    day_strong.append(pred_correct)

        if not day_edges:
            continue

        day_accuracy = float(sum(day_finalized) / len(day_finalized)) if day_finalized else None
        day_strong_accuracy = float(sum(day_strong) / len(day_strong)) if day_strong else None

        daily_rows.append(
            {
                "reference_date": ref.isoformat(),
                "games_with_market": len(day_edges),
                "avg_edge_pick": round(float(np.mean(day_edges)), 4),
                "finalized_games": len(day_finalized),
                "accuracy": round(day_accuracy, 4) if day_accuracy is not None else None,
                "strong_edge_games": len(day_strong),
                "strong_edge_accuracy": round(day_strong_accuracy, 4) if day_strong_accuracy is not None else None,
                "avg_true_clv": round(float(np.mean(day_true_clvs)), 4) if day_true_clvs else None,
            }
        )

    strong_final = [correct for edge, correct in finalized_records if abs(edge) >= 0.05]
    finalized_accuracy = float(sum(correct for _, correct in finalized_records) / len(finalized_records)) if finalized_records else None
    strong_accuracy = float(sum(strong_final) / len(strong_final)) if strong_final else None

    return {
        "mode": "true" if true_clvs else "proxy",
        "days_analyzed": len(daily_rows),
        "games_with_market": len(proxy_edges),
        "finalized_with_market": len(finalized_records),
        "avg_edge_pick": round(float(np.mean(proxy_edges)), 4) if proxy_edges else None,
        "median_edge_pick": round(float(np.median(proxy_edges)), 4) if proxy_edges else None,
        "positive_edge_rate": round(float(sum(1 for edge in proxy_edges if edge > 0) / len(proxy_edges)), 4) if proxy_edges else None,
        "finalized_accuracy": round(finalized_accuracy, 4) if finalized_accuracy is not None else None,
        "strong_edge_games": len(strong_final),
        "strong_edge_accuracy": round(strong_accuracy, 4) if strong_accuracy is not None else None,
        "avg_true_clv": round(float(np.mean(true_clvs)), 4) if true_clvs else None,
        "daily": daily_rows,
    }


def _build_team_dataframe(
    standings: dict[str, dict[str, Any]],
    mlb_api_by_team: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    all_teams = sorted(set(standings.keys()) | set(mlb_api_by_team.keys()))
    rows: list[dict[str, Any]] = []

    for team in all_teams:
        st = standings.get(team, {})
        api = mlb_api_by_team.get(team, {})

        wins = _to_native(st.get("wins")) or 0
        losses = _to_native(st.get("losses")) or 0
        games_played = wins + losses
        if games_played == 0:
            games_played = _to_native(api.get(GAMES_PLAYED_KEY)) or 0

        win_pct = st.get("win_pct")
        if win_pct is None and games_played > 0:
            win_pct = wins / games_played

        rows.append(
            {
                "Team": team,
                "wins": wins,
                "losses": losses,
                "games_played": games_played,
                "win_pct": win_pct if win_pct is not None else 0.0,
                "run_diff": _to_native(st.get("run_diff")) or 0,
                "streak": st.get("streak") or "-",
                "games_back": st.get("games_back") or "-",
                "ops": api.get(OPS_KEY),
                "hitting_ops_risp": api.get(HITTING_OPS_RISP_KEY),
                "hitting_ops_risp2": api.get(HITTING_OPS_RISP2_KEY),
                "hitting_ops_late_close": api.get(HITTING_OPS_LATE_CLOSE_KEY),
                "obp": api.get(OBP_KEY),
                "slg": api.get(SLG_KEY),
                "iso_adv": api.get(ISO_ADV_KEY),
                "hitting_walks_per_strikeout": api.get(HITTING_WALKS_PER_STRIKEOUT_KEY),
                "plate_appearances": api.get(PLATE_APPEARANCES_KEY),
                "stolen_base_pct": api.get(HITTING_STOLEN_BASE_PCT_KEY),
                "stolen_bases": api.get(HITTING_STOLEN_BASES_KEY),
                "caught_stealing": api.get(HITTING_CAUGHT_STEALING_KEY),
                "runs_scored": api.get(RUNS_SCORED_KEY),
                "runs_allowed": api.get(RUNS_ALLOWED_KEY),
                "pitching_ops_allowed": api.get(PITCHING_OPS_ALLOWED_KEY),
                "pitching_ops_risp_allowed": api.get(PITCHING_OPS_RISP_ALLOWED_KEY),
                "pitching_ops_risp2_allowed": api.get(PITCHING_OPS_RISP2_ALLOWED_KEY),
                "pitching_ops_late_close_allowed": api.get(PITCHING_OPS_LATE_CLOSE_ALLOWED_KEY),
                "era": api.get(ERA_KEY),
                "whip": api.get(WHIP_KEY),
                "pitching_kbb": api.get(PITCHING_KBB_KEY),
                "pitching_hr9": api.get(PITCHING_HR9_KEY),
                "pitching_whiff_pct": api.get(PITCHING_WHIFF_PCT_KEY),
                "games_started_pitching": api.get(PITCHING_GAMES_STARTED_KEY),
                "quality_starts": api.get(PITCHING_QUALITY_STARTS_KEY),
                "pitching_k_minus_bb_pct": api.get(PITCHING_K_MINUS_BB_PCT_KEY),
                "saves": api.get(SAVES_KEY),
                "blown_saves": api.get(BLOWN_SAVES_KEY),
                "save_opportunities": api.get(SAVE_OPPORTUNITIES_KEY),
                "holds": api.get(HOLDS_KEY),
                "fielding": api.get(FIELDING_KEY),
                "fielding_caught_stealing_pct": api.get(FIELDING_CAUGHT_STEALING_PCT_KEY),
                "hits_hitting": api.get(HITS_KEY),
                "doubles_hitting": api.get(DOUBLES_KEY),
                "triples_hitting": api.get(TRIPLES_KEY),
                "home_runs_hitting": api.get(HOME_RUNS_KEY),
                "at_bats": api.get(AT_BATS_KEY),
                "sac_flies_hitting": api.get(SAC_FLIES_KEY),
                "strikeouts_hitting": api.get(HITTING_STRIKEOUTS_KEY),
                "hitting_balls_in_play_adv": api.get(HITTING_BALLS_IN_PLAY_ADV_KEY),
                "hitting_line_hits_adv": api.get(HITTING_LINE_HITS_ADV_KEY),
                "hitting_fly_hits_adv": api.get(HITTING_FLY_HITS_ADV_KEY),
                "xwobacon_raw": _first_api_metric(api, XWOBA_CON_CANDIDATE_KEYS),
                "xfip_raw": _first_api_metric(api, XFIP_CANDIDATE_KEYS),
                "pitching_strikeouts_per9_adv": api.get(PITCHING_STRIKEOUTS_PER9_ADV_KEY),
                "pitching_bb_per9_adv": api.get(PITCHING_BB_PER9_ADV_KEY),
                "pitching_hr_per9_adv": api.get(PITCHING_HR_PER9_ADV_KEY),
                "pitching_fly_ball_pct_adv": api.get(PITCHING_FLY_BALL_PCT_ADV_KEY),
            }
        )

    df = pd.DataFrame(rows)
    numeric_cols = [
        "wins",
        "losses",
        "games_played",
        "win_pct",
        "run_diff",
        "ops",
        "hitting_ops_risp",
        "hitting_ops_risp2",
        "hitting_ops_late_close",
        "obp",
        "slg",
        "iso_adv",
        "hitting_walks_per_strikeout",
        "plate_appearances",
        "stolen_base_pct",
        "stolen_bases",
        "caught_stealing",
        "runs_scored",
        "runs_allowed",
        "pitching_ops_allowed",
        "pitching_ops_risp_allowed",
        "pitching_ops_risp2_allowed",
        "pitching_ops_late_close_allowed",
        "era",
        "whip",
        "pitching_kbb",
        "pitching_hr9",
        "pitching_whiff_pct",
        "games_started_pitching",
        "quality_starts",
        "pitching_k_minus_bb_pct",
        "saves",
        "blown_saves",
        "save_opportunities",
        "holds",
        "fielding",
        "fielding_caught_stealing_pct",
        "hits_hitting",
        "doubles_hitting",
        "triples_hitting",
        "home_runs_hitting",
        "at_bats",
        "sac_flies_hitting",
        "strikeouts_hitting",
        "hitting_balls_in_play_adv",
        "hitting_line_hits_adv",
        "hitting_fly_hits_adv",
        "xwobacon_raw",
        "xfip_raw",
        "pitching_strikeouts_per9_adv",
        "pitching_bb_per9_adv",
        "pitching_hr_per9_adv",
        "pitching_fly_ball_pct_adv",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["win_pct"] = df["win_pct"].fillna(0.0).clip(0.0, 1.0)
    df["run_diff"] = df["run_diff"].fillna(0.0)
    df["games_played"] = df["games_played"].replace(0, np.nan)
    df["run_diff_per_game"] = (df["run_diff"] / df["games_played"]).fillna(0.0)
    df["runs_scored_per_game"] = (df["runs_scored"] / df["games_played"]).fillna(df["run_diff_per_game"] + 4.5)
    df["runs_allowed_per_game"] = (df["runs_allowed"] / df["games_played"]).fillna(4.5)
    df["streak_score"] = df["streak"].apply(_streak_score)

    run_diff_norm = _normalize_series(df["run_diff_per_game"]).fillna(0.5)
    ops_norm = _normalize_series(df["ops"]).fillna(0.5)
    obp_norm = _normalize_series(df["obp"]).fillna(0.5)
    slg_norm = _normalize_series(df["slg"]).fillna(0.5)
    iso_norm = _normalize_series(df["iso_adv"]).fillna(0.5)
    discipline_norm = _normalize_series(df["hitting_walks_per_strikeout"]).fillna(0.5)
    era_inv_norm = (1 - _normalize_series(df["era"])).fillna(0.5)
    whip_inv_norm = (1 - _normalize_series(df["whip"])).fillna(0.5)
    fielding_norm = _normalize_series(df["fielding"]).fillna(0.5)
    catcher_control_norm = _normalize_series(df["fielding_caught_stealing_pct"]).fillna(0.5)

    plate_appearances_per_game = (df["plate_appearances"] / df["games_played"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    plate_appearances_z = (plate_appearances_per_game - plate_appearances_per_game.mean()) / plate_appearances_per_game.std(ddof=0)
    plate_appearances_z = plate_appearances_z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    plate_appearances_norm = _normalize_series(plate_appearances_z).fillna(0.5)

    runs_scored_pg_z = (df["runs_scored_per_game"] - df["runs_scored_per_game"].mean()) / df["runs_scored_per_game"].std(ddof=0)
    runs_scored_pg_z = runs_scored_pg_z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    runs_scored_pg_norm = _normalize_series(runs_scored_pg_z).fillna(0.5)

    runs_allowed_pg_z = (df["runs_allowed_per_game"] - df["runs_allowed_per_game"].mean()) / df["runs_allowed_per_game"].std(ddof=0)
    runs_allowed_pg_z = runs_allowed_pg_z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    runs_allowed_pg_norm = _normalize_series(runs_allowed_pg_z).fillna(0.5)
    runs_allowed_inv_norm = (1.0 - runs_allowed_pg_norm).fillna(0.5)

    hitting_ops_base = df["ops"].fillna(df["ops"].median()).fillna(0.720).clip(0.550, 1.020)
    hitting_ops_risp = df["hitting_ops_risp"].fillna(hitting_ops_base).clip(0.450, 1.200)
    hitting_ops_risp2 = df["hitting_ops_risp2"].fillna(hitting_ops_risp).clip(0.420, 1.220)
    hitting_ops_late_close = df["hitting_ops_late_close"].fillna(hitting_ops_base).clip(0.420, 1.200)

    pitching_ops_base = df["pitching_ops_allowed"].fillna(df["pitching_ops_allowed"].median()).fillna(0.720).clip(0.480, 1.080)
    pitching_ops_risp_allowed = df["pitching_ops_risp_allowed"].fillna(pitching_ops_base).clip(0.420, 1.220)
    pitching_ops_risp2_allowed = df["pitching_ops_risp2_allowed"].fillna(pitching_ops_risp_allowed).clip(0.420, 1.220)
    pitching_ops_late_close_allowed = df["pitching_ops_late_close_allowed"].fillna(pitching_ops_base).clip(0.420, 1.220)

    leverage_sample_conf = (df["games_played"].fillna(0.0) / 95.0).clip(0.25, 1.0)

    leverage_offense_raw = (
        0.50 * _normalize_series(hitting_ops_late_close)
        + 0.30 * _normalize_series(hitting_ops_risp2)
        + 0.20 * _normalize_series(hitting_ops_risp)
    ).fillna(0.5).clip(0.0, 1.0)
    leverage_pitching_raw = (
        0.50 * (1.0 - _normalize_series(pitching_ops_late_close_allowed))
        + 0.30 * (1.0 - _normalize_series(pitching_ops_risp2_allowed))
        + 0.20 * (1.0 - _normalize_series(pitching_ops_risp_allowed))
    ).fillna(0.5).clip(0.0, 1.0)

    leverage_offense_quality = (leverage_sample_conf * leverage_offense_raw + (1.0 - leverage_sample_conf) * 0.5).clip(0.0, 1.0)
    leverage_pitching_quality = (leverage_sample_conf * leverage_pitching_raw + (1.0 - leverage_sample_conf) * 0.5).clip(0.0, 1.0)
    leverage_net_quality = (0.52 * leverage_offense_quality + 0.48 * leverage_pitching_quality).clip(0.0, 1.0)

    clutch_offense_delta_ops = (
        0.55 * (hitting_ops_late_close - hitting_ops_base)
        + 0.45 * (hitting_ops_risp2 - hitting_ops_base)
    ).clip(-0.250, 0.250)
    clutch_pitching_delta_ops = (
        pitching_ops_base
        - (0.55 * pitching_ops_late_close_allowed + 0.45 * pitching_ops_risp2_allowed)
    ).clip(-0.250, 0.250)

    clutch_offense_delta_ops = (clutch_offense_delta_ops * leverage_sample_conf).clip(-0.250, 0.250)
    clutch_pitching_delta_ops = (clutch_pitching_delta_ops * leverage_sample_conf).clip(-0.250, 0.250)
    clutch_net_delta_ops = (0.55 * clutch_offense_delta_ops + 0.45 * clutch_pitching_delta_ops).clip(-0.250, 0.250)
    clutch_index = (0.5 + clutch_net_delta_ops / 0.30).clip(0.0, 1.0)

    stolen_base_pct = df["stolen_base_pct"].fillna(df["stolen_base_pct"].median())
    sb_attempts_per_game = (
        (df["stolen_bases"].fillna(0.0) + df["caught_stealing"].fillna(0.0))
        / df["games_played"]
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    hits_hitting = df["hits_hitting"].fillna(0.0)
    doubles_hitting = df["doubles_hitting"].fillna(0.0)
    triples_hitting = df["triples_hitting"].fillna(0.0)
    home_runs_hitting = df["home_runs_hitting"].fillna(0.0)
    at_bats = df["at_bats"].fillna(0.0)
    strikeouts_hitting = df["strikeouts_hitting"].fillna(0.0)
    balls_in_play = df["hitting_balls_in_play_adv"].replace(0, np.nan)
    fallback_bip = (at_bats - strikeouts_hitting - home_runs_hitting).clip(lower=1.0)
    contact_events = balls_in_play.fillna(fallback_bip).replace(0, np.nan)

    singles_hitting = (hits_hitting - doubles_hitting - triples_hitting - home_runs_hitting).clip(lower=0.0)
    contact_woba = (
        0.89 * singles_hitting
        + 1.27 * doubles_hitting
        + 1.62 * triples_hitting
        + 2.10 * home_runs_hitting
    ) / contact_events
    contact_woba = contact_woba.replace([np.inf, -np.inf], np.nan)

    line_hits_rate = (df["hitting_line_hits_adv"].fillna(0.0) / contact_events).replace([np.inf, -np.inf], np.nan)
    fly_hits_rate = (df["hitting_fly_hits_adv"].fillna(0.0) / contact_events).replace([np.inf, -np.inf], np.nan)

    iso_raw = df["iso_adv"].fillna(0.16).clip(lower=0.05, upper=0.35)
    xwobacon_raw = pd.to_numeric(df["xwobacon_raw"], errors="coerce")
    xwobacon_raw = xwobacon_raw.where(xwobacon_raw <= 1.5, xwobacon_raw / 100.0)
    xwobacon_raw = xwobacon_raw.where(xwobacon_raw <= 1.0, xwobacon_raw / 1000.0)

    xwobacon_formula = (
        0.72 * contact_woba.fillna(contact_woba.median()).fillna(0.330)
        + 0.16 * line_hits_rate.fillna(line_hits_rate.median()).fillna(0.200)
        + 0.07 * fly_hits_rate.fillna(fly_hits_rate.median()).fillna(0.130)
        + 0.05 * iso_raw.fillna(0.16)
    )
    df["xwobacon_proxy"] = xwobacon_raw.fillna(xwobacon_formula).clip(0.220, 0.460)
    xwobacon_norm = _normalize_series(df["xwobacon_proxy"]).fillna(0.5)

    k9_adv = df["pitching_strikeouts_per9_adv"].fillna(df["pitching_k_minus_bb_pct"] * 0.30 + 8.4)
    bb9_adv = df["pitching_bb_per9_adv"].fillna(3.1)
    hr9_adv = df["pitching_hr_per9_adv"].fillna(df["pitching_hr9"]).fillna(1.15)
    fly_ball_pct = _as_ratio(df["pitching_fly_ball_pct_adv"]).fillna(0.35).clip(0.20, 0.55)

    league_fly_ball = float(fly_ball_pct.median()) if not fly_ball_pct.dropna().empty else 0.35
    expected_hr9 = (0.60 * hr9_adv + 0.40 * 1.15 + 0.55 * (fly_ball_pct - league_fly_ball)).clip(0.55, 2.20)

    xfip_raw = pd.to_numeric(df["xfip_raw"], errors="coerce")
    xfip_raw = xfip_raw.where(xfip_raw <= 12, xfip_raw / 10.0)
    xfip_formula = (3.20 + 0.28 * (bb9_adv - 3.1) - 0.17 * (k9_adv - 8.6) + 0.30 * (expected_hr9 - 1.1)).clip(2.30, 6.40)
    df["xfip_proxy"] = xfip_raw.fillna(xfip_formula).clip(2.20, 6.80)
    xfip_inv_norm = (1 - _normalize_series(df["xfip_proxy"])).fillna(0.5)

    # Joe Score v3: fixed 45/40/15 blend (xwOBAcon/xFIP-inverse/clutch) with shrinkage and uncertainty bands.
    joe_clutch_signal = (0.65 * leverage_net_quality + 0.35 * clutch_index).clip(0.0, 1.0)

    joe_weight_xwobacon = 0.45
    joe_weight_xfip = 0.40
    joe_weight_clutch = 0.15

    joe_raw_norm = (
        joe_weight_xwobacon * xwobacon_norm
        + joe_weight_xfip * xfip_inv_norm
        + joe_weight_clutch * joe_clutch_signal
    ).clip(0.0, 1.0)
    joe_sample_conf = (df["games_played"].fillna(0.0) / 100.0).clip(0.20, 1.0)
    joe_signal_disagreement = (
        0.55 * (xwobacon_norm - xfip_inv_norm).abs()
        + 0.25 * (xwobacon_norm - joe_clutch_signal).abs()
        + 0.20 * (xfip_inv_norm - joe_clutch_signal).abs()
    ).clip(0.0, 1.0)
    joe_agreement_conf = (1.0 - joe_signal_disagreement).clip(0.0, 1.0)
    joe_reliability = (0.76 * joe_sample_conf + 0.24 * joe_agreement_conf).clip(0.18, 1.0)

    joe_score_norm = (joe_reliability * joe_raw_norm + (1.0 - joe_reliability) * 0.5).clip(0.0, 1.0)
    joe_uncertainty_points = ((1.0 - joe_reliability) * 20.0 + joe_signal_disagreement * 8.0).clip(2.0, 20.0)

    offense_quality = (
        0.22 * ops_norm
        + 0.16 * obp_norm
        + 0.16 * slg_norm
        + 0.12 * iso_norm
        + 0.08 * discipline_norm
        + 0.16 * xwobacon_norm
        + 0.10 * leverage_offense_quality
    ).clip(0.0, 1.0)

    kbb_norm = _normalize_series(df["pitching_kbb"]).fillna(0.5)
    hr9_inv_norm = (1 - _normalize_series(df["pitching_hr9"])).fillna(0.5)
    whiff_norm = _normalize_series(df["pitching_whiff_pct"]).fillna(0.5)
    k_minus_bb_pct_norm = _normalize_series(df["pitching_k_minus_bb_pct"]).fillna(0.5)

    run_prevention_quality = (
        0.20 * era_inv_norm
        + 0.16 * whip_inv_norm
        + 0.14 * kbb_norm
        + 0.11 * hr9_inv_norm
        + 0.08 * whiff_norm
        + 0.08 * k_minus_bb_pct_norm
        + 0.14 * xfip_inv_norm
        + 0.09 * leverage_pitching_quality
    ).clip(0.0, 1.0)

    save_opp = df["save_opportunities"].fillna(0.0)
    saves = df["saves"].fillna(0.0)
    blown = df["blown_saves"].fillna(0.0)
    holds = df["holds"].fillna(0.0)

    bullpen_conversion = (saves + 0.5 * holds + 5.0) / (save_opp + holds + blown + 10.0)
    blown_rate = blown / (save_opp + blown + 8.0)

    bullpen_quality = (
        0.60 * _normalize_series(bullpen_conversion)
        + 0.25 * (1 - _normalize_series(blown_rate))
        + 0.15 * run_prevention_quality
    ).fillna(0.5).clip(0.0, 1.0)

    rf = df["runs_scored"].fillna(0.0) + 1.0
    ra = df["runs_allowed"].fillna(0.0) + 1.0
    exponent = 1.83
    pythag = (rf**exponent) / ((rf**exponent) + (ra**exponent))
    df["pythag_win_pct"] = pythag.fillna(0.5).clip(0.0, 1.0)

    trend_norm = _normalize_series(df["streak_score"]).fillna(0.5)
    scoring_form_norm = _normalize_series(df["runs_scored_per_game"] - df["runs_allowed_per_game"]).fillna(0.5)
    regression_signal = _normalize_series(df["pythag_win_pct"] - df["win_pct"]).fillna(0.5)

    quality_starts_rate = (
        df["quality_starts"]
        / df["games_started_pitching"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)

    starter_share = (0.52 + 0.26 * quality_starts_rate).clip(0.45, 0.80)
    bullpen_share = (1.0 - starter_share).clip(0.20, 0.55)

    blown_saves_norm = _normalize_series(df["blown_saves"].fillna(0.0)).fillna(0.5)

    # Friend-style base term proxies (series / sweep behavior), bounded in ~[0, 2].
    series_strength = _normalize_series(df["win_pct"]).fillna(0.5)
    sweep_resilience = (0.65 * runs_allowed_inv_norm + 0.35 * (1.0 - blown_saves_norm)).clip(0.0, 1.0)
    base_bucket = (series_strength + sweep_resilience).clip(0.0, 2.0)

    # Friend-style offense bucket proxies (wRC+/BsR/PA/Runs style).
    ops_point = _signed_scale(df["ops"] - df["ops"].median()).fillna(0.0)
    baserunning_raw = 0.65 * (stolen_base_pct - stolen_base_pct.median()) + 0.35 * (sb_attempts_per_game - sb_attempts_per_game.median())
    baserunning_point = _signed_scale(baserunning_raw).fillna(0.0)
    offense_bucket = 0.485 * (ops_point + baserunning_point + plate_appearances_norm + runs_scored_pg_norm)

    # Friend-style defense bucket proxies with starter/bullpen weighting and defense context.
    starter_quality = (
        0.45 * run_prevention_quality
        + 0.35 * k_minus_bb_pct_norm
        + 0.20 * _normalize_series(quality_starts_rate).fillna(0.5)
    ).clip(0.0, 1.0)
    bullpen_style = (
        0.58 * bullpen_quality
        + 0.22 * (1.0 - blown_saves_norm)
        + 0.20 * k_minus_bb_pct_norm
    ).clip(0.0, 1.0)
    zone_proxy_norm = _normalize_series(df["pitching_whiff_pct"]).fillna(0.5)
    defense_bucket = 0.485 * (
        (starter_quality * starter_share)
        + (bullpen_style * bullpen_share)
        + fielding_norm
        + catcher_control_norm
        + zone_proxy_norm
        + runs_allowed_inv_norm
    )

    # Small context bucket (pythag/regression + starter/bullpen differential + expected quality gap).
    pythag_point = _signed_scale(df["pythag_win_pct"] - df["win_pct"]).fillna(0.0)
    starter_diff_point = _signed_scale(starter_quality - run_prevention_quality).fillna(0.0)
    bullpen_diff_point = _signed_scale(bullpen_style - run_prevention_quality).fillna(0.0)
    leverage_point = _signed_scale(leverage_net_quality - leverage_net_quality.median()).fillna(0.0)
    clutch_point = _signed_scale(clutch_net_delta_ops).fillna(0.0)
    expected_quality_gap = _signed_scale(
        (df["xwobacon_proxy"] - df["xwobacon_proxy"].median())
        - 0.35 * (df["xfip_proxy"] - df["xfip_proxy"].median())
    ).fillna(0.0)
    context_bucket = (
        0.025 * (pythag_point + starter_diff_point + bullpen_diff_point)
        + 0.015 * expected_quality_gap
        + 0.010 * leverage_point
        + 0.008 * clutch_point
    )

    # Improvement: confidence shrinkage early in season to reduce noise from small samples.
    mscore_confidence = (df["games_played"].fillna(0.0) / 80.0).clip(0.35, 1.0)
    base_bucket_adj = base_bucket * (0.80 + 0.20 * mscore_confidence)
    offense_bucket_adj = offense_bucket * (0.60 + 0.40 * mscore_confidence)
    defense_bucket_adj = defense_bucket * (0.60 + 0.40 * mscore_confidence)
    context_bucket_adj = context_bucket * mscore_confidence

    mscore_raw = base_bucket_adj + offense_bucket_adj + defense_bucket_adj + context_bucket_adj

    # Absolute (non-minmax) score mapping so day-to-day league extremes do not pin top teams at 100.
    mscore_center = 2.90
    mscore_scale = 0.55
    mscore_norm = 1.0 / (1.0 + np.exp(-((mscore_raw - mscore_center) / max(1e-6, mscore_scale))))
    mscore_norm = pd.Series(mscore_norm, index=df.index).clip(0.001, 0.999)

    df["mscore"] = mscore_norm * 100.0
    df["mscore_base_component"] = base_bucket_adj
    df["mscore_offense_component"] = offense_bucket_adj
    df["mscore_defense_component"] = defense_bucket_adj
    df["mscore_context_component"] = context_bucket_adj

    talent_norm = (
        0.50 * mscore_norm
        + 0.20 * offense_quality
        + 0.20 * run_prevention_quality
        + 0.10 * fielding_norm
    ).clip(0.0, 1.0)

    form_norm = (
        0.42 * trend_norm
        + 0.23 * scoring_form_norm
        + 0.20 * run_diff_norm
        + 0.15 * regression_signal
    ).clip(0.0, 1.0)

    games_confidence = (df["games_played"].fillna(0.0) / 75.0).clip(0.35, 1.0)
    elite_strength = ((mscore_norm - 0.82) / 0.18).clip(0.0, 1.0)

    overperf_risk = _normalize_series(df["win_pct"] - df["pythag_win_pct"]).fillna(0.5)
    sample_risk = (1.0 - games_confidence).clip(0.0, 1.0)
    run_prevention_risk = (1.0 - run_prevention_quality).clip(0.0, 1.0)
    bullpen_risk = (1.0 - bullpen_quality).clip(0.0, 1.0)

    risk_norm = (
        0.35 * bullpen_risk
        + 0.30 * overperf_risk
        + 0.20 * sample_risk
        + 0.15 * run_prevention_risk
    ).clip(0.0, 1.0)

    context_raw_core = (
        0.52 * (talent_norm - mscore_norm)
        + 0.38 * (form_norm - 0.5)
        - 0.55 * (risk_norm - 0.5)
    )

    context_scaled = pd.Series(context_raw_core * games_confidence, index=df.index)
    negative_factor = (1.0 - 0.50 * elite_strength).clip(0.55, 1.0)
    context_scaled = pd.Series(np.where(context_scaled < 0, context_scaled * negative_factor, context_scaled), index=df.index)

    context_cap = (0.015 + 0.08 * games_confidence).clip(0.04, 0.095)
    core_context_adjustment = pd.Series(np.clip(context_scaled, -context_cap, context_cap), index=df.index)

    # Coaching/execution proxy: uses residual win conversion, closeout quality, and tactical execution proxies.
    pythag_execution_edge = (df["win_pct"] - df["pythag_win_pct"]).clip(-0.18, 0.18)
    bullpen_closeout_expected = (0.50 + 0.40 * bullpen_quality).clip(0.25, 0.90)
    bullpen_closeout_edge = (bullpen_conversion - bullpen_closeout_expected).clip(-0.25, 0.25)
    baserunning_execution_edge = (stolen_base_pct - stolen_base_pct.median()).fillna(0.0) * np.sqrt(sb_attempts_per_game.clip(lower=0.0, upper=4.0) + 0.5)
    discipline_execution_edge = (
        df["hitting_walks_per_strikeout"].fillna(df["hitting_walks_per_strikeout"].median())
        - df["hitting_walks_per_strikeout"].median()
    ).fillna(0.0)
    defense_execution_edge = (
        0.70 * (fielding_norm - fielding_norm.median())
        + 0.30 * (catcher_control_norm - catcher_control_norm.median())
    ).fillna(0.0)

    coaching_proxy_raw = (
        0.42 * _signed_scale(pythag_execution_edge).fillna(0.0)
        + 0.23 * _signed_scale(bullpen_closeout_edge).fillna(0.0)
        + 0.15 * _signed_scale(baserunning_execution_edge).fillna(0.0)
        + 0.10 * _signed_scale(discipline_execution_edge).fillna(0.0)
        + 0.10 * _signed_scale(defense_execution_edge).fillna(0.0)
    ).clip(-1.0, 1.0)

    coaching_sample_conf = (df["games_played"].fillna(0.0) / 95.0).clip(0.20, 1.0)
    coaching_dispersion = float(pd.to_numeric(coaching_proxy_raw, errors="coerce").fillna(0.0).std(ddof=0))
    coaching_abs_mean = float(pd.to_numeric(coaching_proxy_raw, errors="coerce").fillna(0.0).abs().mean())

    # Meaningfulness gate: if league-wide signal is weak, coaching overlay naturally shuts off.
    coaching_strength_gate = float(np.clip((coaching_dispersion - 0.12) / 0.22, 0.0, 1.0))
    coaching_reliability_gate = float(np.clip((coaching_abs_mean - 0.10) / 0.25, 0.0, 1.0))
    coaching_meaningful_gate = coaching_strength_gate * coaching_reliability_gate

    coaching_team_signal = np.clip(np.abs(coaching_proxy_raw) / 0.55, 0.0, 1.0)
    coaching_confidence = (coaching_sample_conf * (0.45 + 0.55 * coaching_team_signal) * coaching_meaningful_gate).clip(0.0, 1.0)
    coaching_adjustment = (0.06 * coaching_proxy_raw * coaching_confidence).clip(-0.03, 0.03)

    coaching_weight_target = 0.05
    coaching_weight_effective = float(coaching_weight_target * coaching_meaningful_gate)

    context_adjustment = (
        ((1.0 - coaching_weight_effective) * core_context_adjustment)
        + (coaching_weight_effective * coaching_adjustment)
    ).clip(-0.11, 0.11)

    # Apply context on logit scale so elite teams can still move up/down without clipping artifacts.
    mscore_logit = np.log(mscore_norm / (1.0 - mscore_norm))
    context_logit_weight = 6.0
    power_logit = mscore_logit + context_adjustment * context_logit_weight
    power_norm = 1.0 / (1.0 + np.exp(-power_logit))
    power_norm = pd.Series(power_norm, index=df.index).clip(0.001, 0.999)

    df["live_power_score"] = power_norm * 100.0
    # D+ Score: near-equal bucket blend with higher pitching quality/health emphasis than offense.
    pitching_quality_bucket = (
        0.42 * starter_quality
        + 0.30 * bullpen_quality
        + 0.16 * k_minus_bb_pct_norm
        + 0.12 * (1.0 - blown_saves_norm)
    ).clip(0.0, 1.0)
    pitching_health_bucket = (
        0.55 * (1.0 - blown_saves_norm)
        + 0.25 * bullpen_quality
        + 0.20 * _normalize_series(quality_starts_rate).fillna(0.5)
    ).clip(0.0, 1.0)
    defense_pitch_bucket = (
        0.58 * defense_bucket_adj
        + 0.42 * (1.8 * (0.65 * pitching_quality_bucket + 0.35 * pitching_health_bucket))
    )
    context_bucket_rescaled = (1.0 + 14.0 * context_bucket_adj).clip(0.2, 1.8)

    dplus_raw = (
        0.24 * base_bucket_adj
        + 0.21 * offense_bucket_adj
        + 0.31 * defense_pitch_bucket
        + 0.24 * context_bucket_rescaled
    )
    dplus_center = 1.00
    dplus_scale = 0.16
    dplus_norm = 1.0 / (1.0 + np.exp(-((dplus_raw - dplus_center) / max(1e-6, dplus_scale))))
    dplus_norm = pd.Series(dplus_norm, index=df.index).clip(0.001, 0.999)
    df["d_plus_score"] = dplus_norm * 100.0
    df["joe_score"] = joe_score_norm * 100.0
    df["joe_confidence"] = joe_reliability * 100.0
    df["joe_band_low"] = (df["joe_score"] - joe_uncertainty_points).clip(0.0, 100.0)
    df["joe_band_high"] = (df["joe_score"] + joe_uncertainty_points).clip(0.0, 100.0)
    df["joe_uncertainty_points"] = joe_uncertainty_points
    df["joe_uncertainty_level"] = np.where(
        joe_uncertainty_points >= 12.0,
        "high",
        np.where(joe_uncertainty_points >= 7.0, "medium", "low"),
    )
    df["joe_weight_xwobacon_pct"] = joe_weight_xwobacon * 100.0
    df["joe_weight_xfip_pct"] = joe_weight_xfip * 100.0
    df["joe_weight_clutch_pct"] = joe_weight_clutch * 100.0
    df["hitting_ops_risp"] = hitting_ops_risp
    df["hitting_ops_risp2"] = hitting_ops_risp2
    df["hitting_ops_late_close"] = hitting_ops_late_close
    df["pitching_ops_allowed"] = pitching_ops_base
    df["pitching_ops_risp_allowed"] = pitching_ops_risp_allowed
    df["pitching_ops_risp2_allowed"] = pitching_ops_risp2_allowed
    df["pitching_ops_late_close_allowed"] = pitching_ops_late_close_allowed
    df["leverage_offense_quality"] = leverage_offense_quality * 100.0
    df["leverage_pitching_quality"] = leverage_pitching_quality * 100.0
    df["leverage_net_quality"] = leverage_net_quality * 100.0
    df["clutch_index"] = clutch_index * 100.0
    df["clutch_offense_delta_ops"] = clutch_offense_delta_ops
    df["clutch_pitching_delta_ops"] = clutch_pitching_delta_ops
    df["power_minus_d_plus"] = (power_norm - dplus_norm) * 100.0

    # Keep legacy gap for compatibility.
    df["power_minus_mscore"] = (power_norm - mscore_norm) * 100.0

    df["talent_score"] = talent_norm * 100.0
    df["form_score"] = form_norm * 100.0
    df["risk_score"] = risk_norm * 100.0
    df["power_context_adjustment"] = context_adjustment * 100.0
    df["power_context_confidence"] = games_confidence * 100.0
    df["coaching_execution_score"] = (0.5 + 0.5 * coaching_proxy_raw).clip(0.0, 1.0) * 100.0
    df["coaching_execution_adjustment"] = coaching_adjustment * 100.0
    df["coaching_execution_confidence"] = coaching_confidence * 100.0
    df["coaching_execution_weight"] = coaching_weight_effective * 100.0

    df["live_power_score"] = df["live_power_score"].clip(0.0, 100.0)
    df["d_plus_score"] = df["d_plus_score"].clip(0.0, 100.0)
    df["joe_score"] = df["joe_score"].clip(0.0, 100.0)
    df["joe_band_low"] = df["joe_band_low"].clip(0.0, 100.0)
    df["joe_band_high"] = df["joe_band_high"].clip(0.0, 100.0)
    df["projected_win_pct"] = (0.52 * df["win_pct"] + 0.38 * (df["live_power_score"] / 100.0) + 0.10 * form_norm).clip(0, 1)
    df["projected_wins_162"] = df["projected_win_pct"] * 162


    df = df.sort_values(["live_power_score", "win_pct"], ascending=False).reset_index(drop=True)
    df["live_rank"] = df.index + 1
    df["joe_rank"] = pd.to_numeric(df["joe_score"], errors="coerce").rank(ascending=False, method="min")
    df["workbook_rank"] = np.nan
    df["rank_delta"] = np.nan

    return df



def _teams_to_records(df: pd.DataFrame, mlb_api_by_team: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        team_name = row["Team"]
        mlb_api_stats = mlb_api_by_team.get(team_name, {})

        live_stats = {
            "live_rank": _to_native(row.get("live_rank")),
            "workbook_rank": None,
            "rank_delta": None,
            "mscore": _to_native(row.get("mscore"), 3),
            "mscore_base_component": _to_native(row.get("mscore_base_component"), 4),
            "mscore_offense_component": _to_native(row.get("mscore_offense_component"), 4),
            "mscore_defense_component": _to_native(row.get("mscore_defense_component"), 4),
            "mscore_context_component": _to_native(row.get("mscore_context_component"), 4),
            "xwobacon_proxy": _to_native(row.get("xwobacon_proxy"), 4),
            "xfip_proxy": _to_native(row.get("xfip_proxy"), 3),
            "hitting_ops_risp": _to_native(row.get("hitting_ops_risp"), 3),
            "hitting_ops_risp2": _to_native(row.get("hitting_ops_risp2"), 3),
            "hitting_ops_late_close": _to_native(row.get("hitting_ops_late_close"), 3),
            "pitching_ops_allowed": _to_native(row.get("pitching_ops_allowed"), 3),
            "pitching_ops_risp_allowed": _to_native(row.get("pitching_ops_risp_allowed"), 3),
            "pitching_ops_risp2_allowed": _to_native(row.get("pitching_ops_risp2_allowed"), 3),
            "pitching_ops_late_close_allowed": _to_native(row.get("pitching_ops_late_close_allowed"), 3),
            "leverage_offense_quality": _to_native(row.get("leverage_offense_quality"), 2),
            "leverage_pitching_quality": _to_native(row.get("leverage_pitching_quality"), 2),
            "leverage_net_quality": _to_native(row.get("leverage_net_quality"), 2),
            "clutch_index": _to_native(row.get("clutch_index"), 2),
            "clutch_offense_delta_ops": _to_native(row.get("clutch_offense_delta_ops"), 3),
            "clutch_pitching_delta_ops": _to_native(row.get("clutch_pitching_delta_ops"), 3),
            "joe_score": _to_native(row.get("joe_score"), 2),
            "joe_rank": _to_native(row.get("joe_rank")),
            "joe_confidence": _to_native(row.get("joe_confidence"), 1),
            "joe_band_low": _to_native(row.get("joe_band_low"), 1),
            "joe_band_high": _to_native(row.get("joe_band_high"), 1),
            "joe_uncertainty_points": _to_native(row.get("joe_uncertainty_points"), 1),
            "joe_uncertainty_level": row.get("joe_uncertainty_level") or "medium",
            "joe_weight_xwobacon_pct": _to_native(row.get("joe_weight_xwobacon_pct"), 1),
            "joe_weight_xfip_pct": _to_native(row.get("joe_weight_xfip_pct"), 1),
            "joe_weight_clutch_pct": _to_native(row.get("joe_weight_clutch_pct"), 1),
            "live_power_score": _to_native(row.get("live_power_score"), 2),
            "d_plus_score": _to_native(row.get("d_plus_score"), 2),
            "power_minus_d_plus": _to_native(row.get("power_minus_d_plus"), 3),
            "power_minus_mscore": _to_native(row.get("power_minus_mscore"), 3),
            "talent_score": _to_native(row.get("talent_score"), 2),
            "form_score": _to_native(row.get("form_score"), 2),
            "risk_score": _to_native(row.get("risk_score"), 2),
            "power_context_adjustment": _to_native(row.get("power_context_adjustment"), 3),
            "power_context_confidence": _to_native(row.get("power_context_confidence"), 1),
            "coaching_execution_score": _to_native(row.get("coaching_execution_score"), 2),
            "coaching_execution_adjustment": _to_native(row.get("coaching_execution_adjustment"), 3),
            "coaching_execution_confidence": _to_native(row.get("coaching_execution_confidence"), 1),
            "coaching_execution_weight": _to_native(row.get("coaching_execution_weight"), 2),
            "wins": _to_native(row.get("wins")),
            "losses": _to_native(row.get("losses")),
            "win_pct": _to_native(row.get("win_pct"), 3),
            "run_diff": _to_native(row.get("run_diff")),
            "streak": row.get("streak") or "-",
            "games_back": row.get("games_back") or "-",
            "projected_win_pct": _to_native(row.get("projected_win_pct"), 3),
            "projected_wins_162": _to_native(row.get("projected_wins_162"), 1),
        }

        records.append(
            {
                "team": team_name,
                "workbook_rank": None,
                "live_rank": _to_native(row.get("live_rank")),
                "rank_delta": None,
                "mscore": _to_native(row.get("mscore"), 3),
                "mscore_base_component": _to_native(row.get("mscore_base_component"), 4),
                "mscore_offense_component": _to_native(row.get("mscore_offense_component"), 4),
                "mscore_defense_component": _to_native(row.get("mscore_defense_component"), 4),
                "mscore_context_component": _to_native(row.get("mscore_context_component"), 4),
                "xwobacon_proxy": _to_native(row.get("xwobacon_proxy"), 4),
                "xfip_proxy": _to_native(row.get("xfip_proxy"), 3),
                "hitting_ops_risp": _to_native(row.get("hitting_ops_risp"), 3),
                "hitting_ops_risp2": _to_native(row.get("hitting_ops_risp2"), 3),
                "hitting_ops_late_close": _to_native(row.get("hitting_ops_late_close"), 3),
                "pitching_ops_allowed": _to_native(row.get("pitching_ops_allowed"), 3),
                "pitching_ops_risp_allowed": _to_native(row.get("pitching_ops_risp_allowed"), 3),
                "pitching_ops_risp2_allowed": _to_native(row.get("pitching_ops_risp2_allowed"), 3),
                "pitching_ops_late_close_allowed": _to_native(row.get("pitching_ops_late_close_allowed"), 3),
                "leverage_offense_quality": _to_native(row.get("leverage_offense_quality"), 2),
                "leverage_pitching_quality": _to_native(row.get("leverage_pitching_quality"), 2),
                "leverage_net_quality": _to_native(row.get("leverage_net_quality"), 2),
                "clutch_index": _to_native(row.get("clutch_index"), 2),
                "clutch_offense_delta_ops": _to_native(row.get("clutch_offense_delta_ops"), 3),
                "clutch_pitching_delta_ops": _to_native(row.get("clutch_pitching_delta_ops"), 3),
                "joe_score": _to_native(row.get("joe_score"), 2),
                "joe_rank": _to_native(row.get("joe_rank")),
                "joe_confidence": _to_native(row.get("joe_confidence"), 1),
                "joe_band_low": _to_native(row.get("joe_band_low"), 1),
                "joe_band_high": _to_native(row.get("joe_band_high"), 1),
                "joe_uncertainty_points": _to_native(row.get("joe_uncertainty_points"), 1),
                "joe_uncertainty_level": row.get("joe_uncertainty_level") or "medium",
                "joe_weight_xwobacon_pct": _to_native(row.get("joe_weight_xwobacon_pct"), 1),
                "joe_weight_xfip_pct": _to_native(row.get("joe_weight_xfip_pct"), 1),
                "joe_weight_clutch_pct": _to_native(row.get("joe_weight_clutch_pct"), 1),
                "live_power_score": _to_native(row.get("live_power_score"), 2),
                "d_plus_score": _to_native(row.get("d_plus_score"), 2),
                "power_minus_d_plus": _to_native(row.get("power_minus_d_plus"), 3),
                "power_minus_mscore": _to_native(row.get("power_minus_mscore"), 3),
                "talent_score": _to_native(row.get("talent_score"), 2),
                "form_score": _to_native(row.get("form_score"), 2),
                "risk_score": _to_native(row.get("risk_score"), 2),
                "power_context_adjustment": _to_native(row.get("power_context_adjustment"), 3),
                "power_context_confidence": _to_native(row.get("power_context_confidence"), 1),
                "coaching_execution_score": _to_native(row.get("coaching_execution_score"), 2),
                "coaching_execution_adjustment": _to_native(row.get("coaching_execution_adjustment"), 3),
                "coaching_execution_confidence": _to_native(row.get("coaching_execution_confidence"), 1),
                "coaching_execution_weight": _to_native(row.get("coaching_execution_weight"), 2),
                "wins": _to_native(row.get("wins")),
                "losses": _to_native(row.get("losses")),
                "win_pct": _to_native(row.get("win_pct"), 3),
                "run_diff": _to_native(row.get("run_diff")),
                "streak": row.get("streak") or "-",
                "games_back": row.get("games_back") or "-",
                "projected_win_pct": _to_native(row.get("projected_win_pct"), 3),
                "projected_wins_162": _to_native(row.get("projected_wins_162"), 1),
                "workbook_stats": {},
                "live_stats": live_stats,
                "mlb_api_stats": mlb_api_stats,
            }
        )

    return records



def _build_mscore_model_diagnostics(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"formula": "", "components": [], "ablation": []}

    component_defs = [
        {
            "feature": "mscore_base_component",
            "label": "Base Bucket",
            "formula": "win profile + sweep resilience",
            "values": pd.to_numeric(df["mscore_base_component"], errors="coerce").fillna(0.0),
        },
        {
            "feature": "mscore_offense_component",
            "label": "Offense Bucket",
            "formula": "OPS/BsR proxy/PA per game/R per game blend",
            "values": pd.to_numeric(df["mscore_offense_component"], errors="coerce").fillna(0.0),
        },
        {
            "feature": "mscore_defense_component",
            "label": "Defense Bucket",
            "formula": "starter + bullpen + fielding/catcher/zone/run-prevention blend",
            "values": pd.to_numeric(df["mscore_defense_component"], errors="coerce").fillna(0.0),
        },
        {
            "feature": "mscore_context_component",
            "label": "Context Bucket",
            "formula": "pythag/regression + starter/bullpen differential + expected gap + leverage/clutch pressure terms",
            "values": pd.to_numeric(df["mscore_context_component"], errors="coerce").fillna(0.0),
        },
    ]

    base_score = pd.Series(0.0, index=df.index, dtype=float)
    for component in component_defs:
        base_score = base_score + component["values"]

    base_rank = base_score.rank(ascending=False, method="min")
    base_order = list(df.loc[base_score.sort_values(ascending=False).index, "Team"])
    base_top5 = set(base_order[:5])

    raw_span = float(base_score.max() - base_score.min())
    raw_span = raw_span if raw_span > 1e-9 else 1.0

    spreads: list[float] = []
    for component in component_defs:
        centered = component["values"] - float(component["values"].mean())
        spreads.append(float(centered.abs().mean()))
    total_spread = sum(spreads) or 1.0

    components: list[dict[str, Any]] = []
    ablation: list[dict[str, Any]] = []

    for idx, component in enumerate(component_defs):
        values = component["values"]
        centered_points = ((values - float(values.mean())) / raw_span) * 100.0
        spread = spreads[idx]
        weight = spread / total_spread

        components.append(
            {
                "feature": component["feature"],
                "label": component["label"],
                "formula": component["formula"],
                "weight": round(float(weight), 4),
                "weight_pct": round(float(weight * 100.0), 2),
                "avg_points": round(float(centered_points.abs().mean()), 3),
                "std_points": round(float(centered_points.std(ddof=0)), 3),
            }
        )

        reduced_score = base_score - values
        reduced_rank = reduced_score.rank(ascending=False, method="min")

        rank_shift = (base_rank - reduced_rank).abs()
        mean_shift = float(rank_shift.mean())
        max_shift = int(rank_shift.max()) if len(rank_shift) else 0

        reduced_order = list(df.loc[reduced_score.sort_values(ascending=False).index, "Team"])
        reduced_top5 = set(reduced_order[:5])
        top5_turnover = len(base_top5.symmetric_difference(reduced_top5)) // 2

        impacted_rows: list[dict[str, Any]] = []
        impacted_idx = list(rank_shift.sort_values(ascending=False).index[:3])
        for impacted in impacted_idx:
            shift_value = int(rank_shift.loc[impacted])
            if shift_value <= 0:
                continue
            impacted_rows.append(
                {
                    "team": str(df.loc[impacted, "Team"]),
                    "base_rank": int(base_rank.loc[impacted]),
                    "ablated_rank": int(reduced_rank.loc[impacted]),
                    "rank_shift": shift_value,
                }
            )

        ablation.append(
            {
                "feature": component["feature"],
                "label": component["label"],
                "mean_abs_rank_shift": round(mean_shift, 3),
                "max_rank_shift": max_shift,
                "top5_turnover": int(top5_turnover),
                "most_impacted_teams": impacted_rows,
            }
        )

    components.sort(key=lambda row: row["weight"], reverse=True)
    ablation.sort(key=lambda row: row["mean_abs_rank_shift"], reverse=True)

    return {
        "formula": "mscore = sigmoid((base_bucket + offense_bucket + defense_bucket + context_bucket - 2.90)/0.55) * 100, with confidence shrinkage and leverage/clutch pressure context",
        "components": components,
        "ablation": ablation,
    }


def build_snapshot(workbook_path: Path = WORKBOOK_PATH) -> dict[str, Any]:
    tz = ZoneInfo(TIMEZONE)
    now = dt.datetime.now(tz)
    today = now.date()

    standings = fetch_standings(season=now.year)
    games_window = fetch_games_window(reference_date=today, days_back=1, days_forward=1)
    mlb_api_by_team, mlb_api_catalog = fetch_team_api_metrics(season=now.year)

    teams_df = _build_team_dataframe(standings=standings, mlb_api_by_team=mlb_api_by_team)
    mscore_diagnostics = _build_mscore_model_diagnostics(teams_df)

    today_games = [game for game in games_window if _safe_iso_date(game.get("official_date")) == today]
    games_for_dashboard = _select_relevant_games(games_window, today=today)

    total_games_today = len(today_games)
    completed_games_today = sum(1 for game in today_games if game.get("is_final"))
    live_games_today = sum(1 for game in today_games if game.get("is_live"))
    has_live_games = any(game.get("is_live") for game in games_for_dashboard)

    top_row = teams_df.iloc[0]
    teams = _teams_to_records(teams_df, mlb_api_by_team=mlb_api_by_team)

    predictions = [
        {
            "team": team["team"],
            "live_power_score": team["live_power_score"],
            "projected_win_pct": team["projected_win_pct"],
            "projected_wins_162": team["projected_wins_162"],
        }
        for team in teams[:10]
    ]

    prediction_error = None
    matchup_predictions: list[dict[str, Any]] = []
    backtest_summary: dict[str, Any] = {}
    win_model_diagnostics: dict[str, Any] = {"feature_importance": [], "ablation": {"baseline": {}, "rows": []}}
    archive_status = get_prediction_archive_status(limit=14)
    market_clv_tracker = _build_market_clv_tracker(archive_status.get("entries", []), max_days=30)

    try:
        backtest_report = get_last_season_report(reference_date=today)
        matchup_bundle = get_today_matchup_predictions(
            reference_date=today,
            current_season=now.year,
            backtest_season=today.year - 1,
        )

        matchup_predictions = matchup_bundle.get("games", [])
        model_block = matchup_bundle.get("model", {})
        win_model_diagnostics = {
            "season": backtest_report.get("season"),
            "trained_at": backtest_report.get("trained_at"),
            "feature_importance": model_block.get("feature_importance", backtest_report.get("feature_importance", [])),
            "ablation": model_block.get("ablation", backtest_report.get("ablation", {"baseline": {}, "rows": []})),
            "metrics_test": backtest_report.get("metrics", {}).get("test", {}),
            "metrics_validation": backtest_report.get("metrics", {}).get("validation", {}),
            "rolling_cv_mean": model_block.get("rolling_cv_mean", backtest_report.get("metrics", {}).get("rolling_cv", {}).get("mean", {})),
            "calibration_test": model_block.get("calibration_test", backtest_report.get("metrics", {}).get("calibration_test", {})),
            "drift_monitor": model_block.get("drift_monitor", backtest_report.get("metrics", {}).get("drift_monitor", {})),
            "market_blend_policy": model_block.get("market_blend_policy", {}),
            "market_blend_summary": model_block.get("market_blend_summary", {}),
            "pregame_context_summary": model_block.get("pregame_context_summary", {}),
        }

        backtest_summary = {
            "season": backtest_report.get("season"),
            "games_total": backtest_report.get("games_total"),
            "games_validation": backtest_report.get("games_validation"),
            "games_test": backtest_report.get("games_test"),
            "metrics_validation": backtest_report.get("metrics", {}).get("validation", {}),
            "metrics_test": backtest_report.get("metrics", {}).get("test", {}),
            "rolling_cv_mean": model_block.get("rolling_cv_mean", backtest_report.get("metrics", {}).get("rolling_cv", {}).get("mean", {})),
            "calibration_test": model_block.get("calibration_test", backtest_report.get("metrics", {}).get("calibration_test", {})),
            "drift_monitor": model_block.get("drift_monitor", backtest_report.get("metrics", {}).get("drift_monitor", {})),
            "market_blend_policy": model_block.get("market_blend_policy", {}),
            "market_blend_summary": model_block.get("market_blend_summary", {}),
            "pregame_context_summary": model_block.get("pregame_context_summary", {}),
            "blend_weight": model_block.get("blend_weight"),
            "calibration": model_block.get("calibration", {}),
            "model_version": model_block.get("model_version", backtest_report.get("model_version")),
            "model_variant": model_block.get("model_variant", backtest_report.get("model_variant", "v4")),
            "seasons_back": model_block.get("seasons_back", backtest_report.get("seasons_back", 3)),
            "market_odds_enabled": bool(matchup_bundle.get("market_odds_enabled")),
            "market_status": matchup_bundle.get("market_status"),
            "market_matchups_with_lines": matchup_bundle.get("market_matchups_with_lines"),
            "trained_at": backtest_report.get("trained_at"),
        }
    except Exception as exc:
        prediction_error = str(exc)

    _attach_matchup_prediction_deltas(
        matchup_predictions,
        reference_date=today,
        archive_entries=archive_status.get("entries", []),
    )

    prediction_value_board = _build_prediction_value_board(matchup_predictions)
    market_matchups_today = sum(1 for row in prediction_value_board if row.get("market_edge_pick") is not None)
    actionable_value_spots = sum(1 for row in prediction_value_board if bool(row.get("bet_quality_actionable")))
    high_quality_value_spots = sum(1 for row in prediction_value_board if float(row.get("bet_quality_score") or 0.0) >= 75.0)

    return {
        "meta": {
            "generated_at": now.isoformat(),
            "timezone": TIMEZONE,
            "reference_local_date": today.isoformat(),
            "season": now.year,
            "source_mode": "mlb_api_realtime",
            "workbook_path": str(workbook_path),
            "refresh_interval_minutes": REFRESH_INTERVAL_MINUTES,
            "workbook_sheet_count": 0,
            "workbook_metric_count": 0,
            "mlb_api_metric_count": len(mlb_api_catalog),
            "has_live_games": has_live_games,
            "games_window_start": (today - dt.timedelta(days=1)).isoformat(),
            "games_window_end": (today + dt.timedelta(days=1)).isoformat(),
            "prediction_error": prediction_error,
        },
        "summary": {
            "top_team": top_row["Team"],
            "top_live_power_score": _to_native(top_row["live_power_score"], 2),
            "average_mscore": _to_native(teams_df["mscore"].mean(), 3),
            "average_win_pct": _to_native(teams_df["win_pct"].mean(), 3),
            "total_games_today": total_games_today,
            "completed_games_today": completed_games_today,
            "live_games_today": live_games_today,
            "displayed_games": len(games_for_dashboard),
            "matchup_predictions_count": len(matchup_predictions),
            "market_matchups_today": market_matchups_today,
            "actionable_value_spots": actionable_value_spots,
            "high_quality_value_spots": high_quality_value_spots,
        },
        "teams": teams,
        "games_today": games_for_dashboard,
        "predictions": predictions,
        "matchup_predictions": matchup_predictions,
        "prediction_value_board": prediction_value_board,
        "prediction_engine": backtest_summary,
        "model_diagnostics": {
            "win_model": win_model_diagnostics,
            "mscore_model": mscore_diagnostics,
            "archive_tracking": archive_status,
            "market_clv": market_clv_tracker,
        },
        "stat_catalog": {
            "ranking_table": RANKING_TABLE_DEFINITIONS,
            "live": LIVE_STAT_CATALOG,
            "mlb_api": mlb_api_catalog,
            "workbook": [],
        },
    }



RRG_ALLOWED_METRICS = {
    "live_power_score",
    "mscore",
    "talent_score",
    "form_score",
    "risk_score",
    "d_plus_score",
    "joe_score",
    "power_minus_d_plus",
    "power_minus_mscore",
    "coaching_execution_score",
    "leverage_net_quality",
    "clutch_index",
}


def _snapshot_archive_path(reference_date: dt.date) -> Path:
    return SNAPSHOT_ARCHIVE_DIR / f"{reference_date.isoformat()}.json"


def _snapshot_archive_payload(snapshot: dict[str, Any]) -> dict[str, Any]:
    teams = []
    for team in snapshot.get("teams", []):
        teams.append(
            {
                "team": team.get("team"),
                "live_rank": team.get("live_rank"),
                "win_pct": team.get("win_pct"),
                "run_diff": team.get("run_diff"),
                "mscore": team.get("mscore"),
                "live_power_score": team.get("live_power_score"),
                "d_plus_score": team.get("d_plus_score"),
                "joe_score": team.get("joe_score"),
                "joe_rank": team.get("joe_rank"),
                "joe_confidence": team.get("joe_confidence"),
                "joe_band_low": team.get("joe_band_low"),
                "joe_band_high": team.get("joe_band_high"),
                "joe_uncertainty_points": team.get("joe_uncertainty_points"),
                "joe_uncertainty_level": team.get("joe_uncertainty_level"),
                "joe_weight_xwobacon_pct": team.get("joe_weight_xwobacon_pct"),
                "joe_weight_xfip_pct": team.get("joe_weight_xfip_pct"),
                "joe_weight_clutch_pct": team.get("joe_weight_clutch_pct"),
                "power_minus_d_plus": team.get("power_minus_d_plus"),
                "power_minus_mscore": team.get("power_minus_mscore"),
                "talent_score": team.get("talent_score"),
                "form_score": team.get("form_score"),
                "risk_score": team.get("risk_score"),
                "power_context_adjustment": team.get("power_context_adjustment"),
                "power_context_confidence": team.get("power_context_confidence"),
                "coaching_execution_score": team.get("coaching_execution_score"),
                "coaching_execution_adjustment": team.get("coaching_execution_adjustment"),
                "coaching_execution_confidence": team.get("coaching_execution_confidence"),
                "coaching_execution_weight": team.get("coaching_execution_weight"),
                "leverage_net_quality": team.get("leverage_net_quality"),
                "clutch_index": team.get("clutch_index"),
            }
        )

    return {
        "reference_date": snapshot.get("meta", {}).get("reference_local_date"),
        "generated_at": snapshot.get("meta", {}).get("generated_at"),
        "season": snapshot.get("meta", {}).get("season"),
        "teams": teams,
    }


def write_snapshot_archive(
    snapshot: dict[str, Any],
    reference_date: dt.date | None = None,
) -> dict[str, Any]:
    if reference_date is None:
        ref = _safe_iso_date(snapshot.get("meta", {}).get("reference_local_date"))
        reference_date = ref or dt.date.today()

    payload = _snapshot_archive_payload(snapshot)
    path = _snapshot_archive_path(reference_date)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def read_snapshot_archive(reference_date: dt.date) -> dict[str, Any] | None:
    path = _snapshot_archive_path(reference_date)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def list_snapshot_archives(limit: int = 240) -> list[dict[str, Any]]:
    if not SNAPSHOT_ARCHIVE_DIR.exists():
        return []

    files = sorted(SNAPSHOT_ARCHIVE_DIR.glob("*.json"), key=lambda item: item.name, reverse=True)
    rows: list[dict[str, Any]] = []

    for path in files[: max(1, limit)]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        rows.append(
            {
                "reference_date": payload.get("reference_date"),
                "generated_at": payload.get("generated_at"),
                "season": payload.get("season"),
                "teams": len(payload.get("teams", [])),
                "path": str(path),
            }
        )

    return rows


def build_archive_snapshot_for_date(reference_date: dt.date) -> dict[str, Any]:
    tz = ZoneInfo(TIMEZONE)
    now = dt.datetime.now(tz)
    season = reference_date.year

    standings = fetch_standings(season=season, reference_date=reference_date)
    mlb_api_by_team, _ = fetch_team_api_metrics(season=season, through_date=reference_date)

    if not standings and not mlb_api_by_team:
        teams: list[dict[str, Any]] = []
    else:
        teams_df = _build_team_dataframe(standings=standings, mlb_api_by_team=mlb_api_by_team)
        teams = _teams_to_records(teams_df, mlb_api_by_team=mlb_api_by_team) if not teams_df.empty else []

    return {
        "meta": {
            "generated_at": now.isoformat(),
            "timezone": TIMEZONE,
            "reference_local_date": reference_date.isoformat(),
            "season": season,
            "source_mode": "mlb_api_historical_backfill",
            "has_live_games": False,
        },
        "teams": teams,
    }


def backfill_snapshot_archives(
    days_back: int = 30,
    end_date: dt.date | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    if end_date is None:
        end_date = dt.date.today()

    safe_days = max(1, min(int(days_back), 365))
    start_date = end_date - dt.timedelta(days=safe_days - 1)

    created: list[str] = []
    skipped: list[str] = []
    errors: list[dict[str, str]] = []

    current = start_date
    while current <= end_date:
        if current > dt.date.today():
            current += dt.timedelta(days=1)
            continue

        path = _snapshot_archive_path(current)
        if path.exists() and not overwrite:
            skipped.append(current.isoformat())
            current += dt.timedelta(days=1)
            continue

        try:
            snapshot = build_archive_snapshot_for_date(reference_date=current)
            write_snapshot_archive(snapshot=snapshot, reference_date=current)
            created.append(current.isoformat())
        except Exception as exc:
            errors.append({"date": current.isoformat(), "error": str(exc)})

        current += dt.timedelta(days=1)

    archive_count = len(list_snapshot_archives(limit=5000))

    return {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "requested_days": safe_days,
        "overwrite": bool(overwrite),
        "created_count": len(created),
        "skipped_count": len(skipped),
        "error_count": len(errors),
        "created_dates": created,
        "skipped_dates": skipped,
        "errors": errors[:20],
        "archive_count": archive_count,
    }


def _rrg_quadrant(x_value: float, y_value: float) -> str:
    if x_value >= 100 and y_value >= 100:
        return "Leading"
    if x_value >= 100 and y_value < 100:
        return "Weakening"
    if x_value < 100 and y_value < 100:
        return "Lagging"
    return "Improving"


def _resolve_rrg_metric(metric: str | None) -> str:
    metric_raw = str(metric or "power").strip().lower()
    alias = {
        "power": "live_power_score",
        "mscore": "mscore",
        "talent": "talent_score",
        "form": "form_score",
        "risk": "risk_score",
        "dplus": "d_plus_score",
        "d_plus": "d_plus_score",
        "d+": "d_plus_score",
        "joe": "joe_score",
        "joe_score": "joe_score",
        "gap": "power_minus_d_plus",
        "power_minus": "power_minus_d_plus",
        "power_minus_d_plus": "power_minus_d_plus",
        "power_minus_mscore": "power_minus_mscore",
    }

    metric_key = alias.get(metric_raw, metric_raw)
    if metric_key not in RRG_ALLOWED_METRICS:
        return "live_power_score"
    return metric_key


def _build_rrg_signal_validation(
    team_frames: dict[str, pd.DataFrame],
    league_means: dict[str, float],
    horizon_days: int = 7,
) -> dict[str, Any]:
    leading_excess: list[float] = []
    improving_excess: list[float] = []

    for _team, frame in team_frames.items():
        if len(frame) <= horizon_days + 1:
            continue

        quadrants = [_rrg_quadrant(float(x), float(y)) for x, y in zip(frame["rs_ratio"], frame["rs_momentum"])]

        for index in range(1, len(frame) - horizon_days):
            prev_q = quadrants[index - 1]
            cur_q = quadrants[index]

            date_now = str(frame.iloc[index]["date"])
            date_future = str(frame.iloc[index + horizon_days]["date"])

            league_now = league_means.get(date_now)
            league_future = league_means.get(date_future)
            if league_now is None or league_future is None:
                continue

            metric_now = float(frame.iloc[index]["metric"])
            metric_future = float(frame.iloc[index + horizon_days]["metric"])
            excess = (metric_future - metric_now) - (float(league_future) - float(league_now))

            if cur_q == "Leading" and prev_q != "Leading":
                leading_excess.append(float(excess))
            if cur_q == "Improving" and prev_q != "Improving":
                improving_excess.append(float(excess))

    def _summary(values: list[float]) -> tuple[int, float | None, float | None]:
        if not values:
            return 0, None, None
        hit_rate = float(sum(1 for value in values if value > 0) / len(values))
        avg_excess = float(np.mean(values))
        return len(values), round(hit_rate, 4), round(avg_excess, 4)

    lead_n, lead_hit, lead_avg = _summary(leading_excess)
    imp_n, imp_hit, imp_avg = _summary(improving_excess)

    return {
        "horizon_days": int(max(1, horizon_days)),
        "leading_entries": lead_n,
        "leading_hit_rate": lead_hit,
        "leading_avg_excess": lead_avg,
        "improving_entries": imp_n,
        "improving_hit_rate": imp_hit,
        "improving_avg_excess": imp_avg,
    }


def build_rrg_payload(
    metric: str = "power",
    lookback_days: int = 120,
    trail_days: int = 16,
    reference_date: dt.date | None = None,
    min_games: int = 8,
) -> dict[str, Any]:
    metric_key = _resolve_rrg_metric(metric)
    if reference_date is None:
        reference_date = dt.date.today()

    lookback = max(20, min(int(lookback_days), 365))
    trail = max(3, min(int(trail_days), 45))
    min_points = max(3, min(int(min_games), 60))

    latest = read_snapshot()
    if latest:
        latest_date = _safe_iso_date(latest.get("meta", {}).get("reference_local_date"))
        if latest_date is not None and not _snapshot_archive_path(latest_date).exists():
            try:
                write_snapshot_archive(latest, reference_date=latest_date)
            except Exception:
                pass

    start_date = reference_date - dt.timedelta(days=lookback - 1)
    archives: list[dict[str, Any]] = []

    if SNAPSHOT_ARCHIVE_DIR.exists():
        for path in sorted(SNAPSHOT_ARCHIVE_DIR.glob("*.json"), key=lambda item: item.name):
            ref = _safe_iso_date(path.stem)
            if ref is None or ref < start_date or ref > reference_date:
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            archives.append(payload)

    if not archives and latest:
        archives = [_snapshot_archive_payload(latest)]

    by_team: dict[str, list[dict[str, Any]]] = {}
    league_means: dict[str, float] = {}
    league_stds: dict[str, float] = {}

    for payload in archives:
        date_iso = str(payload.get("reference_date") or "")
        if not date_iso:
            continue

        teams = payload.get("teams", [])
        metric_values: list[float] = []
        for row in teams:
            raw_metric = row.get(metric_key)
            if raw_metric is None and metric_key == "power_minus_d_plus":
                raw_metric = row.get("power_minus_mscore")
            value = pd.to_numeric(raw_metric, errors="coerce")
            if pd.notna(value):
                metric_values.append(float(value))

        if len(metric_values) < 6:
            continue

        league_mean = float(np.mean(metric_values))
        league_std = float(np.std(metric_values))
        if league_std < 1e-6:
            league_std = 1.0
        league_means[date_iso] = league_mean
        league_stds[date_iso] = league_std

        for row in teams:
            team_name = row.get("team")
            if not team_name:
                continue

            raw_metric = row.get(metric_key)
            if raw_metric is None and metric_key == "power_minus_d_plus":
                raw_metric = row.get("power_minus_mscore")
            metric_val = pd.to_numeric(raw_metric, errors="coerce")
            if pd.isna(metric_val):
                continue

            numeric_metric = float(metric_val)
            std = league_stds.get(date_iso, 1.0)
            rel_strength = 100.0 + ((numeric_metric - league_mean) / max(1e-6, float(std))) * 8.0
            rel_strength = float(np.clip(rel_strength, 70.0, 130.0))

            by_team.setdefault(str(team_name), []).append(
                {
                    "date": date_iso,
                    "metric": numeric_metric,
                    "relative": float(rel_strength),
                    "live_rank": row.get("live_rank"),
                    "power_minus_d_plus": row.get("power_minus_d_plus", row.get("power_minus_mscore")),
                    "power_minus_mscore": row.get("power_minus_mscore"),
                    "form_score": row.get("form_score"),
                    "win_pct": row.get("win_pct"),
                }
            )

    max_points_available = max((len(rows) for rows in by_team.values()), default=0)
    effective_min_points = min_points
    history_is_limited = False
    history_note: str | None = None
    if max_points_available > 0 and max_points_available < min_points:
        effective_min_points = max(1, max_points_available)
        history_is_limited = True
        history_note = (
            f"Limited history mode: only {max_points_available} archived day(s) available. "
            "RRG momentum uses a form-based proxy until enough daily archives exist for true rotation dynamics."
        )

    points: list[dict[str, Any]] = []
    team_frames: dict[str, pd.DataFrame] = {}
    viewport_x_values: list[float] = []
    viewport_y_values: list[float] = []

    for team_name, rows in by_team.items():
        frame = pd.DataFrame(rows)
        frame = frame.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
        if len(frame) < effective_min_points:
            continue

        rel = pd.to_numeric(frame["relative"], errors="coerce").fillna(100.0)
        rs_ratio = rel.ewm(span=7, adjust=False, min_periods=1).mean()

        if len(frame) >= 4:
            delta = rs_ratio - rs_ratio.shift(3).fillna(rs_ratio.iloc[0])
            rs_momentum = 100.0 + delta * 2.5
        elif len(frame) >= 2:
            delta = rs_ratio - rs_ratio.shift(1).fillna(rs_ratio.iloc[0])
            rs_momentum = 100.0 + delta * 3.2
        else:
            # Early-history fallback: use form context as a momentum proxy so chart is not flat at 100.
            form_proxy = pd.to_numeric(frame.get("form_score"), errors="coerce").fillna(50.0)
            rs_momentum = 100.0 + (form_proxy - 50.0) * 0.36

        rs_momentum = rs_momentum.clip(85.0, 115.0)

        frame["rs_ratio"] = rs_ratio
        frame["rs_momentum"] = rs_momentum
        team_frames[team_name] = frame

        viewport_x_values.extend([float(v) for v in frame["rs_ratio"].tolist() if pd.notna(v)])
        viewport_y_values.extend([float(v) for v in frame["rs_momentum"].tolist() if pd.notna(v)])

        latest_row = frame.iloc[-1]
        latest_metric = float(latest_row["metric"])

        delta_days = max(0, min(7, len(frame) - 1))
        delta_7d = None
        if delta_days > 0:
            delta_7d = float(latest_metric - float(frame.iloc[-(delta_days + 1)]["metric"]))

        trail_rows = frame.iloc[-min(trail, len(frame)) :]
        trail_points = [
            {
                "date": str(row["date"]),
                "x": round(float(row["rs_ratio"]), 3),
                "y": round(float(row["rs_momentum"]), 3),
                "metric": round(float(row["metric"]), 3),
            }
            for _, row in trail_rows.iterrows()
        ]

        x_val = float(latest_row["rs_ratio"])
        y_val = float(latest_row["rs_momentum"])
        points.append(
            {
                "team": team_name,
                "x": round(x_val, 3),
                "y": round(y_val, 3),
                "quadrant": _rrg_quadrant(x_val, y_val),
                "metric": round(latest_metric, 3),
                "delta_7d": round(delta_7d, 3) if delta_7d is not None else None,
                "delta_days": int(delta_days),
                "live_rank": _to_native(latest_row.get("live_rank")),
                "power_minus_d_plus": _to_native(latest_row.get("power_minus_d_plus", latest_row.get("power_minus_mscore")), 3),
                "power_minus_mscore": _to_native(latest_row.get("power_minus_mscore"), 3),
                "win_pct": _to_native(latest_row.get("win_pct"), 3),
                "trail": trail_points,
            }
        )

    points.sort(key=lambda row: (row.get("quadrant", ""), -float(row.get("x", 0.0))))

    quadrant_counts = {
        "Leading": sum(1 for row in points if row.get("quadrant") == "Leading"),
        "Weakening": sum(1 for row in points if row.get("quadrant") == "Weakening"),
        "Lagging": sum(1 for row in points if row.get("quadrant") == "Lagging"),
        "Improving": sum(1 for row in points if row.get("quadrant") == "Improving"),
    }

    validation = _build_rrg_signal_validation(team_frames=team_frames, league_means=league_means, horizon_days=7)

    viewport = {
        "x_min": 90.0,
        "x_max": 110.0,
        "y_min": 90.0,
        "y_max": 110.0,
    }

    if viewport_x_values and viewport_y_values:
        x_array = np.array(viewport_x_values, dtype=float)
        y_array = np.array(viewport_y_values, dtype=float)

        x_lo = float(np.quantile(x_array, 0.03))
        x_hi = float(np.quantile(x_array, 0.97))
        y_lo = float(np.quantile(y_array, 0.03))
        y_hi = float(np.quantile(y_array, 0.97))

        x_span = max(8.0, x_hi - x_lo)
        y_span = max(8.0, y_hi - y_lo)
        x_pad = max(1.2, x_span * 0.12)
        y_pad = max(1.2, y_span * 0.12)

        x_min = x_lo - x_pad
        x_max = x_hi + x_pad
        y_min = y_lo - y_pad
        y_max = y_hi + y_pad

        # Ensure current points are always visible.
        if points:
            cur_x = [float(item.get("x", 100.0)) for item in points]
            cur_y = [float(item.get("y", 100.0)) for item in points]
            x_min = min(x_min, min(cur_x) - 0.8)
            x_max = max(x_max, max(cur_x) + 0.8)
            y_min = min(y_min, min(cur_y) - 0.8)
            y_max = max(y_max, max(cur_y) + 0.8)

        x_min = max(70.0, x_min)
        x_max = min(130.0, x_max)
        y_min = max(70.0, y_min)
        y_max = min(130.0, y_max)

        if x_max - x_min < 10.0:
            x_mid = (x_min + x_max) / 2.0
            x_min = x_mid - 5.0
            x_max = x_mid + 5.0

        if y_max - y_min < 10.0:
            y_mid = (y_min + y_max) / 2.0
            y_min = y_mid - 5.0
            y_max = y_mid + 5.0

        viewport = {
            "x_min": round(float(x_min), 3),
            "x_max": round(float(x_max), 3),
            "y_min": round(float(y_min), 3),
            "y_max": round(float(y_max), 3),
        }

    return {
        "reference_date": reference_date.isoformat(),
        "metric": metric_key,
        "lookback_days": lookback,
        "trail_days": trail,
        "min_points": min_points,
        "effective_min_points": effective_min_points,
        "history_points_available": int(max_points_available),
        "history_is_limited": history_is_limited,
        "history_note": history_note,
        "points": points,
        "quadrant_counts": quadrant_counts,
        "signal_validation": validation,
        "viewport": viewport,
        "quadrant_guide": {
            "Leading": "Strong relative strength with rising momentum.",
            "Weakening": "Still strong, but momentum is fading.",
            "Lagging": "Weak relative strength and weak momentum.",
            "Improving": "Weak base, but momentum is improving.",
        },
    }

def write_snapshot(snapshot: dict[str, Any], cache_path: Path = CACHE_PATH) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")


def read_snapshot(cache_path: Path = CACHE_PATH) -> dict[str, Any] | None:
    if not cache_path.exists():
        return None
    return json.loads(cache_path.read_text(encoding="utf-8"))


def refresh_snapshot(workbook_path: Path = WORKBOOK_PATH, cache_path: Path = CACHE_PATH) -> dict[str, Any]:
    snapshot = build_snapshot(workbook_path=workbook_path)
    write_snapshot(snapshot=snapshot, cache_path=cache_path)

    ref_date = _safe_iso_date(snapshot.get("meta", {}).get("reference_local_date"))
    write_snapshot_archive(snapshot=snapshot, reference_date=ref_date or dt.date.today())
    return snapshot
