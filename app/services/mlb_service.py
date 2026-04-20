from __future__ import annotations

import datetime as dt
import re
from typing import Any

import requests

BASE_API_URL = "https://statsapi.mlb.com/api/v1"
REQUEST_TIMEOUT_SECONDS = 20

LIVE_GAME_STATES = {"Live", "In Progress", "Manager Challenge", "Delayed", "Warmup"}
FINAL_GAME_STATES = {"Final", "Game Over", "Completed Early"}

API_STAT_DESCRIPTION_MAP = {
    "avg": "Batting average or opponent average, depending on stat group.",
    "obp": "On-base percentage.",
    "slg": "Slugging percentage.",
    "ops": "On-base plus slugging.",
    "era": "Earned run average.",
    "whip": "Walks and hits allowed per inning pitched.",
    "runs": "Total runs scored or allowed in the selected group.",
    "homeRuns": "Total home runs.",
    "strikeOuts": "Total strikeouts.",
    "baseOnBalls": "Total walks.",
    "wins": "Total team wins credited in the selected split.",
    "losses": "Total team losses credited in the selected split.",
    "fielding": "Fielding percentage.",
    "errors": "Official scoring errors.",
    "assists": "Total defensive assists.",
    "putOuts": "Total defensive putouts.",
    "doublePlays": "Total double plays turned.",
    "stolenBases": "Total stolen bases.",
    "caughtStealing": "Total runners caught stealing.",
    "plateAppearances": "Total plate appearances.",
    "babip": "Batting average on balls in play.",
    "iso": "Isolated power.",
}


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


def _clean_text(value: str) -> str:
    return " ".join(str(value).replace("\n", " ").split())


def _humanize_stat_key(stat_key: str) -> str:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", stat_key)
    return _clean_text(spaced).title()


def _api_stat_key(stat_type: str, group: str, stat_key: str) -> str:
    return f"mlb::{stat_type}::{group}::{stat_key}"


def _coerce_api_value(value: Any) -> Any:
    if isinstance(value, str):
        raw = value.strip()
        if raw in {"", "-", "--", "-.--"}:
            return None
        try:
            if "." in raw:
                number = float(raw)
                return int(number) if number.is_integer() else number
            return int(raw)
        except ValueError:
            try:
                return float(raw)
            except ValueError:
                return raw
    return value


def _describe_api_stat(stat_key: str, group: str, stat_type: str) -> str:
    direct = API_STAT_DESCRIPTION_MAP.get(stat_key)
    if direct:
        return f"{direct} Source: MLB Stats API ({group}, {stat_type})."
    human = _humanize_stat_key(stat_key)
    return f"{human} from MLB Stats API team {group} metrics ({stat_type})."


def fetch_team_name_lookup(season: int | None = None) -> dict[int, str]:
    if season is None:
        season = dt.date.today().year

    response = requests.get(
        f"{BASE_API_URL}/teams",
        params={"sportId": 1, "season": season},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()

    lookup: dict[int, str] = {}
    for team in payload.get("teams", []):
        team_id = team.get("id")
        team_name = team.get("name")
        if team_id is None or not team_name:
            continue
        lookup[int(team_id)] = str(team_name)

    return lookup


def fetch_standings(
    season: int | None = None,
    reference_date: dt.date | None = None,
) -> dict[str, dict[str, Any]]:
    if season is None:
        season = dt.date.today().year

    params: dict[str, Any] = {
        "leagueId": "103,104",
        "season": season,
        "standingsTypes": "regularSeason",
    }
    if reference_date is not None:
        params["date"] = reference_date.isoformat()

    response = requests.get(
        f"{BASE_API_URL}/standings",
        params=params,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()

    team_name_lookup = fetch_team_name_lookup(season=season)
    standings: dict[str, dict[str, Any]] = {}

    for division in payload.get("records", []):
        for row in division.get("teamRecords", []):
            team_obj = row.get("team", {})
            team_id = team_obj.get("id")
            team_name = team_obj.get("name")

            canonical_name = None
            if team_id is not None:
                canonical_name = team_name_lookup.get(int(team_id))
            if not canonical_name:
                canonical_name = team_name
            if not canonical_name:
                continue

            wins = _to_int(row.get("wins"), 0)
            losses = _to_int(row.get("losses"), 0)
            total_games = max(wins + losses, 1)
            win_pct = _to_float(row.get("winningPercentage"), wins / total_games)

            standings[canonical_name] = {
                "wins": wins,
                "losses": losses,
                "win_pct": round(win_pct, 3),
                "run_diff": _to_int(row.get("runDifferential"), 0),
                "division_rank": _to_int(row.get("divisionRank"), 0),
                "games_back": row.get("divisionGamesBack", "-"),
                "streak": row.get("streak", {}).get("streakCode", "-"),
            }

    return standings


def fetch_games_window(
    reference_date: dt.date | None = None,
    days_back: int = 1,
    days_forward: int = 1,
) -> list[dict[str, Any]]:
    if reference_date is None:
        reference_date = dt.date.today()

    start_date = reference_date - dt.timedelta(days=max(days_back, 0))
    end_date = reference_date + dt.timedelta(days=max(days_forward, 0))

    response = requests.get(
        f"{BASE_API_URL}/schedule",
        params={
            "sportId": 1,
            "startDate": start_date.isoformat(),
            "endDate": end_date.isoformat(),
            "hydrate": "linescore,team",
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()

    games: list[dict[str, Any]] = []

    for day_block in payload.get("dates", []):
        official_date = day_block.get("date")
        for game in day_block.get("games", []):
            status = game.get("status", {})
            linescore = game.get("linescore", {})
            teams = game.get("teams", {})

            inning_half = linescore.get("inningHalf")
            inning_number = linescore.get("currentInning")
            inning_display = None
            if inning_half and inning_number:
                inning_display = f"{inning_half} {inning_number}"

            state = status.get("abstractGameState", "Unknown")

            games.append(
                {
                    "game_pk": game.get("gamePk"),
                    "game_date": game.get("gameDate"),
                    "official_date": official_date,
                    "state": state,
                    "is_live": state in LIVE_GAME_STATES,
                    "is_final": state in FINAL_GAME_STATES,
                    "status": status.get("detailedState", "Unknown"),
                    "inning": inning_display,
                    "home_team": teams.get("home", {}).get("team", {}).get("name", ""),
                    "away_team": teams.get("away", {}).get("team", {}).get("name", ""),
                    "home_score": teams.get("home", {}).get("score"),
                    "away_score": teams.get("away", {}).get("score"),
                    "venue": game.get("venue", {}).get("name", ""),
                }
            )

    games.sort(key=lambda item: item.get("game_date") or "")
    return games


def fetch_games_for_date(target_date: dt.date | None = None) -> list[dict[str, Any]]:
    if target_date is None:
        target_date = dt.date.today()

    return [
        game
        for game in fetch_games_window(reference_date=target_date, days_back=0, days_forward=0)
        if game.get("official_date") == target_date.isoformat()
    ]


def fetch_team_api_metrics(
    season: int | None = None,
    through_date: dt.date | None = None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    if season is None:
        season = dt.date.today().year

    today = dt.date.today()
    historical_mode = through_date is not None and through_date < today

    params: dict[str, Any] = {
        "sportId": 1,
        "season": season,
        "group": "hitting,pitching,fielding",
    }

    if historical_mode:
        start_date = dt.date(season, 1, 1)
        params.update(
            {
                "stats": "byDateRange",
                "startDate": start_date.isoformat(),
                "endDate": through_date.isoformat(),
                "gameType": "R",
            }
        )
    else:
        params.update({"stats": "season,seasonAdvanced"})

    response = requests.get(
        f"{BASE_API_URL}/teams/stats",
        params=params,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()

    team_stats: dict[str, dict[str, Any]] = {}
    catalog: list[dict[str, str]] = []
    seen_keys: set[str] = set()

    for block in payload.get("stats", []):
        stat_type_raw = _clean_text(block.get("type", {}).get("displayName", "unknown"))
        stat_type = "season" if historical_mode and stat_type_raw == "byDateRange" else stat_type_raw
        group = _clean_text(block.get("group", {}).get("displayName", "misc"))
        group_title = group.capitalize()

        for split in block.get("splits", []):
            team_name = split.get("team", {}).get("name")
            if not team_name:
                continue

            team_bucket = team_stats.setdefault(team_name, {})
            stats = split.get("stat", {})

            for stat_key, stat_value in stats.items():
                composite_key = _api_stat_key(stat_type, group, stat_key)
                team_bucket[composite_key] = _coerce_api_value(stat_value)

                if composite_key in seen_keys:
                    continue

                seen_keys.add(composite_key)
                catalog.append(
                    {
                        "key": composite_key,
                        "label": f"{_humanize_stat_key(stat_key)} ({group_title}, {stat_type})",
                        "description": _describe_api_stat(stat_key, group, stat_type),
                        "group": f"MLB API - {group_title} ({stat_type})",
                        "sheet": "MLB Stats API",
                    }
                )

    return team_stats, catalog
