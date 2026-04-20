from __future__ import annotations

import datetime as dt
import os
import threading
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import (
    AUTO_REFRESH_ON_STARTUP,
    CACHE_MAX_AGE_MINUTES,
    LIVE_REFRESH_SECONDS,
    NIGHTLY_ARCHIVE_HOUR,
    NIGHTLY_ARCHIVE_MINUTE,
    NIGHTLY_RETRAIN_HOUR,
    NIGHTLY_RETRAIN_MINUTE,
    REFRESH_INTERVAL_MINUTES,
    TIMEZONE,
)
from app.services.prediction_service import (
    get_prediction_archive_status,
    get_today_matchup_predictions,
    list_prediction_archives,
    read_prediction_archive,
    run_backtest_for_season,
    run_backtest_model_comparison,
    run_nightly_retrain_and_archive,
    write_daily_prediction_archive,
)
from app.services.snapshot_service import (
    backfill_snapshot_archives,
    build_rrg_payload,
    read_snapshot,
    refresh_snapshot,
)

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
INDEX_FILE = STATIC_DIR / "index.html"

app = FastAPI(title="MLB Mscore Dashboard", version="0.7.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

scheduler: BackgroundScheduler | None = None
snapshot_lock = threading.Lock()
snapshot_cache: dict[str, Any] | None = None


def _now_local() -> dt.datetime:
    return dt.datetime.now(ZoneInfo(TIMEZONE))


def _refresh_and_cache() -> dict[str, Any]:
    global snapshot_cache
    with snapshot_lock:
        snapshot = refresh_snapshot()
        snapshot_cache = snapshot
        return snapshot


def _generated_at(snapshot: dict[str, Any]) -> dt.datetime | None:
    generated_at = snapshot.get("meta", {}).get("generated_at")
    if not generated_at:
        return None

    try:
        return dt.datetime.fromisoformat(generated_at)
    except ValueError:
        return None


def _cache_age_seconds(snapshot: dict[str, Any]) -> float:
    generated_dt = _generated_at(snapshot)
    if generated_dt is None:
        return float("inf")

    return max(0.0, (_now_local() - generated_dt).total_seconds())


def _needs_refresh(snapshot: dict[str, Any]) -> bool:
    age_seconds = _cache_age_seconds(snapshot)
    age_minutes = age_seconds / 60.0

    reference_local_date = snapshot.get("meta", {}).get("reference_local_date")
    if reference_local_date != _now_local().date().isoformat():
        return True

    has_live_games = bool(snapshot.get("meta", {}).get("has_live_games"))
    if has_live_games and age_seconds >= LIVE_REFRESH_SECONDS:
        return True

    return age_minutes >= CACHE_MAX_AGE_MINUTES


def _scheduled_refresh() -> None:
    try:
        _refresh_and_cache()
    except Exception as exc:  # pragma: no cover
        print(f"[scheduler] refresh failed: {exc}")


def _scheduled_nightly_retrain() -> None:
    try:
        today = _now_local().date()
        run_backtest_for_season(season=today.year - 1, force_retrain=True, model_variant="v4", seasons_back=3)
        _refresh_and_cache()
    except Exception as exc:  # pragma: no cover
        print(f"[scheduler] nightly retrain failed: {exc}")


def _scheduled_daily_archive() -> None:
    try:
        target_date = _now_local().date() - dt.timedelta(days=1)
        write_daily_prediction_archive(
            reference_date=target_date,
            current_season=target_date.year,
            backtest_season=target_date.year - 1,
            force_rebuild=True,
        )
    except Exception as exc:  # pragma: no cover
        print(f"[scheduler] daily archive failed: {exc}")


@app.on_event("startup")
def on_startup() -> None:
    global scheduler, snapshot_cache

    try:
        snapshot_cache = read_snapshot()

        if AUTO_REFRESH_ON_STARTUP or snapshot_cache is None or _needs_refresh(snapshot_cache):
            snapshot_cache = _refresh_and_cache()
    except Exception as exc:  # pragma: no cover
        print(f"[startup] initial refresh failed: {exc}")

    scheduler = BackgroundScheduler(timezone=TIMEZONE)
    scheduler.add_job(
        _scheduled_refresh,
        trigger="interval",
        minutes=REFRESH_INTERVAL_MINUTES,
        id="dashboard-refresh",
        replace_existing=True,
    )
    scheduler.add_job(
        _scheduled_nightly_retrain,
        trigger="cron",
        hour=NIGHTLY_RETRAIN_HOUR,
        minute=NIGHTLY_RETRAIN_MINUTE,
        id="nightly-retrain",
        replace_existing=True,
    )
    scheduler.add_job(
        _scheduled_daily_archive,
        trigger="cron",
        hour=NIGHTLY_ARCHIVE_HOUR,
        minute=NIGHTLY_ARCHIVE_MINUTE,
        id="daily-prediction-archive",
        replace_existing=True,
    )
    scheduler.start()


@app.on_event("shutdown")
def on_shutdown() -> None:
    if scheduler:
        scheduler.shutdown(wait=False)


@app.get("/")
def index() -> FileResponse:
    if not INDEX_FILE.exists():
        raise HTTPException(status_code=404, detail="Dashboard file not found")
    return FileResponse(path=INDEX_FILE)


@app.get("/api/health")
def health() -> dict[str, Any]:
    refresh_job = scheduler.get_job("dashboard-refresh") if scheduler else None
    nightly_job = scheduler.get_job("nightly-retrain") if scheduler else None
    archive_job = scheduler.get_job("daily-prediction-archive") if scheduler else None
    age_seconds = _cache_age_seconds(snapshot_cache) if snapshot_cache else None
    archive_status = get_prediction_archive_status(limit=1)
    odds_key_configured = bool(os.getenv("MSCORE_ODDS_API_KEY", "").strip())
    market_snapshot = snapshot_cache.get("prediction_engine", {}) if snapshot_cache else {}

    return {
        "status": "ok",
        "scheduler_running": bool(scheduler and scheduler.running),
        "refresh_interval_minutes": REFRESH_INTERVAL_MINUTES,
        "cache_max_age_minutes": CACHE_MAX_AGE_MINUTES,
        "live_refresh_seconds": LIVE_REFRESH_SECONDS,
        "nightly_retrain_hour": NIGHTLY_RETRAIN_HOUR,
        "nightly_retrain_minute": NIGHTLY_RETRAIN_MINUTE,
        "nightly_archive_hour": NIGHTLY_ARCHIVE_HOUR,
        "nightly_archive_minute": NIGHTLY_ARCHIVE_MINUTE,
        "next_refresh": refresh_job.next_run_time.isoformat() if refresh_job and refresh_job.next_run_time else None,
        "next_nightly_retrain": nightly_job.next_run_time.isoformat() if nightly_job and nightly_job.next_run_time else None,
        "next_daily_archive": archive_job.next_run_time.isoformat() if archive_job and archive_job.next_run_time else None,
        "last_refresh": snapshot_cache.get("meta", {}).get("generated_at") if snapshot_cache else None,
        "cache_age_seconds": round(age_seconds, 1) if age_seconds is not None else None,
        "latest_archive": archive_status.get("latest"),
        "odds_api_key_configured": odds_key_configured,
        "market_status": market_snapshot.get("market_status"),
        "market_matchups_with_lines": market_snapshot.get("market_matchups_with_lines"),
    }


@app.get("/api/dashboard")
def get_dashboard() -> dict[str, Any]:
    global snapshot_cache

    if snapshot_cache is None:
        snapshot_cache = read_snapshot()

    if snapshot_cache is None:
        return _refresh_and_cache()

    if _needs_refresh(snapshot_cache):
        try:
            snapshot_cache = _refresh_and_cache()
        except Exception:
            pass

    return snapshot_cache


@app.post("/api/refresh")
def force_refresh() -> dict[str, Any]:
    return _refresh_and_cache()


@app.get("/api/backtest/last-season")
def backtest_last_season(
    force_retrain: bool = False,
    model_variant: str = "v4",
    seasons_back: int = 3,
) -> dict[str, Any]:
    today = _now_local().date()
    season = today.year - 1
    return run_backtest_for_season(
        season=season,
        force_retrain=force_retrain,
        model_variant=model_variant,
        seasons_back=max(1, min(seasons_back, 8)),
    )


@app.get("/api/backtest/compare")
def backtest_compare(
    force_retrain: bool = False,
    seasons_back: int = 3,
) -> dict[str, Any]:
    today = _now_local().date()
    season = today.year - 1
    return run_backtest_model_comparison(
        season=season,
        force_retrain=force_retrain,
        seasons_back=max(1, min(seasons_back, 8)),
    )


@app.post("/api/backtest/nightly-run")
def run_nightly_backtest_and_archive() -> dict[str, Any]:
    return run_nightly_retrain_and_archive(reference_date=_now_local().date(), model_variant="v4", seasons_back=3)


@app.get("/api/predictions/today")
def predictions_today(
    model_variant: str = "v4",
    seasons_back: int = 3,
) -> dict[str, Any]:
    today = _now_local().date()
    return get_today_matchup_predictions(
        reference_date=today,
        current_season=today.year,
        backtest_season=today.year - 1,
        model_variant=model_variant,
        seasons_back=max(1, min(seasons_back, 8)),
    )


@app.post("/api/predictions/archive/daily")
def archive_predictions_daily(
    days_ago: int = 1,
    force_rebuild: bool = False,
    model_variant: str = "v4",
    seasons_back: int = 3,
) -> dict[str, Any]:
    days_ago = max(0, min(days_ago, 30))
    target_date = _now_local().date() - dt.timedelta(days=days_ago)
    return write_daily_prediction_archive(
        reference_date=target_date,
        current_season=target_date.year,
        backtest_season=target_date.year - 1,
        force_rebuild=force_rebuild,
        model_variant=model_variant,
        seasons_back=max(1, min(seasons_back, 8)),
    )


@app.get("/api/predictions/archive")
def prediction_archive(date: dt.date) -> dict[str, Any]:
    payload = read_prediction_archive(date)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"No archive found for {date.isoformat()}")
    return payload


@app.get("/api/predictions/archive/index")
def prediction_archive_index(limit: int = 30) -> dict[str, Any]:
    safe_limit = max(1, min(limit, 180))
    rows = list_prediction_archives(limit=safe_limit)
    return {
        "count": len(rows),
        "entries": rows,
    }


@app.get("/api/rrg")
def rrg_view(
    metric: str = "power",
    lookback_days: int = 120,
    trail_days: int = 16,
    min_points: int = 8,
) -> dict[str, Any]:
    today = _now_local().date()
    return build_rrg_payload(
        metric=metric,
        lookback_days=max(20, min(lookback_days, 365)),
        trail_days=max(3, min(trail_days, 45)),
        reference_date=today,
        min_games=max(3, min(min_points, 60)),
    )


@app.post("/api/archives/snapshots/backfill")
def backfill_snapshots(
    days: int = 30,
    overwrite: bool = False,
) -> dict[str, Any]:
    today = _now_local().date()
    return backfill_snapshot_archives(
        days_back=max(1, min(days, 365)),
        end_date=today,
        overwrite=overwrite,
    )

