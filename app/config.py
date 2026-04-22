from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_WORKBOOK_PATH = BASE_DIR.parent / "2026 MLB Mscore_041326.xlsx"


def _load_dotenv(path: Path) -> None:
    if not path.exists() or not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.lower().startswith("export "):
            line = line[7:].strip()

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue

        parsed = value.strip()
        if len(parsed) >= 2 and parsed[0] == parsed[-1] and parsed[0] in {'"', "'"}:
            parsed = parsed[1:-1]

        os.environ[key] = parsed


def _read_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


_load_dotenv(BASE_DIR / ".env")

WORKBOOK_PATH = Path(os.getenv("MSCORE_WORKBOOK_PATH", str(DEFAULT_WORKBOOK_PATH)))
CACHE_PATH = Path(os.getenv("MSCORE_CACHE_PATH", str(BASE_DIR / "data" / "latest_snapshot.json")))
TIMEZONE = os.getenv("MSCORE_TIMEZONE", "America/Los_Angeles")

# Baseline background refresh cadence.
REFRESH_INTERVAL_MINUTES = max(5, _read_int("MSCORE_REFRESH_INTERVAL_MINUTES", 15))

# When live games exist, serve data with near real-time freshness.
LIVE_REFRESH_SECONDS = max(15, _read_int("MSCORE_LIVE_REFRESH_SECONDS", 30))

# Hard upper bound on cache age for normal requests.
CACHE_MAX_AGE_MINUTES = max(1, _read_int("MSCORE_CACHE_MAX_AGE_MINUTES", 1))

AUTO_REFRESH_ON_STARTUP = os.getenv("MSCORE_AUTO_REFRESH_ON_STARTUP", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

# Nightly model retrain schedule (local dashboard timezone).
NIGHTLY_RETRAIN_HOUR = min(23, max(0, _read_int("MSCORE_NIGHTLY_RETRAIN_HOUR", 0)))
NIGHTLY_RETRAIN_MINUTE = min(59, max(0, _read_int("MSCORE_NIGHTLY_RETRAIN_MINUTE", 20)))

# Daily archive schedule (typically after retrain).
NIGHTLY_ARCHIVE_HOUR = min(23, max(0, _read_int("MSCORE_NIGHTLY_ARCHIVE_HOUR", 0)))
NIGHTLY_ARCHIVE_MINUTE = min(59, max(0, _read_int("MSCORE_NIGHTLY_ARCHIVE_MINUTE", 35)))

# Prediction archive storage for true OOS tracking.
PREDICTION_ARCHIVE_DIR = Path(
    os.getenv("MSCORE_PREDICTION_ARCHIVE_DIR", str(CACHE_PATH.parent / "prediction_archive"))
)

# Daily snapshot archive for RRG/time-series analytics.
SNAPSHOT_ARCHIVE_DIR = Path(
    os.getenv("MSCORE_SNAPSHOT_ARCHIVE_DIR", str(CACHE_PATH.parent / "snapshot_archive"))
)
