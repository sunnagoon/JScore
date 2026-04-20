# MLB M_SCORE Dashboard

API-first MLB analytics dashboard built with FastAPI.

It provides:
- Real-time standings and game status updates
- Team ranking scores (Mscore, Power, D+, Joe v2)
- Daily matchup win probabilities with context adjustments
- Value/upset lens with optional market odds blend
- Model diagnostics, calibration, and archive tracking
- RRG momentum visualization and PNG exports

## What The App Does

### 1) Live Team Rankings
- Pulls MLB team standings and team-level stat feeds
- Computes and ranks teams using:
  - `Mscore`: multi-bucket team strength blend
  - `Power`: context-adjusted score built on top of Mscore
  - `D+`: defense/pitching-weighted alternative score
  - `Joe`: xwOBAcon + xFIP based score (v2 with shrinkage + uncertainty)
  - `Power-D+`: context gap signal

### 2) Prediction Lens (Daily Matchups)
- Builds game-level win probabilities for today
- Applies context adjustments (starter, bullpen, travel, lineup, luck, advanced quality)
- Optionally blends with market odds when `MSCORE_ODDS_API_KEY` is configured
- Adds uncertainty bands and uncertainty levels
- Includes per-matchup **SP Compare** with starter side-by-side metrics and best-stat highlighting

### 3) Value / Upset Finder
- Compares model edge vs market implied probabilities
- Flags potential upset/value spots
- Reduces edge aggressiveness under high uncertainty

### 4) Diagnostics + Monitoring
- Reliability curve and calibration summaries
- Brier decomposition components
- Feature importance and ablation views
- Daily prediction archive support for out-of-sample tracking

### 5) RRG and Historical Views
- Relative Rotation Graph (RRG) across multiple score metrics
- Team selection, trails, viewport controls, export PNG
- Snapshot archive backfill endpoint for historical RRG depth

## Data Sources

- **MLB Stats API** (primary live data source)
- **Odds API** (optional, for market blend/value lens)

This app is currently API-first. The workbook path setting is retained for compatibility/metadata, but live ranking data is generated from API feeds.

## Run

```powershell
Set-Location "T:\Lab Stuff 4U\misc\M_SCORE"
python -m pip install --user -r requirements.txt
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8787
```

Open:
- `http://127.0.0.1:8787`

## Core API Endpoints

### Dashboard + Health
- `GET /` - UI entrypoint
- `GET /api/dashboard` - current snapshot (auto-refreshes stale cache)
- `POST /api/refresh` - force rebuild snapshot now
- `GET /api/health` - scheduler/cache/odds status

### Backtests + Model Ops
- `GET /api/backtest/last-season`
- `GET /api/backtest/compare`
- `POST /api/backtest/nightly-run`

### Predictions + Archive
- `GET /api/predictions/today`
- `POST /api/predictions/archive/daily`
- `GET /api/predictions/archive?date=YYYY-MM-DD`
- `GET /api/predictions/archive/index?limit=30`

### RRG + Snapshot Archive
- `GET /api/rrg`
- `POST /api/archives/snapshots/backfill`

## Environment Variables

### Core
- `MSCORE_WORKBOOK_PATH`
  - Default: `..\2026 MLB Mscore_041326.xlsx`
  - Legacy compatibility path (app runs API-first)
- `MSCORE_CACHE_PATH`
  - Default: `data\latest_snapshot.json`
- `MSCORE_TIMEZONE`
  - Default: `America/Los_Angeles`

### Refresh Cadence
- `MSCORE_REFRESH_INTERVAL_MINUTES`
  - Default: `15` (minimum enforced: `5`)
- `MSCORE_LIVE_REFRESH_SECONDS`
  - Default: `30` (minimum enforced: `15`)
- `MSCORE_CACHE_MAX_AGE_MINUTES`
  - Default: `1` (minimum enforced: `1`)
- `MSCORE_AUTO_REFRESH_ON_STARTUP`
  - Default: `1` (`true/yes/on` also accepted)

### Nightly Jobs
- `MSCORE_NIGHTLY_RETRAIN_HOUR` (default `0`)
- `MSCORE_NIGHTLY_RETRAIN_MINUTE` (default `20`)
- `MSCORE_NIGHTLY_ARCHIVE_HOUR` (default `0`)
- `MSCORE_NIGHTLY_ARCHIVE_MINUTE` (default `35`)

### Archive Paths
- `MSCORE_PREDICTION_ARCHIVE_DIR`
  - Default: `data/prediction_archive`
- `MSCORE_SNAPSHOT_ARCHIVE_DIR`
  - Default: `data/snapshot_archive`

### Market Odds
- `MSCORE_ODDS_API_KEY`
  - Default: empty
  - Required for true market odds in Value/Upset and market blend

Example (PowerShell):
```powershell
$env:MSCORE_ODDS_API_KEY = "<your_odds_api_key>"
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8787
```

## Score Cheat Sheet

- `Mscore`
  - Base + offense + defense + context buckets mapped to a stable 0-100 scale
- `Power`
  - Mscore with context/risk/form/talent overlays and confidence weighting
- `D+`
  - Defense/pitching-emphasized score with stronger run-prevention weighting
- `Joe (v2)`
  - xwOBAcon + inverse xFIP composite with:
    - data-driven internal weighting
    - sample-size shrinkage toward neutral baseline
    - confidence and uncertainty bands

## Generated Local Artifacts

Typical runtime files in `data/`:
- `latest_snapshot.json`
- `prediction_archive/*.json`
- `snapshot_archive/*.json`
- `backtest_report_*.json`
- `backtest_model_*.joblib`

These are generated artifacts, not source code.

## Troubleshooting

### Dashboard loads but market rows are missing
- Ensure `MSCORE_ODDS_API_KEY` is set in the same shell used to start Uvicorn
- Check `GET /api/health`:
  - `odds_api_key_configured`
  - `market_status`

### Old frontend behavior after updates
- Hard refresh browser (`Ctrl+F5`)
- Restart server if needed

### No matchup starter metrics available
- Probable pitcher feed can be incomplete earlier in the day
- SP Compare panel will show available fields only

## Project Structure

- `app/main.py` - FastAPI app + routes + scheduler wiring
- `app/config.py` - environment configuration defaults
- `app/services/mlb_service.py` - MLB API pulls and parsing
- `app/services/prediction_service.py` - matchup model and adjustments
- `app/services/snapshot_service.py` - team ranking snapshot build + RRG
- `app/static/` - frontend UI (`index.html`, `app.js`, `styles.css`)
- `data/` - generated cache/archive artifacts

## License / Usage

Internal analytics project. Add your preferred license if you plan public redistribution.
