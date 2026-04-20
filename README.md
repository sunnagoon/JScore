# MLB Mscore Live Dashboard

A FastAPI dashboard that combines your workbook rankings with live MLB stats and game status.

## What It Does

- Reads the `Mscore` sheet from your workbook (`2026 MLB Mscore_041326.xlsx` by default).
- Pulls live standings and daily game states from MLB Stats API.
- Builds a merged snapshot with:
  - workbook rank + mscore
  - live wins/losses, win%, run differential, streak
  - `live_power_score` blend
  - simple projection metrics (win% and 162-game pace)
- Stores cache at `data/latest_snapshot.json`.
- Auto-refreshes on a scheduler (`15` minutes by default), with one-minute freshness checks when live games are active.
- Serves an aesthetic responsive dashboard UI.

## Run

```powershell
Set-Location "T:\Lab Stuff 4U\misc\M_SCORE"
python -m pip install --user -r requirements.txt
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8787
```

Open: `http://127.0.0.1:8787`

## API

- `GET /api/dashboard` - current snapshot (uses cache, auto-refreshes if stale)
- `POST /api/refresh` - force refresh now
- `GET /api/health` - scheduler and refresh metadata

## Config (Environment Variables)

- `MSCORE_WORKBOOK_PATH` default: `..\2026 MLB Mscore_041326.xlsx`
- `MSCORE_CACHE_PATH` default: `data\latest_snapshot.json`
- `MSCORE_TIMEZONE` default: `America/Los_Angeles`
- `MSCORE_REFRESH_INTERVAL_MINUTES` default: `15`
- `MSCORE_CACHE_MAX_AGE_MINUTES` default: `15`
- `MSCORE_LIVE_REFRESH_SECONDS` default: `60`
- `MSCORE_AUTO_REFRESH_ON_STARTUP` default: `1`
- `MSCORE_ODDS_API_KEY` default: empty (required for true market odds in Upset/Value Finder)

### Market Odds Setup (Upset/Value Finder)

Set your odds key in PowerShell before launching the server:

```powershell
$env:MSCORE_ODDS_API_KEY = "<your_odds_api_key>"
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8787
```

If this key is missing, the dashboard runs in model-only mode and market edge/upset signals will be unavailable.

## Next Feature Hooks

- Add team detail drilldowns (rolling trend charts by metric).
- Replace baseline projection with game-level predictive model.
- Add historical snapshot archive for backtesting.
- Add authentication + deploy to cloud host for always-on updates.

