# Frontend + API Usage

This document describes the local full-stack workflow for controlling and running the reproducible pipeline.

## Services

- Backend API: FastAPI app at `src/ai_dentistry/api/app.py`
- Frontend UI: React app at `apps/web`

## Start backend

```bash
cd "/Users/uday/AI in Dentistry"
source .venv/bin/activate
python scripts/run_api.py --host 127.0.0.1 --port 8000 --reload
```

## Start frontend

```bash
cd "/Users/uday/AI in Dentistry/apps/web"
npm install
npm run dev
```

Open `http://127.0.0.1:5173`.

## UI sections

1. Domain preset selector
2. Search strategy editor (add/remove query blocks)
3. Temporal binning controls with balanced-bin preview
4. Institution keyword editors (Dental/Medical/Technical)
5. Preprocessing controls
6. Funding controls (enable + US federal whitelist)
7. Run controls and live job logs
8. Artifact list

## Backend endpoints

- `GET /api/templates`
- `GET /api/protocol/current`
- `POST /api/protocol/validate`
- `POST /api/periods/preview`
- `POST /api/run`
- `GET /api/run/{job_id}`
- `GET /api/run/{job_id}/logs`
- `GET /api/run/{job_id}/artifacts`

## Job model

- Single active job at a time (queue size 1)
- Stages: `fetch`, `pipeline`, `descriptive_table`, `flow_figures`, `sankey_html`, `finalize`
- Logs written to `outputs/logs/job_<job_id>.log`
- Run manifests written to `outputs/runs/<timestamp>/run_manifest.json`
