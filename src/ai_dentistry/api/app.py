from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import traceback
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import yaml

from ai_dentistry.affiliations import approximate_publication_date
from ai_dentistry.config import build_balanced_periods, load_protocol, validate_protocol_dict
from ai_dentistry.pipeline import _load_all_raw_records, run_pipeline
from ai_dentistry.pubmed import fetch_query_snapshot


# Allow isolating runs in a separate workspace folder when launching the API.
# Useful for parallel projects that should not share outputs.
CODE_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(os.getenv("AI_DENTISTRY_PROJECT_ROOT", CODE_ROOT)).resolve()
DEFAULT_PROTOCOL_PATH = PROJECT_ROOT / "config" / "protocol.yaml"
DEFAULT_RAW_GLOB = "data/raw/pubmed_records_*.jsonl"


class ProtocolValidateRequest(BaseModel):
    protocol: dict[str, Any]


class PeriodPreviewRequest(BaseModel):
    raw_glob: str = DEFAULT_RAW_GLOB
    n_bins: int = 6
    min_year: int | None = 1946
    max_year: int | None = 2025
    min_years_per_bin: int = 1
    use_observed_year_range: bool = True


class RunRequest(BaseModel):
    protocol: dict[str, Any] | None = None
    protocol_path: str = "config/protocol.yaml"
    raw_glob: str = DEFAULT_RAW_GLOB
    fetch_first: bool = False
    fetch_output_dir: str = "data/raw"
    clean_output: bool = False
    period_mode: str | None = Field(default="dynamic")
    with_funding: bool = True
    funding_policy: str | None = None
    generate_descriptive_table: bool = True
    generate_flow_figures: bool = True
    generate_sankey_html: bool = True


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    created_at: str
    updated_at: str
    stage: str
    error: str | None = None
    result: dict[str, Any] | None = None
    log_path: str


@dataclass
class JobState:
    job_id: str
    request: RunRequest
    created_at: datetime
    updated_at: datetime
    status: str = "queued"
    stage: str = "queued"
    error: str | None = None
    result: dict[str, Any] | None = None
    log_path: Path | None = None
    artifacts: list[str] = field(default_factory=list)

    def as_response(self) -> JobStatusResponse:
        return JobStatusResponse(
            job_id=self.job_id,
            status=self.status,
            created_at=self.created_at.isoformat(),
            updated_at=self.updated_at.isoformat(),
            stage=self.stage,
            error=self.error,
            result=self.result,
            log_path=str(self.log_path) if self.log_path else "",
        )


_jobs: dict[str, JobState] = {}
_jobs_lock = threading.Lock()
_active_job_id: str | None = None


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_resolve(path_like: str) -> Path:
    path = Path(path_like)
    resolved = path if path.is_absolute() else (PROJECT_ROOT / path)
    resolved = resolved.resolve()
    if PROJECT_ROOT not in resolved.parents and resolved != PROJECT_ROOT:
        raise HTTPException(status_code=400, detail=f"Path escapes project root: {path_like}")
    return resolved


def _relative_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except Exception:
        return str(path)


def _artifact_kind(path: Path) -> str:
    rel = _relative_path(path)
    suffix = path.suffix.lower()
    if rel.startswith("outputs/tables/"):
        return "table"
    if rel.startswith("outputs/figures/"):
        if suffix == ".html":
            return "figure_html"
        if suffix == ".png":
            return "figure_png"
        if suffix == ".pdf":
            return "figure_pdf"
        return "figure_other"
    if rel.startswith("outputs/runs/"):
        return "run_manifest"
    return "other"


def _load_local_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def _write_log(job: JobState, message: str) -> None:
    if not job.log_path:
        return
    stamp = _now().strftime("%Y-%m-%d %H:%M:%S UTC")
    with job.log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{stamp}] {message}\n")


def _collect_artifacts() -> list[str]:
    candidates: list[Path] = []
    for pattern in [
        "outputs/tables/*.csv",
        "outputs/tables/*.md",
        "outputs/figures/flow_options/*.html",
        "outputs/figures/flow_options/*.png",
        "outputs/figures/flow_options/*.pdf",
        "outputs/figures/flow_options/funding/*.html",
        "outputs/runs/*/run_manifest.json",
    ]:
        candidates.extend(sorted(PROJECT_ROOT.glob(pattern)))
    return [str(path.resolve()) for path in candidates]


def _collect_artifact_items() -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for path_str in _collect_artifacts():
        path = Path(path_str).resolve()
        items.append(
            {
                "path": str(path),
                "relative_path": _relative_path(path),
                "name": path.name,
                "kind": _artifact_kind(path),
            }
        )
    return items


def _read_descriptive_table(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        rows = [dict(row) for row in reader]
    return columns, rows


def _run_subprocess(job: JobState, cmd: list[str], stage: str) -> None:
    _write_log(job, f"Running {stage}: {' '.join(cmd)}")
    completed = subprocess.run(
        cmd,
        cwd=CODE_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.stdout:
        _write_log(job, completed.stdout.strip())
    if completed.stderr:
        _write_log(job, completed.stderr.strip())
    if completed.returncode != 0:
        raise RuntimeError(f"{stage} failed with return code {completed.returncode}")


def _execute_job(job_id: str) -> None:
    global _active_job_id
    with _jobs_lock:
        job = _jobs[job_id]
        job.status = "running"
        job.stage = "initializing"
        job.updated_at = _now()

    try:
        request = job.request
        _load_local_env_file(PROJECT_ROOT / ".env")
        log_dir = PROJECT_ROOT / "outputs" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        job.log_path = log_dir / f"job_{job_id}.log"
        _write_log(job, "Job started.")

        runtime_protocol_path = _safe_resolve(request.protocol_path)
        if request.protocol is not None:
            validate_protocol_dict(request.protocol)
            runtime_protocol_path = PROJECT_ROOT / "outputs" / "runs" / job_id / "runtime_protocol.yaml"
            runtime_protocol_path.parent.mkdir(parents=True, exist_ok=True)
            runtime_protocol_path.write_text(
                yaml.safe_dump(request.protocol, sort_keys=False),
                encoding="utf-8",
            )
            _write_log(job, f"Using runtime protocol snapshot: {runtime_protocol_path}")

        if request.fetch_first:
            job.stage = "fetch"
            job.updated_at = _now()
            _write_log(job, "Fetching PubMed snapshots.")
            protocol = load_protocol(runtime_protocol_path)
            output_dir = _safe_resolve(request.fetch_output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            if request.clean_output:
                for existing in output_dir.glob("pubmed_records_*.jsonl"):
                    existing.unlink()
            for query in protocol["queries"]:
                out_path = fetch_query_snapshot(
                    query_id=query["id"],
                    query_text=query["query"],
                    entrez_cfg=protocol.get("entrez", {}),
                    output_dir=output_dir,
                )
                _write_log(job, f"Wrote snapshot: {out_path}")

        job.stage = "pipeline"
        job.updated_at = _now()
        _write_log(job, "Running core pipeline.")
        pipeline_summary = run_pipeline(
            protocol_path=runtime_protocol_path,
            raw_glob=request.raw_glob,
            project_root=PROJECT_ROOT,
            period_mode_override=request.period_mode,
            clean_output_dirs=request.clean_output,
            with_funding=request.with_funding,
            funding_policy_override=request.funding_policy,
        )
        _write_log(job, f"Pipeline summary: {json.dumps(pipeline_summary)}")

        if request.generate_descriptive_table:
            job.stage = "descriptive_table"
            job.updated_at = _now()
            _run_subprocess(
                job,
                [
                    sys.executable,
                    str(CODE_ROOT / "scripts" / "generate_descriptive_table.py"),
                    "--project-root",
                    str(PROJECT_ROOT),
                ],
                stage="generate_descriptive_table",
            )

        if request.generate_flow_figures:
            job.stage = "flow_figures"
            job.updated_at = _now()
            _run_subprocess(
                job,
                [
                    sys.executable,
                    str(CODE_ROOT / "scripts" / "generate_flow_prototypes.py"),
                    "--project-root",
                    str(PROJECT_ROOT),
                    "--output-dir",
                    str(PROJECT_ROOT / "outputs" / "figures" / "flow_options"),
                ],
                stage="generate_flow_prototypes",
            )

        if request.generate_sankey_html:
            job.stage = "sankey_html"
            job.updated_at = _now()
            _run_subprocess(
                job,
                [
                    sys.executable,
                    str(CODE_ROOT / "scripts" / "generate_interactive_sankey.py"),
                    "--project-root",
                    str(PROJECT_ROOT),
                    "--mode",
                    "role",
                    "--output-html",
                    str(PROJECT_ROOT / "outputs" / "figures" / "flow_options" / "role_flow_sankey.html"),
                    "--write-field-small-multiples",
                    "--field-small-multiples-html",
                    str(
                        PROJECT_ROOT / "outputs" / "figures" / "flow_options" / "role_flow_sankey_fields.html"
                    ),
                    "--write-playable-timeline",
                    "--save-data-csv",
                ],
                stage="generate_interactive_sankey",
            )

        job.stage = "finalize"
        job.updated_at = _now()
        job.status = "completed"
        job.result = {"pipeline_summary": pipeline_summary}
        job.artifacts = _collect_artifacts()
        _write_log(job, "Job completed.")
    except Exception as exc:  # noqa: BLE001
        job.status = "failed"
        job.stage = "failed"
        job.error = f"{exc}"
        job.result = {"traceback": traceback.format_exc()}
        _write_log(job, f"Job failed: {exc}")
        _write_log(job, traceback.format_exc())
    finally:
        job.updated_at = _now()
        with _jobs_lock:
            _active_job_id = None


def _start_job(request: RunRequest) -> JobState:
    global _active_job_id
    with _jobs_lock:
        if _active_job_id is not None and _active_job_id in _jobs:
            active = _jobs[_active_job_id]
            if active.status in {"queued", "running"}:
                raise HTTPException(
                    status_code=409,
                    detail=f"Another job is already active: {_active_job_id}",
                )
        job_id = uuid.uuid4().hex[:12]
        job = JobState(
            job_id=job_id,
            request=request,
            created_at=_now(),
            updated_at=_now(),
        )
        _jobs[job_id] = job
        _active_job_id = job_id

    thread = threading.Thread(target=_execute_job, args=(job_id,), daemon=True)
    thread.start()
    return job


def create_app() -> FastAPI:
    app = FastAPI(title="AI in Dentistry Reproducibility API", version="0.2.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/templates")
    def get_templates() -> dict[str, Any]:
        protocol = load_protocol(DEFAULT_PROTOCOL_PATH)
        custom = json.loads(json.dumps(protocol))
        custom["queries"] = [
            {
                "id": "custom_query_1",
                "query": "(artificial intelligence) AND (dentistry)",
            }
        ]
        return {
            "default_template_id": "dental_default",
            "templates": [
                {
                    "id": "dental_default",
                    "name": "Dental Default",
                    "description": "Current AI-in-dentistry reproducibility protocol.",
                    "protocol": protocol,
                },
                {
                    "id": "custom",
                    "name": "Custom Domain",
                    "description": "Start from editable custom query and settings.",
                    "protocol": custom,
                },
            ],
        }

    @app.get("/api/protocol/current")
    def get_protocol_current() -> dict[str, Any]:
        protocol = load_protocol(DEFAULT_PROTOCOL_PATH)
        return {"protocol": protocol, "protocol_path": str(DEFAULT_PROTOCOL_PATH)}

    @app.post("/api/protocol/validate")
    def validate_protocol_endpoint(payload: ProtocolValidateRequest) -> dict[str, Any]:
        try:
            validate_protocol_dict(payload.protocol)
        except Exception as exc:  # noqa: BLE001
            return {"valid": False, "error": str(exc)}
        return {"valid": True}

    @app.post("/api/periods/preview")
    def preview_periods(payload: PeriodPreviewRequest) -> dict[str, Any]:
        records = _load_all_raw_records(payload.raw_glob, project_root=PROJECT_ROOT)
        years: list[int] = []
        for record in records:
            pub_date = approximate_publication_date(record.get("DP"))
            if pd_is_na(pub_date):
                continue
            years.append(int(pub_date.year))
        periods = build_balanced_periods(
            publication_years=years,
            n_bins=payload.n_bins,
            min_year=payload.min_year,
            max_year=payload.max_year,
            min_years_per_bin=payload.min_years_per_bin,
            use_observed_year_range=payload.use_observed_year_range,
        )
        rows: list[dict[str, Any]] = []
        for period in periods:
            count = sum(1 for y in years if period.start_year <= y <= period.end_year)
            rows.append(
                {
                    "label": period.label,
                    "start_year": period.start_year,
                    "end_year": period.end_year,
                    "publication_count": count,
                }
            )
        return {"periods": rows, "total_publications": len(years)}

    @app.post("/api/run")
    def start_run(payload: RunRequest) -> dict[str, Any]:
        _safe_resolve(payload.protocol_path)
        job = _start_job(payload)
        return {"job_id": job.job_id, "status": job.status}

    @app.get("/api/run/{job_id}", response_model=JobStatusResponse)
    def get_run(job_id: str) -> JobStatusResponse:
        if job_id not in _jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        return _jobs[job_id].as_response()

    @app.get("/api/run/{job_id}/logs")
    def get_run_logs(job_id: str) -> dict[str, Any]:
        if job_id not in _jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        job = _jobs[job_id]
        if not job.log_path or not job.log_path.exists():
            return {"job_id": job_id, "log": ""}
        return {"job_id": job_id, "log": job.log_path.read_text(encoding="utf-8")}

    @app.get("/api/run/{job_id}/artifacts")
    def get_run_artifacts(job_id: str) -> dict[str, Any]:
        if job_id not in _jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        job = _jobs[job_id]
        artifacts = job.artifacts or _collect_artifacts()
        artifact_items = _collect_artifact_items()
        return {"job_id": job_id, "artifacts": artifacts, "artifact_items": artifact_items}

    @app.get("/api/run/{job_id}/descriptive-table")
    def get_run_descriptive_table(job_id: str) -> dict[str, Any]:
        if job_id not in _jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        csv_path = PROJECT_ROOT / "outputs" / "tables" / "table_descriptive_by_period.csv"
        md_path = PROJECT_ROOT / "outputs" / "tables" / "table_descriptive_by_period.md"
        if not csv_path.exists():
            return {
                "job_id": job_id,
                "available": False,
                "csv_path": str(csv_path),
                "md_path": str(md_path),
                "columns": [],
                "rows": [],
            }
        columns, rows = _read_descriptive_table(csv_path)
        return {
            "job_id": job_id,
            "available": True,
            "csv_path": str(csv_path.resolve()),
            "md_path": str(md_path.resolve()),
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
        }

    @app.get("/api/resource")
    def get_resource(path: str) -> FileResponse:
        resolved = _safe_resolve(path)
        if not resolved.exists() or not resolved.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(path=str(resolved), filename=resolved.name)

    return app


def pd_is_na(value: Any) -> bool:
    try:
        return bool(value != value)
    except Exception:
        return False


app = create_app()
