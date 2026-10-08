"""Small dependency-light web server for the recorded-video analytics app."""
from __future__ import annotations

import json
import mimetypes
import os
import re
import shutil
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

from pipeline import (
    ROLE_LABEL,
    VIDEO_EXTENSIONS,
    analyze_video,
    build_cross_camera_candidates,
    write_json,
    write_summary_csv,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("RETAIL_ANALYTICS_DATA", str(BASE_DIR / "data"))).resolve()
JOBS_DIR = DATA_DIR / "jobs"
UPLOAD_LIMIT = int(os.environ.get("RETAIL_ANALYTICS_MAX_UPLOAD", str(1024 * 1024 * 1024)))
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8080"))

JOBS: dict = {}
JOBS_LOCK = threading.RLock()
ANALYSIS_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _job_dir(job_id: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{12}", job_id):
        raise ValueError("Invalid analysis ID")
    return JOBS_DIR / job_id


def _persist(job: dict) -> None:
    path = _job_dir(job["id"]) / "job.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(job, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def _load_jobs() -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    for path in JOBS_DIR.glob("*/job.json"):
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
            if job.get("status") in {"running", "queued"}:
                job["status"] = "failed"
                job["error"] = "The server restarted while this analysis was running. Start a new analysis to continue."
            JOBS[job["id"]] = job
        except Exception:
            continue


def _public_job(job: dict) -> dict:
    return {
        "id": job.get("id"),
        "status": job.get("status"),
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
        "progress": job.get("progress", 0),
        "stage": job.get("stage", ""),
        "videos": job.get("videos", {}),
        "summary": job.get("summary"),
        "video_results": job.get("video_results", {}),
        "outputs": job.get("outputs", []),
        "warnings": job.get("warnings", []),
        "error": job.get("error"),
    }


def _update_job(job_id: str, **values) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job.update(values)
        job["updated_at"] = _now()
        _persist(job)


def _video_offset(job: dict, role: str):
    value = (job.get("config") or {}).get("clip_start_offsets", {}).get(role)
    try:
        value = float(value)
        return value if value >= 0 else None
    except (TypeError, ValueError):
        return None


def _run_analysis(job_id: str) -> None:
    with ANALYSIS_LOCK:
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if not job:
                return
            job["status"] = "running"
            job["stage"] = "Preparing analysis"
            job["progress"] = 1
            job["updated_at"] = _now()
            _persist(job)
            snapshot = json.loads(json.dumps(job))
        folder = _job_dir(job_id)
        output_dir = folder / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        roles = [role for role in ("entrance", "store", "checkout") if role in snapshot.get("videos", {})]
        role_results = {}
        warnings = []
        try:
            for role_index, role in enumerate(roles):
                video_info = snapshot["videos"][role]
                video_path = folder / "uploads" / video_info["stored_name"]
                if not video_path.exists():
                    raise FileNotFoundError(f"Uploaded {ROLE_LABEL[role]} file is missing.")

                def report_progress(local_progress, stage, idx=role_index, count=len(roles)):
                    overall = 5 + ((idx + max(0.0, min(1.0, local_progress)) / 1.0) / max(count, 1)) * 85
                    _update_job(job_id, progress=int(min(92, overall)), stage=f"{ROLE_LABEL[role]} · {stage}")

                result = analyze_video(
                    str(video_path), role, str(output_dir),
                    snapshot.get("config", {}), progress=report_progress,
                )
                role_results[role] = result
                warnings.extend(result["report"].get("warnings", []))
                _update_job(job_id, progress=int(5 + ((role_index + 1) / max(len(roles), 1)) * 85),
                            stage=f"Completed {ROLE_LABEL[role]}")

            offsets = {role: _video_offset(snapshot, role) for role in roles}
            cross_candidates = build_cross_camera_candidates(role_results, offsets)
            camera_reports = {role: result["report"] for role, result in role_results.items()}
            local_people_sum = sum(r["summary"]["camera_local_visitors_estimated"] for r in camera_reports.values())
            summary = {
                "videos_analyzed": len(roles),
                "camera_local_visitors_sum_not_deduplicated": local_people_sum,
                "distinct_cross_camera_visitors": None,
                "entrance_entries": sum(r["summary"]["entry_events"] for r in camera_reports.values()),
                "entrance_exits": sum(r["summary"]["exit_events"] for r in camera_reports.values()),
                "groups_detected": sum(r["summary"]["distinct_group_memberships"] for r in camera_reports.values()),
                "group_intervals": sum(r["summary"]["group_intervals"] for r in camera_reports.values()),
                "peak_occupancy_max_per_camera": max((r["summary"]["peak_detected_occupancy"] for r in camera_reports.values()), default=0),
                "low_movement_intervals": sum(r["summary"]["low_movement_intervals"] for r in camera_reports.values()),
                "checkout_presence_events": sum(r["summary"]["checkout_camera_presence_events"] for r in camera_reports.values()),
                "payments_verified": 0,
                "cross_camera_review_candidates": len(cross_candidates),
            }
            global_report = {
                "schema_version": "2.0",
                "pipeline_version": "2.1.0",
                "analysis_id": job_id,
                "analysis_timestamp": _now(),
                "summary": summary,
                "camera_reports": camera_reports,
                "cross_camera_candidates": cross_candidates,
                "cross_camera_policy": {
                    "automatic_merge": False,
                    "start_offsets": offsets,
                    "note": "Candidate links are generated only when both clips have supplied offsets on the same timeline. They are based on time ordering and coarse clothing color; review is required.",
                },
                "limitations": [
                    "Visitor IDs are anonymous and camera-local. Counts across camera videos are not deduplicated.",
                    "Group labels describe temporary proximity/co-movement only, not family or personal relationships.",
                    "Checkout-camera presence does not prove a payment or purchase. No POS transaction feed was provided.",
                    "Confidence values are heuristic and have not been calibrated on your store footage.",
                    "Automatic staff, age, gender, intent, and theft classification are intentionally not performed.",
                ],
            }
            global_path = output_dir / "global_report.json"
            summary_path = output_dir / "summary.csv"
            write_json(str(global_path), global_report)
            write_summary_csv(str(summary_path), global_report)
            outputs = []
            video_results = {}
            for role, result in role_results.items():
                report = result["report"]
                video_results[role] = report["summary"] | {
                    "camera_label": report["camera_label"],
                    "warnings": report.get("warnings", []),
                    "video": report["video"],
                    "outputs": report.get("outputs", {}),
                }
                for name, label, kind in [
                    (report["outputs"]["annotated_video"], f"{ROLE_LABEL[role]} annotated video", "video"),
                    (report["outputs"]["tracklets_csv"], f"{ROLE_LABEL[role]} tracklets CSV", "csv"),
                ]:
                    outputs.append({"name": name, "label": label, "type": kind, "role": role,
                                    "href": f"/api/jobs/{job_id}/outputs/{quote(name)}"})
            outputs.extend([
                {"name": "global_report.json", "label": "Full JSON report", "type": "json", "href": f"/api/jobs/{job_id}/outputs/global_report.json"},
                {"name": "summary.csv", "label": "Summary CSV", "type": "csv", "href": f"/api/jobs/{job_id}/outputs/summary.csv"},
            ])
            _update_job(
                job_id,
                status="complete",
                progress=100,
                stage="Analysis complete",
                summary=summary,
                video_results=video_results,
                outputs=outputs,
                warnings=list(dict.fromkeys(warnings)),
                error=None,
                report_file="global_report.json",
            )
        except Exception as exc:
            (folder / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
            _update_job(job_id, status="failed", stage="Analysis failed", error=str(exc), progress=0)


class Handler(BaseHTTPRequestHandler):
    server_version = "RetailFlow/2.0"

    def log_message(self, format, *args):
        # Keep the preview/server log concise; errors remain visible.
        if args and str(args[1]).startswith("5"):
            super().log_message(format, *args)

    def _json(self, value, status=200):
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body_json(self):
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length > 2 * 1024 * 1024:
            raise ValueError("JSON request is too large.")
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8"))

    def _not_found(self, message="Not found"):
        self._json({"error": message}, 404)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Content-Length")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path == "/api/health":
                self._json({"ok": True, "service": "retail-video-analytics", "version": "2.0.0"})
                return
            if path == "/api/jobs":
                with JOBS_LOCK:
                    jobs = sorted(JOBS.values(), key=lambda j: j.get("created_at", ""), reverse=True)[:20]
                    self._json({"jobs": [_public_job(job) for job in jobs]})
                return
            if path == "/":
                self._serve_static(BASE_DIR / "static" / "index.html")
                return
            parts = [unquote(p) for p in path.strip("/").split("/")]
            if len(parts) == 3 and parts[0:2] == ["api", "jobs"]:
                with JOBS_LOCK:
                    job = JOBS.get(parts[2])
                    if not job:
                        self._not_found("Analysis not found")
                    else:
                        self._json(_public_job(job))
                return
            if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "report":
                job_id = parts[2]
                folder = _job_dir(job_id)
                report_path = folder / "outputs" / "global_report.json"
                if not report_path.exists():
                    self._not_found("The report is not ready yet")
                    return
                self._serve_file(report_path, inline=False)
                return
            if len(parts) == 5 and parts[:2] == ["api", "jobs"] and parts[3] == "outputs":
                job_id, name = parts[2], Path(parts[4]).name
                target = _job_dir(job_id) / "outputs" / name
                if not target.is_file():
                    self._not_found("Output is not ready or does not exist")
                    return
                self._serve_file(target, inline=target.suffix.lower() in {".mp4", ".webm"})
                return
            if path.startswith("/static/"):
                name = Path(unquote(path[len("/static/"):])).name
                target = BASE_DIR / "static" / name
                if target.is_file():
                    self._serve_static(target)
                else:
                    self._not_found()
                return
            self._not_found()
        except ValueError as exc:
            self._json({"error": str(exc)}, 400)
        except Exception as exc:
            self._json({"error": str(exc)}, 500)

    def do_POST(self):
        parsed = urlparse(self.path)
        parts = [unquote(p) for p in parsed.path.strip("/").split("/")]
        try:
            if parsed.path == "/api/jobs":
                body = self._body_json()
                job_id = uuid.uuid4().hex[:12]
                folder = _job_dir(job_id)
                (folder / "uploads").mkdir(parents=True, exist_ok=True)
                (folder / "outputs").mkdir(parents=True, exist_ok=True)
                config = body.get("config") if isinstance(body.get("config"), dict) else {}
                job = {
                    "id": job_id,
                    "created_at": _now(),
                    "updated_at": _now(),
                    "status": "draft",
                    "progress": 0,
                    "stage": "Choose at least one video",
                    "videos": {},
                    "config": config,
                    "summary": None,
                    "video_results": {},
                    "outputs": [],
                    "warnings": [],
                    "error": None,
                }
                with JOBS_LOCK:
                    JOBS[job_id] = job
                    _persist(job)
                self._json(_public_job(job), 201)
                return
            if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "analyze":
                job_id = parts[2]
                with JOBS_LOCK:
                    job = JOBS.get(job_id)
                    if not job:
                        self._not_found("Analysis not found")
                        return
                    if not job.get("videos"):
                        self._json({"error": "Upload at least one video first."}, 400)
                        return
                    if job.get("status") in {"queued", "running"}:
                        self._json({"error": "This analysis is already running."}, 409)
                        return
                    job["status"] = "queued"
                    job["progress"] = 1
                    job["stage"] = "Queued for analysis"
                    job["error"] = None
                    job["updated_at"] = _now()
                    _persist(job)
                threading.Thread(target=_run_analysis, args=(job_id,), daemon=True,
                                 name=f"analysis-{job_id}").start()
                self._json(_public_job(job), 202)
                return
            self._not_found()
        except (ValueError, json.JSONDecodeError) as exc:
            self._json({"error": str(exc)}, 400)
        except Exception as exc:
            self._json({"error": str(exc)}, 500)

    def do_PUT(self):
        parsed = urlparse(self.path)
        parts = [unquote(p) for p in parsed.path.strip("/").split("/")]
        if not (len(parts) == 5 and parts[:2] == ["api", "jobs"] and parts[3] == "videos"):
            self._not_found()
            return
        job_id, role = parts[2], parts[4]
        if role == "purchase":
            role = "checkout"
        if role not in {"entrance", "store", "checkout"}:
            self._json({"error": "Video role must be entrance, store, checkout, or purchase (an alias for checkout)."}, 400)
            return
        try:
            length = int(self.headers.get("Content-Length", "0") or 0)
            if length <= 0:
                self._json({"error": "The uploaded file is empty."}, 400)
                return
            if length > UPLOAD_LIMIT:
                self._json({"error": f"File is larger than the {UPLOAD_LIMIT // (1024 * 1024)} MB upload limit."}, 413)
                return
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                if not job:
                    self._not_found("Analysis not found")
                    return
                if job.get("status") in {"queued", "running"}:
                    self._json({"error": "Videos cannot be changed while analysis is running."}, 409)
                    return
            query = parse_qs(parsed.query)
            original = Path(unquote(query.get("filename", [f"{role}.mp4"])[0])).name
            suffix = Path(original).suffix.lower()
            if suffix not in VIDEO_EXTENSIONS:
                self._json({"error": "Unsupported video type. Use MP4, MOV, AVI, MKV, WEBM, or M4V."}, 415)
                return
            safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(original).stem)[:80] or role
            stored_name = f"{role}_{safe_stem}{suffix}"
            folder = _job_dir(job_id) / "uploads"
            folder.mkdir(parents=True, exist_ok=True)
            destination = folder / stored_name
            temporary = destination.with_suffix(destination.suffix + ".part")
            remaining = length
            with temporary.open("wb") as out:
                while remaining:
                    chunk = self.rfile.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise IOError("Upload ended before all bytes were received.")
                    out.write(chunk)
                    remaining -= len(chunk)
            temporary.replace(destination)
            with JOBS_LOCK:
                job = JOBS[job_id]
                job.setdefault("videos", {})[role] = {
                    "filename": original,
                    "stored_name": stored_name,
                    "size_bytes": length,
                    "uploaded_at": _now(),
                }
                job["status"] = "draft"
                job["progress"] = 0
                job["stage"] = f"{ROLE_LABEL[role]} video ready"
                job["updated_at"] = _now()
                _persist(job)
                response = _public_job(job)
            self._json(response, 201)
        except ValueError as exc:
            self._json({"error": str(exc)}, 400)
        except Exception as exc:
            self._json({"error": str(exc)}, 500)

    def do_DELETE(self):
        parts = [unquote(p) for p in urlparse(self.path).path.strip("/").split("/")]
        if len(parts) != 3 or parts[:2] != ["api", "jobs"]:
            self._not_found()
            return
        job_id = parts[2]
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if not job:
                self._not_found("Analysis not found")
                return
            if job.get("status") in {"queued", "running"}:
                self._json({"error": "Cannot delete a running analysis."}, 409)
                return
            JOBS.pop(job_id, None)
        shutil.rmtree(_job_dir(job_id), ignore_errors=True)
        self._json({"ok": True})

    def _serve_static(self, path: Path):
        if not path.is_file():
            self._not_found()
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in {"application/javascript", "application/json"}:
            content_type += "; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, path: Path, inline: bool = False):
        size = path.stat().st_size
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        range_header = self.headers.get("Range")
        start, end, status = 0, size - 1, 200
        if range_header and range_header.startswith("bytes="):
            try:
                first, last = range_header[6:].split("-", 1)
                if first:
                    start = int(first)
                if last:
                    end = min(end, int(last))
                if start > end or start >= size:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                status = 206
            except (ValueError, TypeError):
                start, end, status = 0, size - 1, 200
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(max(0, end - start + 1)))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        disposition = "inline" if inline else "attachment"
        self.send_header("Content-Disposition", f"{disposition}; filename*=UTF-8''{quote(path.name)}")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with path.open("rb") as file:
            file.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                data = file.read(min(1024 * 1024, remaining))
                if not data:
                    break
                self.wfile.write(data)
                remaining -= len(data)


def main():
    _load_jobs()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server.daemon_threads = True
    print(f"Retail Flow Analytics listening on http://{HOST}:{PORT}")
    print(f"Job data directory: {DATA_DIR}")
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("Shutting down Retail Flow Analytics.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
