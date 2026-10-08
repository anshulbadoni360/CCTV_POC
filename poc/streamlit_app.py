from __future__ import annotations

import os
import shutil
import threading
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from pipeline import (
    ROLE_LABEL,
    analyze_video,
    build_cross_camera_candidates,
    write_json,
    write_summary_csv,
)


st.set_page_config(
    page_title="Retail Flow · Video Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("RETAIL_ANALYTICS_DATA", str(ROOT / "data" / "streamlit_jobs"))).resolve()
DATA_ROOT.mkdir(parents=True, exist_ok=True)
ROLES = ["entrance", "store", "checkout"]
ROLE_TITLES = {
    "entrance": "Entrance",
    "store": "Store floor",
    "checkout": "Checkout / purchase camera",
}
VIDEO_TYPES = ["mp4", "mov", "avi", "mkv", "webm", "m4v"]


st.markdown("""
<style>
:root { --rf-ink:#ece9e3; --rf-accent:#dfa85e; --rf-muted:#a0a6ad; --rf-line:#30363d; --rf-panel:#181c20; --rf-bg:#101215; }
.stApp { background:var(--rf-bg); color:var(--rf-ink); }
[data-testid="stHeader"] { background:rgba(16,18,21,.94); }
.block-container { padding-top:1.35rem; padding-bottom:3rem; max-width:1450px; }
[data-testid="stSidebar"] { background:#171a1e; border-right:1px solid #2b3036; }
[data-testid="stSidebar"] * { color:#e7e4de; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color:#9da4ac; }
[data-testid="stSidebar"] [data-baseweb="select"] > div { background:#22262b; border-color:#424950; }
[data-testid="stSidebar"] input { color:#ece9e3 !important; background:#22262b !important; }
[data-testid="stFileUploader"] section { background:#171b1f; border:1px dashed #444b53; border-radius:13px; }
[data-testid="stFileUploader"] section:hover { border-color:var(--rf-accent); }
[data-testid="stMetric"] { background:var(--rf-panel); padding:15px 17px; border:1px solid var(--rf-line); border-radius:14px; }
[data-testid="stMetricLabel"] { color:#a3a9b0; }
[data-testid="stMetricValue"] { color:#f3f0e9; }
[data-testid="stExpander"] { background:var(--rf-panel); border:1px solid var(--rf-line); border-radius:12px; }
.stButton > button[kind="primary"] { background:var(--rf-accent); color:#17191c; border:0; border-radius:10px; min-height:45px; font-weight:750; }
.stButton > button[kind="primary"]:hover { background:#edbb78; color:#131518; }
.stButton > button:not([kind="primary"]), .stDownloadButton > button { background:#202429; color:#e9e6df; border:1px solid #3a4148; border-radius:9px; }
.stButton > button:not([kind="primary"]):hover, .stDownloadButton > button:hover { border-color:var(--rf-accent); color:#fff; }
.rf-eyebrow { text-transform:uppercase; letter-spacing:.16em; color:var(--rf-accent); font-size:10px; font-weight:800; }
.rf-hero { font-size:clamp(35px,4.6vw,58px); line-height:1.03; letter-spacing:-.055em; font-weight:730; margin:.45rem 0 .75rem; color:#f0ede7; }
.rf-sub { color:#a5abb2; font-size:14px; max-width:780px; line-height:1.65; }
.rf-card { background:var(--rf-panel); border:1px solid var(--rf-line); border-radius:14px; padding:16px; color:#e9e6df; }
.rf-note { border-left:2px solid var(--rf-accent); padding:9px 13px; color:#a8adb3; font-size:12px; background:var(--rf-panel); border-radius:0 9px 9px 0; }
[data-testid="stAlert"] { background:#201e1a; border-color:#554633; color:#e9dfce; }
[data-testid="stDataFrame"] { border:1px solid var(--rf-line); border-radius:10px; overflow:hidden; }
hr { border-color:#30363d; }
</style>
""", unsafe_allow_html=True)


def _format_time(value):
    try:
        total = max(0, int(round(float(value))))
        return f"{total // 60}:{total % 60:02d}"
    except (ValueError, TypeError):
        return "—"


def _write_uploaded_video(uploaded, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as output:
        output.write(uploaded.getbuffer())


def _camera_feed_id(role: str, camera_index: int, uploaded) -> str:
    token = uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"{role}|{camera_index}|{uploaded.name}|{uploaded.size}",
    ).hex[:8]
    return f"{role}_{camera_index}_{token}"


def _build_global_report(job_id: str, role_results: dict, offsets: dict):
    camera_reports = {role: item["report"] for role, item in role_results.items()}
    candidates = build_cross_camera_candidates(role_results, offsets)
    summary = {
        "videos_analyzed": len(camera_reports),
        "camera_local_visitors_sum_not_deduplicated": sum(
            report["summary"]["camera_local_visitors_estimated"] for report in camera_reports.values()
        ),
        "distinct_cross_camera_visitors": None,
        "entrance_entries": sum(report["summary"]["entry_events"] for report in camera_reports.values()),
        "entrance_exits": sum(report["summary"]["exit_events"] for report in camera_reports.values()),
        "groups_detected": sum(report["summary"]["distinct_group_memberships"] for report in camera_reports.values()),
        "group_intervals": sum(report["summary"]["group_intervals"] for report in camera_reports.values()),
        "peak_occupancy_max_per_camera": max(
            (report["summary"]["peak_detected_occupancy"] for report in camera_reports.values()), default=0
        ),
        "low_movement_intervals": sum(report["summary"]["low_movement_intervals"] for report in camera_reports.values()),
        "checkout_presence_events": sum(report["summary"]["checkout_camera_presence_events"] for report in camera_reports.values()),
        "payments_verified": 0,
        "cross_camera_review_candidates": len(candidates),
    }
    return {
        "schema_version": "2.1",
        "pipeline_version": "2.1.0-streamlit",
        "analysis_id": job_id,
        "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "camera_reports": camera_reports,
        "cross_camera_candidates": candidates,
        "cross_camera_policy": {
            "automatic_merge": False,
            "start_offsets": offsets,
            "note": "Candidate links require user-supplied offsets on one synchronized timeline. Candidate links are not identity decisions and do not change counts.",
        },
        "limitations": [
            "Visitor IDs are anonymous and camera-local; counts across camera videos are not deduplicated.",
            "Group labels represent temporary proximity, box-overlap, and co-presence evidence—not family or personal relationships.",
            "Stationary-role candidates are heuristic suggestions, not confirmed staff/cashier labels.",
            "Checkout-camera presence is not proof of a payment. No POS transaction feed was provided.",
            "Confidence values are not calibrated; validate against manually labelled footage before business use.",
            "No age, gender, face identity, intent, or theft classification is performed.",
        ],
    }


def _update_analysis_job(job: dict, **updates):
    with job["lock"]:
        job.update(updates)


def _analysis_job_value(job: dict, key: str, default=None):
    with job["lock"]:
        return job.get(key, default)


def _parse_zoom_command(command: str):
    parts = str(command or "").strip().split()
    if len(parts) == 2 and parts[0].lower() == "zoom":
        token = parts[1].upper()
        if token in {"OUT", "NORMAL", "RESET"}:
            return True, None, "Zoom reset to the full preview."
        if token.isdigit():
            return True, f"T{int(token):03d}", f"Zooming to live tracker T{int(token):03d} when visible."
        if len(token) > 1 and token[0] in {"T", "G", "P"} and token[1:].isdigit():
            normalized = f"{token[0]}{int(token[1:]):03d}" if token[0] == "T" else f"{token[0]}{int(token[1:])}"
            return True, normalized, f"Zooming to {normalized} when visible."
    return False, None, "Use `zoom 1`, `zoom T002`, `zoom G1`, `zoom P1`, or `zoom out`."


def _analysis_worker(job: dict, camera_inputs: list, output_folder: Path,
                     settings: dict, offsets: dict, group_exclusions: dict,
                     group_detection_by_role: dict):
    role_results = {}
    feed_count = max(len(camera_inputs), 1)
    try:
        for idx, (role, camera_id, camera_name, source_path, source_filename) in enumerate(camera_inputs):
            _update_analysis_job(
                job, active_camera=camera_name,
                stage=f"Video {idx + 1}/{feed_count} · starting detector and tracker",
                preview=None,
            )

            def on_progress(local_progress, stage, role_idx=idx, camera_label=camera_name):
                local = min(1.0, max(0.0, float(local_progress)))
                overall = (role_idx + local) / feed_count
                _update_analysis_job(
                    job, progress=min(0.99, overall),
                    stage=f"Video {role_idx + 1}/{feed_count} · {stage}",
                    active_camera=camera_label,
                )

            def on_preview(rgb_frame, camera_label=camera_name):
                _update_analysis_job(job, preview=rgb_frame.copy(), active_camera=camera_label)

            camera_settings = {
                **settings,
                "detect_groups": bool(group_detection_by_role.get(role, True)),
                "exclude_group_visitor_ids": group_exclusions.get(camera_id, []),
            }
            result = analyze_video(
                str(source_path), role, str(output_folder), camera_settings,
                progress=on_progress, preview_callback=on_preview,
                camera_id=camera_id, camera_name=camera_name,
                preview_zoom_getter=lambda: _analysis_job_value(job, "zoom_target"),
            )
            result["report"]["source_filename"] = source_filename
            role_results[camera_id] = result
            _update_analysis_job(job, completed_feeds=idx + 1)
            on_progress(1.0, "tracklets reconciled; annotated render ready")

        report = _build_global_report(job["job_id"], role_results, offsets)
        write_json(str(output_folder / "global_report.json"), report)
        write_summary_csv(str(output_folder / "summary.csv"), report)
        result = {
            "job_id": job["job_id"],
            "folder": str(job["folder"]),
            "output_folder": str(output_folder),
            "report": report,
            "role_results": role_results,
        }
        _update_analysis_job(job, status="complete", progress=1.0,
                             stage="Analysis complete", result=result)
    except Exception as exc:
        _update_analysis_job(job, status="error", error=str(exc), traceback=traceback.format_exc())


def _run_analysis(files: dict, settings: dict, offsets: dict, group_exclusions: dict,
                  group_detection_by_role: dict):
    job_id = uuid.uuid4().hex[:12]
    job_folder = DATA_ROOT / job_id
    upload_folder = job_folder / "uploads"
    output_folder = job_folder / "outputs"
    upload_folder.mkdir(parents=True, exist_ok=True)
    output_folder.mkdir(parents=True, exist_ok=True)

    camera_inputs = []
    for role in ROLES:
        selected = files.get(role) or []
        if not isinstance(selected, (list, tuple)):
            selected = [selected]
        for camera_index, uploaded in enumerate(selected, start=1):
            if uploaded is None:
                continue
            camera_id = _camera_feed_id(role, camera_index, uploaded)
            camera_name = f"{ROLE_TITLES[role]} · Cam {camera_index}"
            ext = Path(uploaded.name).suffix.lower() or ".mp4"
            source_path = upload_folder / f"{camera_id}{ext}"
            _write_uploaded_video(uploaded, source_path)
            camera_inputs.append((role, camera_id, camera_name, source_path, uploaded.name))

    if not camera_inputs:
        st.warning("Select at least one video before starting analysis.")
        return

    st.session_state.pop("retail_flow_result", None)
    job = {
        "job_id": job_id,
        "folder": str(job_folder),
        "status": "running",
        "progress": 0.0,
        "stage": f"Queued {len(camera_inputs)} video feed(s)",
        "active_camera": camera_inputs[0][2],
        "feed_count": len(camera_inputs),
        "completed_feeds": 0,
        "preview": None,
        "zoom_target": None,
        "zoom_message": "",
        "error": None,
        "traceback": None,
        "result": None,
        "lock": threading.Lock(),
    }
    st.session_state["retail_flow_job"] = job
    worker = threading.Thread(
        target=_analysis_worker,
        args=(job, camera_inputs, output_folder, dict(settings), dict(offsets),
              dict(group_exclusions), dict(group_detection_by_role)),
        daemon=True,
        name=f"analysis-{job_id}",
    )
    worker.start()


@st.fragment(run_every=0.5)
def _render_analysis_monitor():
    job = st.session_state.get("retail_flow_job")
    if not job:
        return
    with job["lock"]:
        state = {key: value for key, value in job.items() if key != "lock"}
    if state.get("status") == "complete":
        if state.get("result") is not None:
            st.session_state["retail_flow_result"] = state["result"]
            st.rerun()
        return
    if state.get("status") == "error":
        st.error(f"Analysis failed: {state.get('error', 'Unknown error')}")
        with st.expander("Technical details"):
            st.code(state.get("traceback", ""))
        return

    percent = int(min(99, max(0, round(float(state.get("progress", 0.0)) * 100))))
    completed = int(state.get("completed_feeds", 0))
    feed_count = int(state.get("feed_count", 1))
    st.markdown("### Analysis in progress")
    st.progress(
        float(state.get("progress", 0.0)),
        text=f"{percent}% · {completed}/{feed_count} feeds complete · {state.get('active_camera', 'Preparing')} · {state.get('stage', '')}",
    )
    if state.get("preview") is not None:
        st.image(state["preview"], channels="RGB", width="stretch",
                 caption=f"{state.get('active_camera', 'Active feed')} · UI-only live zoom; final MP4 is unchanged")
    else:
        st.caption("The live preview will appear after the first sampled frame.")

    with st.form("live_zoom_terminal", clear_on_submit=True):
        command = st.text_input("Live zoom terminal", placeholder="zoom 1  ·  zoom G1  ·  zoom out")
        submitted = st.form_submit_button("Apply zoom")
    if submitted:
        valid, target, message = _parse_zoom_command(command)
        if valid:
            _update_analysis_job(job, zoom_target=target, zoom_message=message)
            st.success(message)
        else:
            st.error(message)
    st.caption(
        "`zoom 1` targets tracker T001; `zoom G1` targets provisional live group G1 (the final offline ID may change); `zoom out` restores the full frame."
    )


# Header
header_left, header_right = st.columns([4, 1])
with header_left:
    st.markdown('<div class="rf-eyebrow">Recorded video analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="rf-hero">Retail video analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="rf-sub">Upload recorded clips. Review person tracks, proximity groups, movement, entrance events and annotated video.</div>', unsafe_allow_html=True)
with header_right:
    st.markdown("<div class='rf-card'><b>Multiple feeds per role</b><br><span style='color:#a0a6ad;font-size:12px'>Upload one or more clips for each camera type.</span></div>", unsafe_allow_html=True)

st.write("")
live_monitor_slot = st.container()

with st.sidebar:
    st.markdown("### Pipeline settings")
    detector = st.selectbox("Person detector", ["auto", "yolo", "hog"], index=0,
                            help="Auto uses Ultralytics YOLO if installed and otherwise falls back to OpenCV HOG.")
    detector_model = st.text_input("YOLO model", value="yolo11x.pt", help="Defaults to the larger model used by the original prototype; choose yolo11n.pt/s.pt for a faster, lower-capacity run. Used only if Ultralytics is installed.")
    frame_skip = st.slider("Frame skip", 0, 10, 0, 1,
                           help="0 processes every source frame. 1 skips one frame between detections; higher values are faster but less accurate.")
    confidence = st.slider("Detector confidence threshold", 0.10, 0.80, 0.40, 0.05)
    st.markdown("---")
    st.markdown("#### Identity quality")
    reid_backend = st.selectbox(
        "Appearance / ReID feature",
        ["auto", "osnet", "hsv"],
        format_func=lambda value: {"auto": "Auto · cached OSNet, else HSV", "osnet": "OSNet MSMT17 · allow model download", "hsv": "HSV color histogram · no learned model"}[value],
        help="The optional OSNet backend restores the learned ReID feature used by the earlier prototype. Auto only uses cached weights; choose OSNet to download them if needed.",
    )
    min_track_observations = st.slider("Minimum observations per visitor track", 3, 20, 6, 1)
    min_track_sec = st.slider("Minimum track duration", 0.25, 3.0, 0.75, 0.25, format="%.2f sec")
    min_track_confidence = st.slider("Minimum mean detection confidence", 0.0, 0.8, 0.45, 0.05)
    reid_threshold = st.slider("Offline appearance-match threshold", 0.50, 0.99, 0.90, 0.01,
                               help="This is a starting point, not calibrated for your cameras. HSV mode applies a stricter floor to reduce false merges.")
    st.markdown("---")
    st.markdown("#### Groups and role candidates")
    group_detection_by_role = {}
    for group_role in ROLES:
        group_detection_by_role[group_role] = st.checkbox(
            f"Detect groups · {ROLE_TITLES[group_role]}", value=True,
            key=f"detect_groups_{group_role}",
            help="Applies to every uploaded feed of this role. Off skips live and final group detection, but keeps tracking and other metrics.",
        )
    group_seconds = st.slider("Minimum together time", 0.5, 12.0, 3.0, 0.5, format="%.1f sec", 
                               help="Minimum duration for proximity-based group evidence.")
    overlap_seconds = st.slider("Minimum continuous box overlap", 0.5, 12.0, 3.0, 0.5, format="%.1f sec",
                                help="A pair can also seed a group when meaningful box overlap persists this long. Brief missed-detection gaps are tolerated.")
    overlap_ratio = st.slider("Minimum box overlap ratio", 0.05, 0.60, 0.15, 0.05, format="%.2f",
                              help="Intersection area divided by the smaller box area; filters out tiny edge touches without assuming equal-sized people.")
    likely_share_percent = st.slider("Together-time share for likely group", 5, 90, 40, 5, format="%d%%",
                                     help="The pair must meet this share for each person: qualifying together-time divided by that person's observed in-camera time.")
    group_distance = st.slider("Image-space proximity", 0.50, 2.50, 1.35, 0.05, format="%.2f× scene scale",
                                help="Bounding-box size is only a perspective scale proxy, not a person's measured height. A robust scale allowance helps when adults and children have very different box heights.")
    group_ratio = st.slider("Minimum close ratio", 0.20, 1.00, 0.55, 0.05)
    group_reassign_margin = st.slider("Group reassignment margin", 0.00, 0.75, 0.30, 0.05,
                                      help="A new group must clearly outperform the person's existing group.")
    group_reassign_min_sec = st.slider("Alternative must persist", 3, 20, 8, 1, format="%d sec")
    exclude_stationary = st.checkbox("Exclude likely fixed-post candidates", value=True,
                                     help="Requires long, very stable presence plus repeated visitor interactions. Low movement alone stays eligible; this remains a heuristic.")
    st.markdown("---")
    st.markdown("#### Entrance and store zones")
    entrance_line = st.slider("Entrance line height", 10, 90, 52, 1, format="%d%% of image height") / 100
    entry_direction = st.selectbox("Entry direction", ["down", "up"], format_func=lambda x: "Top → bottom is entry" if x == "down" else "Bottom → top is entry")
    zone_layout = st.selectbox("Store-floor zone overlay", ["columns", "quadrants", "off"],
                               format_func=lambda x: {"columns":"3 image columns", "quadrants":"4 image quadrants", "off":"Off"}[x])
    st.markdown("---")
    st.caption("0 frame skip processes every frame. Higher skips trade detail for speed; counts remain estimates.")

settings = {
    "detector": detector,
    "detector_model": detector_model,
    "reid_backend": reid_backend,
    "frame_skip": int(frame_skip),
    "confidence": float(confidence),
    "min_track_observations": int(min_track_observations),
    "min_track_sec": float(min_track_sec),
    "min_track_confidence": float(min_track_confidence),
    "reid_match_threshold": float(reid_threshold),
    "group_min_together_sec": float(group_seconds),
    "group_min_box_overlap_sec": float(overlap_seconds),
    "group_box_overlap_min_ratio": float(overlap_ratio),
    "group_likely_share_ratio": float(likely_share_percent) / 100.0,
    "group_proximity_heights": float(group_distance),
    "group_reassign_margin": float(group_reassign_margin),
    "group_reassign_min_sec": float(group_reassign_min_sec),
    "group_min_close_ratio": float(group_ratio),
    "entrance_line_y": float(entrance_line),
    "entry_direction": entry_direction,
    "zone_layout": zone_layout,
    "exclude_stationary_from_groups": bool(exclude_stationary),
}

st.markdown("### Add recorded clips")
st.caption("Upload one or more clips per role. Each file is analyzed as a separate camera.")
cols = st.columns(3, gap="medium")
files = {}
offsets = {}
group_exclusions = {}
for col, role, icon, description in zip(
    cols,
    ROLES,
    ["↘", "▦", "▣"],
    [
        "Entries and exits.",
        "Tracks, movement, dwell and groups.",
        "Checkout-area presence; not payment verification.",
    ],
):
    with col:
        st.markdown(f"#### {icon} {ROLE_TITLES[role]}")
        st.caption(description)
        uploaded_files = st.file_uploader(
            f"Choose {ROLE_TITLES[role]} videos", type=VIDEO_TYPES,
            key=f"upload_{role}", accept_multiple_files=True,
        )
        files[role] = uploaded_files or []
        for camera_index, uploaded in enumerate(files[role], start=1):
            camera_id = _camera_feed_id(role, camera_index, uploaded)
            st.caption(f"Cam {camera_index}: **{uploaded.name}** · {uploaded.size / 1024 / 1024:.1f} MB")
            raw_offset = st.text_input(
                f"Offset · Cam {camera_index} (sec)", key=f"offset_{camera_id}",
                placeholder="Optional; synced clips only",
            )
            try:
                value = float(raw_offset) if raw_offset.strip() else None
                offsets[camera_id] = value if value is not None and value >= 0 else None
            except ValueError:
                offsets[camera_id] = None
            excluded_raw = st.text_input(
                f"Exclude from groups · Cam {camera_index} IDs",
                key=f"group_exclusions_{camera_id}", placeholder="Optional, e.g. S-V003",
                help="Excludes group membership only, not counts. Use a prior visitor ID (S-V003) or live ID (T002).", 
            )
            group_exclusions[camera_id] = [item.strip() for item in excluded_raw.split(",") if item.strip()]

st.caption("IDs stay camera-local. Group boxes show proximity/overlap only; checkout video does not confirm payment.")
selected_count = sum(len(uploaded_list) for uploaded_list in files.values())
active_job = st.session_state.get("retail_flow_job")
analysis_running = bool(active_job and _analysis_job_value(active_job, "status") == "running")
run_col, reset_col, note_col = st.columns([1.15, 1, 3])
with run_col:
    run_clicked = st.button("Analyze selected videos", type="primary",
                             disabled=selected_count == 0 or analysis_running, width="stretch")
with reset_col:
    if st.button("Clear current results", width="stretch", disabled=analysis_running):
        st.session_state.pop("retail_flow_result", None)
        st.session_state.pop("retail_flow_job", None)
        st.rerun()
with note_col:
    if selected_count:
        st.caption(f"{selected_count} camera feed{'s' if selected_count != 1 else ''} selected.")
    else:
        st.caption("Select at least one recorded clip to run the pipeline.")

if run_clicked:
    try:
        _run_analysis(files, settings, offsets, group_exclusions, group_detection_by_role)
    except Exception as exc:
        st.error(f"Could not start analysis: {exc}")
        st.exception(exc)

with live_monitor_slot:
    st.markdown("### Live monitor & zoom terminal")
    active_job = st.session_state.get("retail_flow_job")
    if active_job:
        job_status = _analysis_job_value(active_job, "status", "running")
        if job_status == "running":
            _render_analysis_monitor()
        elif job_status == "error":
            with active_job["lock"]:
                job_error = active_job.get("error", "Unknown error")
                job_traceback = active_job.get("traceback", "")
            st.error(f"Analysis failed: {job_error}")
            with st.expander("Technical details"):
                st.code(job_traceback)
        elif job_status == "complete":
            completed_result = st.session_state.get("retail_flow_result") or _analysis_job_value(active_job, "result")
            if completed_result and not st.session_state.get("retail_flow_result"):
                st.session_state["retail_flow_result"] = completed_result
            if completed_result:
                analyzed_count = completed_result.get("report", {}).get("summary", {}).get("videos_analyzed", 0)
                st.success(f"Analysis complete · {analyzed_count} video feed(s). Results are below.")
                st.markdown("[Jump to the Video grid and results](#analysis-results)")
    elif st.session_state.get("retail_flow_result"):
        st.success("Analysis complete. Results are below.")
        st.markdown("[Jump to the Video grid and results](#analysis-results)")
    else:
        st.caption("After you click **Analyze selected videos**, live progress and the zoom terminal appear here. The finished Video grid is in the results below.")

result = st.session_state.get("retail_flow_result")
if result:
    st.markdown('<div id="analysis-results"></div>', unsafe_allow_html=True)
    report = result["report"]
    summary = report["summary"]
    st.divider()
    st.markdown("<div class='rf-eyebrow'>Analysis complete</div>", unsafe_allow_html=True)
    st.markdown("## Store flow, at a glance")
    st.caption(f"Analysis {result['job_id']} · {len(report['camera_reports'])} camera clip(s) · {report['analysis_timestamp']}")
    kpis = st.columns(5)
    kpis[0].metric("Videos analyzed", summary["videos_analyzed"])
    kpis[1].metric("Local visitor estimates", summary["camera_local_visitors_sum_not_deduplicated"])
    has_entrance_feed = any(item.get("camera_role") == "entrance" for item in report["camera_reports"].values())
    kpis[2].metric("Entrance entries", summary["entrance_entries"] if has_entrance_feed else "—")
    kpis[3].metric("Peak occupancy", summary["peak_occupancy_max_per_camera"])
    kpis[4].metric("Distinct group memberships", summary["groups_detected"])
    st.caption(f"Group episodes: {summary['group_intervals']} · checkout-camera presence: {summary['checkout_presence_events']} · verified payments: {summary['payments_verified']}")
    st.warning("Camera-local IDs are not deduplicated across clips. Group labels reflect temporary proximity/overlap, not relationships. Checkout-camera presence is not proof of payment.")

    output_folder = Path(result["output_folder"])
    report_path = output_folder / "global_report.json"
    csv_path = output_folder / "summary.csv"
    download_cols = st.columns([1, 1, 4])
    with download_cols[0]:
        st.download_button("Download JSON report", report_path.read_bytes(), file_name="global_report.json", mime="application/json", width="stretch")
    with download_cols[1]:
        st.download_button("Download summary CSV", csv_path.read_bytes(), file_name="summary.csv", mime="text/csv", width="stretch")

    camera_ids = list(report["camera_reports"])
    camera_titles = [
        report["camera_reports"][camera_id].get(
            "camera_name",
            ROLE_TITLES.get(report["camera_reports"][camera_id].get("camera_role", camera_id), camera_id),
        )
        for camera_id in camera_ids
    ]
    tabs = st.tabs(["Video grid"] + camera_titles + ["Groups & events", "Audit report"])
    with tabs[0]:
        st.caption("Each tile is a separate camera clip; use the camera tabs for its metrics and audit details.")
        if camera_ids:
            grid_columns = 2 if len(camera_ids) > 1 else 1
            for row_start in range(0, len(camera_ids), grid_columns):
                row_ids = camera_ids[row_start:row_start + grid_columns]
                columns = st.columns(grid_columns, gap="medium")
                for column, camera_id in zip(columns, row_ids):
                    camera_report = report["camera_reports"][camera_id]
                    camera_name = camera_report.get("camera_name", camera_id)
                    output_info = camera_report.get("outputs", {})
                    annotated_path = output_folder / output_info.get("annotated_video", "")
                    with column:
                        st.markdown(f"#### {camera_name}")
                        st.caption(f"{camera_report.get('camera_role', 'camera').title()} · {camera_report.get('source_filename', camera_name)}")
                        if annotated_path.exists():
                            st.video(str(annotated_path))
                        else:
                            st.warning("Annotated MP4 was not generated.")
        else:
            st.info("No camera videos are available to display.")

    for index, camera_id in enumerate(camera_ids):
        camera_report = report["camera_reports"][camera_id]
        role = camera_report.get("camera_role", camera_id)
        camera_name = camera_report.get("camera_name", ROLE_TITLES.get(role, camera_id))
        output_info = camera_report.get("outputs", {})
        annotated_path = output_folder / output_info.get("annotated_video", "")
        track_csv_path = output_folder / output_info.get("tracklets_csv", "")
        with tabs[index + 1]:
            left, right = st.columns([1.35, 1], gap="large")
            with left:
                st.markdown(f"### Final annotated video · {camera_name}")
                if annotated_path.exists():
                    st.caption("Preview this clip in the Video grid tab.")
                    st.download_button(
                        "Download annotated MP4",
                        annotated_path.read_bytes(),
                        file_name=annotated_path.name,
                        mime="video/mp4",
                        key=f"download_video_{camera_id}_{result['job_id']}",
                    )
                else:
                    st.error("Annotated MP4 was not generated.")
            with right:
                camera_summary = camera_report["summary"]
                metric_cols = st.columns(2)
                metric_cols[0].metric("Camera-local visitor estimates", camera_summary["camera_local_visitors_estimated"])
                metric_cols[1].metric("Tracklets", camera_summary["tracklets"])
                metric_cols[0].metric("Peak occupancy", camera_summary["peak_detected_occupancy"])
                metric_cols[1].metric("Group episodes", camera_summary["group_intervals"])
                if role == "entrance":
                    st.metric("Entry / exit events", f"{camera_summary['entry_events']} / {camera_summary['exit_events']}")
                    st.caption(f"Pre-existing at clip start: {camera_summary['preexisting_occupants_at_clip_start']} · first seen inside later: {camera_summary['first_detected_inside_after_clip_start']}")
                if role == "checkout":
                    st.metric("Checkout-camera presence windows", camera_summary["checkout_camera_presence_events"])
                    st.caption("Visual presence only; no transaction/payment was verified.")
                detector_info = camera_report["detector"]
                st.caption(f"Detector: `{detector_info['backend']}` · tracker: `{detector_info.get('tracker_backend','unknown')}` · frame skip: {camera_report['video'].get('frame_skip', 0)}")
                st.caption(f"Source: `{camera_report.get('source_filename', camera_name)}`")
                if camera_report.get("warnings"):
                    for warning in camera_report["warnings"]:
                        st.warning(warning)
                group_policy = camera_report.get("group_assignment_policy", {})
                if not group_policy.get("group_detection_enabled", True):
                    st.info("Group detection is off for this role. Visitor tracking and other metrics are still active.")
                if group_policy.get("unmatched_manual_exclusion_inputs"): 
                    st.warning(f"No matching group-exclusion ID: {', '.join(group_policy['unmatched_manual_exclusion_inputs'])}")
                if group_policy.get("manually_excluded_visitor_ids"):
                    st.caption("Excluded from groups: " + ", ".join(group_policy["manually_excluded_visitor_ids"]))
            st.markdown("#### Visitor / role estimates")
            visitors = camera_report.get("visitors", [])
            if visitors:
                st.dataframe(visitors, width="stretch", hide_index=True)
            else:
                st.info("No tracks met the configured quality thresholds.")
            stationary = camera_report.get("stationary_role_candidates", [])
            if stationary:
                st.markdown("#### Low-movement / fixed-post candidates · review only")
                st.dataframe(stationary, width="stretch", hide_index=True)
            st.markdown("#### Tracklet audit")
            st.dataframe(camera_report.get("tracklets", []), width="stretch", hide_index=True)
            if track_csv_path.exists():
                st.download_button("Download tracklet CSV", track_csv_path.read_bytes(), file_name=track_csv_path.name,
                                   mime="text/csv", key=f"download_track_csv_{camera_id}_{result['job_id']}")

    groups_tab = tabs[len(camera_ids) + 1]
    with groups_tab:
        st.caption("Likely group status uses the configured together-time share for every member. Pair shares and observed durations are included in the table; group labels do not imply relationships.")
        all_groups = []
        for camera_id, camera_report in report["camera_reports"].items():
            camera_name = camera_report.get("camera_name", camera_id)
            for group in camera_report.get("groups", []):
                all_groups.append({"camera": camera_name, **group})
        if all_groups:
            st.dataframe(all_groups, width="stretch", hide_index=True)
        elif not any(
            camera_report.get("group_settings", {}).get("group_detection_enabled", True)
            for camera_report in report["camera_reports"].values()
        ):
            st.info("Group detection is switched off for all analyzed feeds; visitor tracking and other metrics are unaffected.")
        else:
            st.info("No group intervals met the configured overlap/proximity and time thresholds. Review the video and settings if needed.")
        st.markdown("#### Entrance line-crossing events")
        entry_events = [
            {"camera": camera_report.get("camera_name", camera_id), **event}
            for camera_id, camera_report in report["camera_reports"].items()
            if camera_report.get("camera_role") == "entrance"
            for event in camera_report.get("entrance_events", [])
        ]
        st.dataframe(entry_events, width="stretch", hide_index=True) if entry_events else st.caption("No entrance events in the uploaded entrance feed(s).")
        st.markdown("#### Checkout-camera presence")
        checkout_events = [
            {"camera": camera_report.get("camera_name", camera_id), **event}
            for camera_id, camera_report in report["camera_reports"].items()
            if camera_report.get("camera_role") == "checkout"
            for event in camera_report.get("checkout_presence_events", [])
        ]
        st.dataframe(checkout_events, width="stretch", hide_index=True) if checkout_events else st.caption("No checkout feed was uploaded or no visitor track was retained.")
        candidates = report.get("cross_camera_candidates", [])
        st.markdown("#### Cross-camera review candidates")
        if candidates:
            st.dataframe(candidates, width="stretch", hide_index=True)
            st.caption("These candidate matches are not confirmed and did not change any counts.")
        else:
            st.caption("None. Candidate generation needs at least two clips with user-supplied synchronized start offsets.")

    with tabs[len(camera_ids) + 2]:
        st.json(report, expanded=False)
        st.caption(f"Output folder: `{result['output_folder']}`")
        if st.button("Delete this analysis and uploaded videos", type="secondary"):
            shutil.rmtree(result["folder"], ignore_errors=True)
            st.session_state.pop("retail_flow_result", None)
            st.success("Analysis files deleted.")
            st.rerun()

st.divider()
st.caption("Local research prototype · no face recognition · no family/relationship inference · no payment verification · validate against ground truth before operational use.")
