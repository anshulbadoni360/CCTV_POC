# Retail Flow — recorded-video analytics (Streamlit)

A local Streamlit app for uploading one or more recorded retail-camera videos and reviewing person tracking, tracklet reconciliation, movement, group proximity, entrance events, and checkout-camera presence.

## Run the Streamlit app

Python 3.10+:

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true
```

Open the Streamlit URL printed by the command. In the Arena preview, the app is served on port 8501. The optional `.streamlit/config.toml` sets upload size and preview-friendly server settings.

Each role accepts **one or more recorded clips**. Every uploaded file is analyzed as its own camera feed, with its own local IDs, output video, and report. One clip in any role still works by itself:

- **Entrance** — directional line-crossing estimates and occupancy at clip start.
- **Store floor** — camera-local visitor estimates, groups, movement, and zone dwell.
- **Checkout / purchase camera** — anonymous tracks and visual-presence windows; not proof of payment.

Supported video formats: MP4, MOV, AVI, MKV, WEBM, M4V. The configured per-file Streamlit upload limit is 1 GB.

## Optional YOLO detector

Install the optional Ultralytics/Torch extra for YOLO detection, persistent BoT-SORT tracking, and the OSNet dependencies:

```bash
python -m pip install -r requirements-yolo.txt
```

The app defaults to **Auto**: it tries Ultralytics YOLO (`yolo11x.pt`, matching the original prototype's larger default) and its BoT-SORT tracker if available. You can type a smaller model such as `yolo11n.pt` when runtime matters more than recall. The first YOLO run may download model weights. OSNet **Auto** mode only uses cached ReID weights; choose **OSNet MSMT17** in the sidebar to allow downloading its weights. If YOLO is unavailable, the app falls back to OpenCV HOG plus a low-cost motion/overlap matcher. HOG is a lower-recall fallback, especially for small, distant, overhead, occluded, or crowded people; the report records the detector and tracker backends.

## Processing and live video

During processing, the **Live monitor & zoom terminal** appears near the top of the page with a sampled-frame preview, progress, bounding boxes, tracker IDs (`T…`), and provisional `G#` candidates. Multiple selected feeds are processed sequentially; the status shows the current feed and how many feeds are complete. After all feeds finish, the results include a multi-video grid plus separate camera tabs. The terminal accepts `zoom 1` / `zoom T001`, `zoom G1`, and `zoom out`; zoom is UI-only and never changes the source or final MP4. A live group can show `PENDING SHARE` or `LIKELY SO FAR`; final likelihood uses the complete clip and offline assignments may renumber groups. Report-style HUD titles are not burned into the videos, while person/group labels and boxes remain. After processing, each final MP4 shows `T → camera-local visitor ID` labels, trajectories, the entrance line or store zones, and a separate enclosing box around confirmed group intervals.

**Frame skip** defaults to 0, which processes every source frame. A value of 1 skips one frame between detections; higher values run faster but can reduce tracking and group accuracy. YOLO mode uses persistent Ultralytics BoT-SORT tracker IDs when available. ReID can use the original OSNet MSMT17 feature extractor if its optional dependencies and cached weights are available; **Auto** checks the cache without downloading, while **OSNet** explicitly allows a weight download. Otherwise it falls back to an HSV clothing histogram. Neither similarity threshold is calibrated for your camera.

## What changed after the first prototype

- The initial web build sampled at 6 FPS and used a custom greedy IoU/motion tracker even when YOLO was present. The primary Streamlit app now exposes an explicit frame-skip control and uses persistent Ultralytics BoT-SORT tracker IDs when available.
- The initial visitor filter accepted very short tracks. It now defaults to at least 6 observations, 0.75 seconds, and a configurable mean detection-confidence floor.
- The entrance counter now uses a line deadband and two-sample stable-side confirmation. A track first seen inside later in the clip is no longer mislabeled as a pre-existing occupant at clip start.
- Group detection has independent Entrance, Store floor and Checkout toggles in Streamlit; all default on, and each toggle applies to every uploaded feed of that role. Turning one off skips live/final group analysis only—tracking, counts, and other role metrics continue.
- Group intervals are exclusive by person. Existing assignments persist across intervals; by default, an alternative must remain plausible for 8 seconds and beat the current membership by a 0.30 score margin. A dyad can qualify through image-space proximity or meaningful box overlap sustained for the configured duration (3 seconds by default). “Likely group” status additionally requires qualifying together-time to reach the configured share (40% by default) of each member’s observed in-camera time. Larger groups expand from a dyad only when a new member is near/overlapping at least two members, with co-movement checked when measurable. Low movement alone does not imply staff; fixed-post exclusion needs repeated visitor interactions, and a camera-local ID can always be excluded manually while remaining in counts.
- The Streamlit interface shows tracker boxes during the analysis and the final annotated video afterward.

This addresses the known causes of the first build's faster-but-noisier output. It does **not** make counts ground truth; validate against annotated footage before relying on them.

## Pipeline stages

1. Person detection with Ultralytics YOLO (optional) or OpenCV HOG fallback.
2. Persistent BoT-SORT IDs for YOLO detections, or motion/overlap association in fallback mode.
3. Tracklet quality filtering and offline tracklet-to-identity reconciliation using temporal conflict, movement compatibility, and OSNet MSMT17 embeddings when available (otherwise a coarse HSV histogram).
4. Anonymous camera-local visitor estimates, trajectories, occupancy snapshots, dwell zones, and low-movement intervals.
5. Entrance line events with deadband/hysteresis on the entrance-role video.
6. Live group-candidate boxes and final exclusive group intervals. Dyads may use proximity or sustained bounding-box overlap; overlap is normalized by the smaller box area, not by assumed physical height. A configurable together-time share determines “likely group” status against each member’s observed in-camera time. Low movement alone does not remove someone from a group; only likely fixed-post candidates or explicitly excluded IDs are filtered.
7. Final annotated MP4 rendering and JSON/CSV reports.
8. Optional cross-camera **review candidates** only when the user provides clip offsets on one synchronized timeline. No automatic cross-camera ID merge.

## Results and files

Each run is saved under `data/streamlit_jobs/{analysis_id}/`:

- `outputs/{camera_id}_annotated.mp4` — final annotated video for each feed.
- `outputs/{camera_id}_tracklets.csv` — tracker/tracklet/visitor audit for each feed.
- `outputs/global_report.json` — per-camera reports, event evidence, warnings, and review candidates.
- `outputs/summary.csv` — per-camera roll-up separating distinct group member sets from group intervals.

The Streamlit results view provides a side-by-side video grid for all analyzed feeds, camera detail tabs, JSON/CSV/MP4 downloads, entrance and checkout event tables, groups, tracklets, and stationary-role candidates. The report-style header HUD is not burned into the annotated videos; person/group labels and boxes remain. Use **Delete this analysis and uploaded videos** in the audit tab to remove the saved job.

## Synchronized camera offsets

Each upload has an optional start offset in seconds. Use it only if you know the clip's time relative to the **same synchronized session timeline**. Leave blank otherwise. Even with offsets, cross-camera identity matches are only suggestions using time order and coarse clothing color; they never change the reported camera-local counts.

## Important limitations

- **The supplied report shows a real regression:** its 6 FPS custom tracker, loose filtering, and lack of staff-like track handling could produce more IDs and group episodes than the five people actually present. The same visitor was sometimes counted as both pre-existing and as entering because late first observations were treated as clip-start occupants. Those behaviors have been corrected in the pipeline, but the exact video must be re-run to see whether it now matches the scene.
- OSNet has been restored as an optional appearance backend. If weights/dependencies are unavailable, the report warns that the weaker HSV fallback was used. Even OSNet is not calibrated on this camera and must not be treated as confirmed identity.
- The five-person/two-customer-group/cashier ground truth is not available in this workspace, so I cannot honestly claim the corrected pipeline now returns exactly five people and two groups. Use the annotated output and tracklet audit to verify.
- Camera-local IDs are not deduplicated across camera clips. A person visible in two uploaded videos can be present in both local counts.
- “Stationary-role candidate” is a heuristic, not staff identification. Review the candidate before excluding it.
- Group labels mean temporary proximity, box overlap and co-presence patterns, not family or relationships. A long queue or occlusion can still produce false positives; validate the thresholds on annotated footage.
- An entrance event is a footpoint crossing of a configured image-space line, not definitive proof that someone entered the store. The user must position and orient the line for that camera.
- Checkout-camera presence is not payment evidence. POS transaction linkage needs a separately designed, synchronized POS stage.
- Pixel-space distance is not physical distance; no camera calibration/homography is used.
- Confidence values are heuristic and uncalibrated. No age, gender, face identity, intent, theft, or suspicious-behavior classification is performed.
- The prototype has no login, tenant separation, or production-grade encrypted storage. Do not expose it publicly with CCTV footage.

## Files

- `streamlit_app.py` — primary Streamlit upload, live preview, analysis and report UI.
- `pipeline.py` — detector/tracker adapters, tracklets, offline reconciliation, entrance/group/behavior metrics, annotated video and CSV/JSON output.
- `server.py` and `static/index.html` — the earlier optional custom HTTP UI/API; use Streamlit as the primary app.
- `tests/test_pipeline.py` — deterministic pipeline smoke tests.
