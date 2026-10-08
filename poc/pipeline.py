"""Recorded-video retail analytics pipeline.

The pipeline is deliberately camera-local and conservative: detections become
tracklets, tracklets may be reconciled offline within a clip, and the resulting
IDs are anonymous estimates rather than biometric identities.  Cross-camera
links are only review candidates when the user supplies synchronized clip
start offsets.
"""
from __future__ import annotations

import csv
import json
import math
from bisect import bisect_right
import os
import re
import time
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


ROLE_PREFIX = {"entrance": "E", "store": "S", "checkout": "C"}
ROLE_LABEL = {
    "entrance": "Entrance",
    "store": "Store floor",
    "checkout": "Checkout / purchase camera",
}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


@dataclass
class PipelineSettings:
    detector: str = "auto"              # auto | yolo | hog
    detector_model: str = "yolo11x.pt"
    reid_backend: str = "auto"           # auto cached OSNet | osnet download | hsv
    confidence: float = 0.40
    frame_skip: int = 0
    analysis_fps: Optional[float] = None  # Backward compatibility for the legacy HTTP UI.
    max_width: int = 960
    max_track_age_sec: float = 2.0
    min_track_sec: float = 0.75
    min_track_observations: int = 6
    min_track_confidence: float = 0.45
    merge_gap_sec: float = 4.0
    reid_match_threshold: float = 0.90
    group_proximity_heights: float = 1.35
    group_reassign_margin: float = 0.30
    group_reassign_min_sec: float = 8.0
    exclude_group_visitor_ids: Tuple[str, ...] = ()
    detect_groups: bool = True
    group_min_together_sec: float = 3.0
    group_min_box_overlap_sec: float = 3.0
    group_box_overlap_min_ratio: float = 0.15
    group_likely_share_ratio: float = 0.40
    group_min_close_ratio: float = 0.55
    group_max_gap_sec: float = 0.50
    entrance_line_y: float = 0.52
    entry_direction: str = "down"
    zone_layout: str = "columns"
    exclude_stationary_from_groups: bool = True
    stationary_candidate_min_presence: float = 0.55
    stationary_candidate_radius_heights: float = 0.90

    @classmethod
    def from_dict(cls, raw: Optional[dict]) -> "PipelineSettings":
        raw = raw or {}

        def number(name: str, default: float, low: float, high: float) -> float:
            try:
                value = float(raw.get(name, default))
            except (TypeError, ValueError):
                value = default
            if not math.isfinite(value):
                value = default
            return min(high, max(low, value))

        def boolean(name: str, default: bool) -> bool:
            value = raw.get(name, default)
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            return bool(value)

        detector = str(raw.get("detector", "auto")).lower()
        if detector not in {"auto", "yolo", "hog"}:
            detector = "auto"
        direction = str(raw.get("entry_direction", "down")).lower()
        if direction not in {"down", "up"}:
            direction = "down"
        zone_layout = str(raw.get("zone_layout", "columns")).lower()
        if zone_layout not in {"columns", "quadrants", "off"}:
            zone_layout = "columns"
        model = str(raw.get("detector_model", "yolo11x.pt"))[:120]
        reid_backend = str(raw.get("reid_backend", "auto")).lower()
        if reid_backend not in {"auto", "osnet", "hsv"}:
            reid_backend = "auto"
        excluded_raw = raw.get("exclude_group_visitor_ids", ())
        if isinstance(excluded_raw, str):
            excluded_raw = excluded_raw.split(",")
        if not isinstance(excluded_raw, (list, tuple, set)):
            excluded_raw = ()
        excluded_group_ids = tuple(sorted({str(value).strip() for value in excluded_raw if str(value).strip()}))[:100]
        return cls(
            detector=detector,
            detector_model=model,
            reid_backend=reid_backend,
            confidence=number("confidence", 0.40, 0.05, 0.95),
            frame_skip=int(number("frame_skip", 0, 0, 60)),
            analysis_fps=(number("analysis_fps", 15.0, 1.0, 30.0)
                          if "analysis_fps" in raw and "frame_skip" not in raw else None),
            max_width=int(number("max_width", 960, 320, 1920)),
            max_track_age_sec=number("max_track_age_sec", 2.0, 0.25, 8.0),
            min_track_sec=number("min_track_sec", 0.75, 0.10, 10.0),
            min_track_observations=int(number("min_track_observations", 6, 2, 30)),
            min_track_confidence=number("min_track_confidence", 0.45, 0.0, 0.95),
            merge_gap_sec=number("merge_gap_sec", 4.0, 0.0, 30.0),
            reid_match_threshold=number("reid_match_threshold", 0.90, 0.50, 0.999),
            group_proximity_heights=number("group_proximity_heights", 1.35, 0.4, 3.5),
            group_reassign_margin=number("group_reassign_margin", 0.30, 0.0, 0.75),
            group_reassign_min_sec=number("group_reassign_min_sec", 8.0, 3.0, 30.0),
            exclude_group_visitor_ids=excluded_group_ids,
            detect_groups=boolean("detect_groups", True),
            group_min_together_sec=number("group_min_together_sec", 3.0, 0.5, 30.0),
            group_min_box_overlap_sec=number("group_min_box_overlap_sec", 3.0, 0.5, 30.0),
            group_box_overlap_min_ratio=number("group_box_overlap_min_ratio", 0.15, 0.01, 1.0),
            group_likely_share_ratio=number("group_likely_share_ratio", 0.40, 0.05, 1.0),
            group_min_close_ratio=number("group_min_close_ratio", 0.55, 0.1, 1.0),
            group_max_gap_sec=number("group_max_gap_sec", 0.50, 0.0, 3.0),
            entrance_line_y=number("entrance_line_y", 0.52, 0.05, 0.95),
            entry_direction=direction,
            zone_layout=zone_layout,
            exclude_stationary_from_groups=boolean("exclude_stationary_from_groups", True),
            stationary_candidate_min_presence=number("stationary_candidate_min_presence", 0.55, 0.2, 1.0),
            stationary_candidate_radius_heights=number("stationary_candidate_radius_heights", 0.90, 0.25, 3.0),
        )


@dataclass
class Observation:
    frame: int
    timestamp: float
    bbox: Tuple[int, int, int, int]
    confidence: float
    foot: Tuple[float, float]
    height: float


@dataclass
class Tracklet:
    track_id: int
    source_tracker_id: Optional[int] = None
    observations: List[Observation] = field(default_factory=list)
    feature: Optional[np.ndarray] = None
    feature_count: int = 0
    missed: int = 0

    @property
    def first_frame(self) -> int:
        return self.observations[0].frame if self.observations else 0

    @property
    def last_frame(self) -> int:
        return self.observations[-1].frame if self.observations else 0

    @property
    def mean_confidence(self) -> float:
        if not self.observations:
            return 0.0
        return float(np.mean([o.confidence for o in self.observations]))

    @property
    def mean_height(self) -> float:
        if not self.observations:
            return 1.0
        return float(np.mean([o.height for o in self.observations]))

    def add(self, observation: Observation, feature: Optional[np.ndarray]) -> None:
        self.observations.append(observation)
        self.missed = 0
        if feature is not None:
            feature = feature.astype(np.float32)
            if self.feature is None:
                self.feature = feature.copy()
            else:
                # Small moving update reduces sensitivity to one bad crop.
                self.feature = 0.82 * self.feature + 0.18 * feature
            if np.any(self.feature < -1e-6):
                self.feature /= max(float(np.linalg.norm(self.feature)), 1e-8)
            else:
                self.feature /= max(float(self.feature.sum()), 1e-8)
            self.feature_count += 1


class PersonDetector:
    """YOLO when installed/available; OpenCV HOG is a dependency-light fallback."""

    def __init__(self, settings: PipelineSettings):
        self.settings = settings
        self.backend = ""
        self.warning: Optional[str] = None
        self.model = None
        self.hog = None
        self.tracker_backend = "greedy_motion_fallback"
        if settings.detector in {"auto", "yolo"}:
            try:
                from ultralytics import YOLO  # optional extra
                self.model = YOLO(settings.detector_model)
                self.backend = "ultralytics_yolo"
                self.tracker_backend = "ultralytics_botsort"
            except Exception as exc:
                self.warning = (
                    "Ultralytics YOLO was not available or its model could not be loaded; "
                    "using OpenCV HOG. HOG is a lower-recall fallback, especially for "
                    "small, distant, overhead, or crowded people. Install the optional "
                    "'ultralytics' package and use a compatible YOLO person model for "
                    "better results."
                )
                if settings.detector == "yolo":
                    self.warning = "YOLO requested but unavailable; falling back to OpenCV HOG. " + self.warning
        if not self.backend:
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self.backend = "opencv_hog"

    def detect(self, frame: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], float, Optional[int]]]:
        height, width = frame.shape[:2]
        scale = min(1.0, self.settings.max_width / max(width, 1))
        if scale < 1.0:
            small = cv2.resize(
                frame,
                (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            small = frame
        found: List[Tuple[Tuple[int, int, int, int], float, Optional[int]]] = []
        if self.backend == "ultralytics_yolo":
            # Persist BoT-SORT state between sampled frames. This restores a real
            # tracker rather than relying only on our low-cost fallback matcher.
            result = self.model.track(
                source=small,
                imgsz=max(320, min(1280, self.settings.max_width)),
                conf=self.settings.confidence,
                classes=[0],
                persist=True,
                tracker="botsort.yaml",
                verbose=False,
            )[0]
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                external_ids = (result.boxes.id.int().cpu().tolist()
                                if result.boxes.id is not None else [None] * len(boxes))
                inv_scale = 1.0 / scale
                for box, score, external_id in zip(boxes, scores, external_ids):
                    x1, y1, x2, y2 = box * inv_scale
                    found.append((self._clip_box((x1, y1, x2, y2), width, height),
                                  float(score), int(external_id) if external_id is not None else None))
        else:
            # HOG's standard pedestrian template has a 64x128 window.
            if small.shape[1] >= 72 and small.shape[0] >= 136:
                rects, weights = self.hog.detectMultiScale(
                    small,
                    hitThreshold=0.0,
                    winStride=(8, 8),
                    padding=(8, 8),
                    scale=1.05,
                )
                inv_scale = 1.0 / scale
                for rect, weight in zip(rects, np.asarray(weights).reshape(-1)):
                    x, y, w, h = [float(v) for v in rect]
                    # HOG margins are not calibrated probabilities. This is only
                    # a ranking/confidence proxy, not a model probability.
                    score = float(1.0 / (1.0 + math.exp(-float(np.clip(weight, -8, 8)))))
                    if score < self.settings.confidence:
                        continue
                    found.append((self._clip_box((x * inv_scale, y * inv_scale,
                                                  (x + w) * inv_scale, (y + h) * inv_scale),
                                                 width, height), score, None))
            found = self._nms(found, 0.48)
        return [(box, score, external_id) for box, score, external_id in found
                if box[2] > box[0] and box[3] > box[1]]

    @staticmethod
    def _clip_box(box, width: int, height: int) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = box
        return (
            max(0, min(width - 1, int(round(x1)))),
            max(0, min(height - 1, int(round(y1)))),
            max(1, min(width, int(round(x2)))),
            max(1, min(height, int(round(y2)))),
        )

    @staticmethod
    def _nms(detections, threshold: float):
        if len(detections) < 2:
            return detections
        boxes = [list(item[0]) for item in detections]
        scores = [float(item[1]) for item in detections]
        try:
            indices = cv2.dnn.NMSBoxes(boxes, scores, 0.01, threshold)
            indices = np.asarray(indices).reshape(-1).tolist()
            return [detections[int(i)] for i in indices]
        except Exception:
            return detections


def appearance_descriptor(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    """Coarse HSV clothing histogram; not a learned ReID embedding."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = max(0, x2 - x1), max(0, y2 - y1)
    # Use torso and upper legs; remove some background around box edges.
    xa = max(0, x1 + int(0.08 * bw))
    xb = min(w, x2 - int(0.08 * bw))
    ya = max(0, y1 + int(0.08 * bh))
    yb = min(h, y1 + int(0.72 * bh))
    crop = frame[ya:yb, xa:xb]
    if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 6:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    hist = hist.astype(np.float32).reshape(-1)
    hist /= max(float(hist.sum()), 1e-8)
    return hist


def feature_similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None or a.size != b.size:
        return 0.0
    if np.any(a < -1e-6) or np.any(b < -1e-6):
        # Signed, L2-normalized OSNet embeddings: cosine similarity.
        cosine = float(np.dot(a, b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-8))
        return float(np.clip(cosine, 0.0, 1.0))
    # Nonnegative normalized HSV histograms: Bhattacharyya coefficient.
    return float(np.clip(np.sqrt(np.maximum(a, 0) * np.maximum(b, 0)).sum(), 0.0, 1.0))


def box_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return float(intersection / max(area_a + area_b - intersection, 1))


class TrackletManager:
    """Small online tracker; identity reconciliation is a separate offline stage."""

    def __init__(self, fps: float, settings: PipelineSettings):
        self.fps = fps
        self.settings = settings
        self.max_age_frames = max(1, int(round(settings.max_track_age_sec * fps)))
        self.tracks: Dict[int, Tracklet] = {}
        self.external_to_track: Dict[int, int] = {}
        self.next_id = 1

    def update(self, frame: np.ndarray, frame_idx: int, timestamp: float,
               detections: Sequence[Tuple[Tuple[int, int, int, int], float]],
               appearance_features: Optional[Sequence[np.ndarray]] = None) -> List[int]:
        parsed = []
        for item in detections:
            bbox, confidence = item[0], item[1]
            external_id = item[2] if len(item) > 2 else None
            parsed.append((bbox, float(confidence), int(external_id) if external_id is not None else None))
        if appearance_features is None:
            det_features = [appearance_descriptor(frame, box) for box, _score, _ext in parsed]
        else:
            det_features = [np.asarray(feature, dtype=np.float32) if feature is not None else None
                            for feature in appearance_features]
            if len(det_features) != len(parsed):
                raise ValueError("Appearance feature count must match detection count.")
        assignments: Dict[int, int] = {}
        used_tracks, used_detections = set(), set()

        def create_track(det_idx: int, external_id: Optional[int]) -> int:
            track_id = self.next_id
            self.next_id += 1
            self.tracks[track_id] = Tracklet(track_id=track_id, source_tracker_id=external_id)
            return track_id

        # When YOLO provides persistent BoT-SORT IDs, trust those for online
        # association. They are stronger than a second greedy IoU pass.
        for det_idx, (_bbox, _confidence, external_id) in enumerate(parsed):
            if external_id is None:
                continue
            track_id = self.external_to_track.get(external_id)
            if track_id is not None:
                track = self.tracks.get(track_id)
                if (track is None or frame_idx - track.last_frame > self.max_age_frames
                        or track_id in used_tracks):
                    track_id = None
            if track_id is None:
                track_id = create_track(det_idx, external_id)
                self.external_to_track[external_id] = track_id
            assignments[det_idx] = track_id
            used_tracks.add(track_id)
            used_detections.add(det_idx)

        # OpenCV HOG has no tracker IDs. Use predicted motion, box overlap, and
        # a small appearance tie-breaker as the dependency-light fallback.
        active = [
            track for track in self.tracks.values()
            if track.track_id not in used_tracks
            and 0 <= frame_idx - track.last_frame <= self.max_age_frames
        ]
        candidates = []
        for track in active:
            last = track.observations[-1]
            gap = max(1, frame_idx - last.frame)
            predicted = np.asarray(last.foot, dtype=np.float32)
            if len(track.observations) >= 2:
                prev = track.observations[-2]
                dt = max(1, last.frame - prev.frame)
                velocity = (np.asarray(last.foot) - np.asarray(prev.foot)) / dt
                predicted = predicted + velocity * gap
            for det_idx, (bbox, _score, external_id) in enumerate(parsed):
                if external_id is not None or det_idx in used_detections:
                    continue
                foot = ((bbox[0] + bbox[2]) / 2.0, float(bbox[3]))
                avg_h = max(1.0, 0.5 * (last.height + (bbox[3] - bbox[1])))
                norm_dist = float(np.linalg.norm(predicted - np.asarray(foot)) / avg_h)
                iou = box_iou(last.bbox, bbox)
                appearance = feature_similarity(track.feature, det_features[det_idx])
                if iou < 0.01 and norm_dist > 2.6:
                    continue
                motion_score = math.exp(-norm_dist / 1.25)
                score = 0.48 * iou + 0.37 * motion_score + 0.15 * appearance
                candidates.append((score, track.track_id, det_idx))
        candidates.sort(reverse=True)
        for score, track_id, det_idx in candidates:
            if score < 0.20 or track_id in used_tracks or det_idx in used_detections:
                continue
            assignments[det_idx] = track_id
            used_tracks.add(track_id)
            used_detections.add(det_idx)

        for det_idx, (bbox, confidence, external_id) in enumerate(parsed):
            track_id = assignments.get(det_idx)
            if track_id is None:
                track_id = create_track(det_idx, external_id)
                if external_id is not None:
                    self.external_to_track[external_id] = track_id
            track = self.tracks[track_id]
            if external_id is not None and track.source_tracker_id is None:
                track.source_tracker_id = external_id
            x1, y1, x2, y2 = bbox
            observation = Observation(
                frame=frame_idx,
                timestamp=timestamp,
                bbox=bbox,
                confidence=float(confidence),
                foot=((x1 + x2) / 2.0, float(y2)),
                height=float(max(1, y2 - y1)),
            )
            track.add(observation, det_features[det_idx])
        return [assignments[i] for i in sorted(assignments)]


def _union_find_reconcile(tracks: List[Tracklet], fps: float, settings: PipelineSettings,
                          appearance_threshold: Optional[float] = None):
    valid = [t for t in tracks if t.observations]
    candidates = []
    threshold = settings.reid_match_threshold if appearance_threshold is None else appearance_threshold
    for i, first in enumerate(valid):
        for second in valid[i + 1:]:
            a, b = (first, second) if first.first_frame <= second.first_frame else (second, first)
            gap = (b.first_frame - a.last_frame) / max(fps, 1e-6)
            if gap < 0 or gap > settings.merge_gap_sec:
                continue
            sim = feature_similarity(a.feature, b.feature)
            if sim < threshold:
                continue
            end = np.asarray(a.observations[-1].foot)
            start = np.asarray(b.observations[0].foot)
            mean_h = max(1.0, 0.5 * (a.mean_height + b.mean_height))
            body_distance = float(np.linalg.norm(end - start) / mean_h)
            # Movement plausibility is scale-normalized. It remains a coarse gate
            # because a fixed camera is not a calibrated ground plane.
            if body_distance > max(1.4, gap * 1.8 + 0.8):
                continue
            spatial_score = math.exp(-body_distance / max(0.8, gap * 1.2 + 0.6))
            score = 0.78 * sim + 0.22 * spatial_score
            candidates.append((score, a.track_id, b.track_id, sim, body_distance, gap))
    candidates.sort(reverse=True)
    parent = {t.track_id: t.track_id for t in valid}
    members = {t.track_id: {t.track_id} for t in valid}
    by_id = {t.track_id: t for t in valid}

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    accepted, considered = [], []
    for score, a_id, b_id, sim, distance, gap in candidates:
        ra, rb = root(a_id), root(b_id)
        decision = "accepted"
        if ra == rb:
            decision = "already_connected"
        else:
            conflict = False
            for left_id in members[ra]:
                left = by_id[left_id]
                for right_id in members[rb]:
                    right = by_id[right_id]
                    overlap = not (left.last_frame < right.first_frame or right.last_frame < left.first_frame)
                    if overlap:
                        conflict = True
                        break
                if conflict:
                    break
            if conflict:
                decision = "rejected_temporal_conflict"
            else:
                parent[rb] = ra
                members[ra] |= members[rb]
                del members[rb]
                accepted.append({
                    "tracklet_a": f"T{a_id:04d}",
                    "tracklet_b": f"T{b_id:04d}",
                    "appearance_similarity": round(sim, 4),
                    "spatial_distance_body_lengths": round(distance, 3),
                    "gap_seconds": round(gap, 3),
                    "score": round(score, 4),
                    "decision": "accepted_offline_link",
                })
        considered.append({
            "tracklet_a": f"T{a_id:04d}",
            "tracklet_b": f"T{b_id:04d}",
            "appearance_similarity": round(sim, 4),
            "spatial_distance_body_lengths": round(distance, 3),
            "gap_seconds": round(gap, 3),
            "score": round(score, 4),
            "decision": decision,
        })
    components: Dict[int, List[int]] = {}
    for track_id in parent:
        components.setdefault(root(track_id), []).append(track_id)
    return list(components.values()), accepted, considered


def _stationary_role_candidates(visitor_observations: Dict[str, List[Observation]],
                                clip_duration_sec: float, analysis_fps: float,
                                settings: PipelineSettings) -> Dict[str, dict]:
    """Flag low-motion visitors; infer a fixed-post role only with extra interaction evidence."""
    candidates = {}
    minimum_observations = max(8, int(round(analysis_fps * 3.0)))
    sample_period = 1.0 / max(analysis_fps, 1e-6)
    for visitor_id, observations in visitor_observations.items():
        ordered = sorted(observations, key=lambda obs: obs.frame)
        if len(ordered) < minimum_observations:
            continue
        span = max(0.0, ordered[-1].timestamp - ordered[0].timestamp)
        presence_ratio = span / max(clip_duration_sec, 1e-6)
        if presence_ratio < settings.stationary_candidate_min_presence:
            continue
        positions = np.asarray([obs.foot for obs in ordered], dtype=np.float32)
        center = np.median(positions, axis=0)
        radii = np.linalg.norm(positions - center, axis=1)
        typical_height = max(1.0, float(np.median([obs.height for obs in ordered])))
        radius_heights = float(np.percentile(radii, 90) / typical_height)
        if radius_heights > settings.stationary_candidate_radius_heights:
            continue

        # Low motion alone is not evidence of a cashier. A likely fixed-post
        # candidate additionally stays at a very stable location and has
        # repeated, separate visitors approach that location over the clip.
        interactions = []
        candidate_by_frame = {obs.frame: obs for obs in ordered}
        for other_id, other_observations in visitor_observations.items():
            if other_id == visitor_id:
                continue
            other_by_frame = {obs.frame: obs for obs in other_observations}
            shared_frames = sorted(set(candidate_by_frame) & set(other_by_frame))
            close_frames = []
            for frame in shared_frames:
                fixed_obs, other_obs = candidate_by_frame[frame], other_by_frame[frame]
                distance = float(np.linalg.norm(np.asarray(fixed_obs.foot) - np.asarray(other_obs.foot)))
                if distance / _group_pixel_scale(fixed_obs.height, other_obs.height) <= settings.group_proximity_heights:
                    close_frames.append(frame)
            if len(close_frames) * sample_period >= max(1.0, settings.group_min_together_sec * 0.5):
                interactions.append(other_id)

        likely_fixed_post = (
            presence_ratio >= max(settings.stationary_candidate_min_presence, 0.60)
            and radius_heights <= min(settings.stationary_candidate_radius_heights, 0.55)
            and len(interactions) >= 3
        )
        candidates[visitor_id] = {
            "visitor_id": visitor_id,
            "role_candidate": "likely_fixed_post_candidate" if likely_fixed_post else "stationary_candidate_needs_review",
            "presence_ratio": round(presence_ratio, 3),
            "radius_person_heights_p90": round(radius_heights, 3),
            "distinct_close_visitors": len(interactions),
            "confidence_is_heuristic": True,
            "note": (
                "Very stable long-present track with repeated visitor interactions; possible fixed-post role, not a confirmed cashier."
                if likely_fixed_post else
                "Low movement alone does not establish staff; eligible for group detection unless manually excluded."
            ),
        }
    return candidates


def _group_pixel_scale(height_a: float, height_b: float) -> float:
    """Perspective scale proxy tolerant of different adult/child-sized boxes."""
    return max(1.0, 0.5 * (height_a + height_b), 0.80 * max(height_a, height_b))


def _box_overlap_ratio(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    """Intersection area divided by the smaller box area; no equal-height assumption."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denominator = min(area_a, area_b)
    if denominator <= 0:
        return 0.0
    intersection_width = max(0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0, min(ay2, by2) - max(ay1, by1))
    return float(intersection_width * intersection_height / denominator)


def _split_frame_runs(frames: Iterable[int], max_gap_frames: int) -> List[List[int]]:
    """Split sampled frame indices when a gap exceeds the allowed tolerance."""
    ordered = sorted(set(int(frame) for frame in frames))
    segments, current = [], []
    for frame in ordered:
        if current and frame - current[-1] > max_gap_frames:
            segments.append(current)
            current = []
        current.append(frame)
    if current:
        segments.append(current)
    return segments


def _duration_from_frames(frames: Iterable[int], fps: float, sample_stride: int,
                          max_gap_frames: int) -> float:
    sample_period = sample_stride / max(fps, 1e-6)
    return sum(
        max(sample_period, (segment[-1] - segment[0]) / max(fps, 1e-6) + sample_period)
        for segment in _split_frame_runs(frames, max_gap_frames)
    )


def _observed_presence_seconds(observations: List[Observation], fps: float,
                               sample_stride: int) -> float:
    """Estimate total in-camera presence, summing episodes instead of long ID gaps."""
    if not observations:
        return 0.0
    presence_gap_frames = max(int(round(0.75 * fps)), 2 * sample_stride)
    return _duration_from_frames(
        (obs.frame for obs in observations), fps, sample_stride, presence_gap_frames
    )


def _set_group_time_share_fields(group: dict, together_seconds: float,
                                 member_observed_seconds: Dict[str, float],
                                 threshold: float) -> None:
    shares = {
        member: min(1.0, together_seconds / max(observed_seconds, 1e-6))
        for member, observed_seconds in member_observed_seconds.items()
    }
    minimum_share = min(shares.values(), default=0.0)
    group["together_seconds_total"] = round(together_seconds, 2)
    group["member_observed_seconds"] = {
        member: round(seconds, 2) for member, seconds in member_observed_seconds.items()
    }
    group["member_together_share"] = {
        member: round(share, 4) for member, share in shares.items()
    }
    group["minimum_member_together_share"] = round(minimum_share, 4)
    group["minimum_member_together_share_percent"] = round(minimum_share * 100.0, 1)
    group["likely_group_share_threshold"] = round(threshold, 4)
    group["meets_likely_group_share"] = minimum_share >= threshold
    group["classification"] = "likely_group" if minimum_share >= threshold else "observed_group"


def _co_movement_agrees(velocity_a: Optional[np.ndarray], velocity_b: Optional[np.ndarray],
                        min_speed: float = 0.07) -> Optional[bool]:
    """Compare body-height-normalized image velocities; None means insufficient motion evidence."""
    if velocity_a is None or velocity_b is None:
        return None
    speed_a = float(np.linalg.norm(velocity_a))
    speed_b = float(np.linalg.norm(velocity_b))
    if speed_a < min_speed and speed_b < min_speed:
        return None
    if speed_a < min_speed or speed_b < min_speed:
        return False
    cosine = float(np.dot(velocity_a, velocity_b) / max(speed_a * speed_b, 1e-8))
    speed_ratio = min(speed_a, speed_b) / max(speed_a, speed_b, 1e-8)
    return cosine >= 0.35 and speed_ratio >= 0.25


def _close_group_frames(visitor_observations: Dict[str, List[Observation]], settings: PipelineSettings,
                        fps: float, sample_stride: int, role_prefix: str):
    """Build dyad cores from proximity or sustained box overlap, then expand conservatively."""
    frame_people: Dict[int, Dict[str, Tuple[Tuple[float, float], float, Tuple[int, int, int, int]]]] = {}
    ordered_observations = {}
    velocity_by_frame: Dict[str, Dict[int, np.ndarray]] = {}
    for visitor_id, observations in visitor_observations.items():
        ordered = sorted(observations, key=lambda obs: obs.frame)
        ordered_observations[visitor_id] = ordered
        for obs in ordered:
            frame_people.setdefault(obs.frame, {})[visitor_id] = (obs.foot, obs.height, obs.bbox)
        velocities = {}
        for index in range(1, len(ordered)):
            current = ordered[index]
            # Smooth detector jitter over at most 0.35 seconds, respecting sample gaps.
            previous = ordered[index - 1]
            for candidate_previous in reversed(ordered[:index - 1]):
                elapsed_candidate = current.timestamp - candidate_previous.timestamp
                frame_gap = current.frame - candidate_previous.frame
                if elapsed_candidate <= 0.35 and frame_gap <= int(round(settings.group_max_gap_sec * fps)) + sample_stride:
                    previous = candidate_previous
                else:
                    break
            elapsed = current.timestamp - previous.timestamp
            if elapsed <= 0:
                continue
            delta = np.asarray(current.foot, dtype=np.float32) - np.asarray(previous.foot, dtype=np.float32)
            velocities[current.frame] = delta / (elapsed * max(current.height, 1.0))
        velocity_by_frame[visitor_id] = velocities

    max_gap_frames = max(0, int(round(settings.group_max_gap_sec * fps))) + sample_stride
    sample_period = sample_stride / max(fps, 1e-6)

    # An overlap edge is available only after it has persisted for the configured
    # interval. Small sampling/detection gaps are tolerated, but not a brief pass-by.
    raw_overlap_frames: Dict[Tuple[str, str], List[int]] = {}
    observed_pair_frames: Dict[Tuple[str, str], List[int]] = {}
    overlap_ratios: Dict[Tuple[str, str], Dict[int, float]] = {}
    for frame, people in frame_people.items():
        ids = sorted(people)
        for idx, left in enumerate(ids):
            for right in ids[idx + 1:]:
                pair = (left, right)
                observed_pair_frames.setdefault(pair, []).append(frame)
                ratio = _box_overlap_ratio(people[left][2], people[right][2])
                if ratio < settings.group_box_overlap_min_ratio:
                    continue
                raw_overlap_frames.setdefault(pair, []).append(frame)
                overlap_ratios.setdefault(pair, {})[frame] = ratio

    qualified_overlap_frames: Dict[Tuple[str, str], set] = {}
    for pair, observed_frames in observed_pair_frames.items():
        overlapping = set(raw_overlap_frames.get(pair, []))
        current_segment = []
        last_positive_frame = None
        candidate_segments = []
        for frame in sorted(observed_frames):
            if frame not in overlapping:
                # A jointly observed non-overlap is a real break; only missing
                # detections may be bridged by the configured gap tolerance.
                if current_segment:
                    candidate_segments.append(current_segment)
                    current_segment = []
                last_positive_frame = None
                continue
            if current_segment and last_positive_frame is not None and frame - last_positive_frame > max_gap_frames:
                candidate_segments.append(current_segment)
                current_segment = []
            current_segment.append(frame)
            last_positive_frame = frame
        if current_segment:
            candidate_segments.append(current_segment)
        for segment in candidate_segments:
            if _duration_from_frames(segment, fps, sample_stride, max_gap_frames) >= settings.group_min_box_overlap_sec:
                qualified_overlap_frames.setdefault(pair, set()).update(segment)

    active_members: Dict[Tuple[str, ...], List[int]] = {}

    def co_moving_with_core(candidate: str, members: set, frame: int) -> bool:
        candidate_velocity = velocity_by_frame.get(candidate, {}).get(frame)
        comparisons = [
            _co_movement_agrees(candidate_velocity, velocity_by_frame.get(member, {}).get(frame))
            for member in members
        ]
        observed = [agreement for agreement in comparisons if agreement is not None]
        return all(observed) if observed else True

    for frame, people in frame_people.items():
        ids = sorted(people)
        adjacency = {visitor_id: set() for visitor_id in ids}
        for idx, left in enumerate(ids):
            point_left, height_left, box_left = people[left]
            for right in ids[idx + 1:]:
                point_right, height_right, box_right = people[right]
                distance = float(np.linalg.norm(np.asarray(point_left) - np.asarray(point_right)))
                # Box height is a perspective scale proxy only, not a biological-height match.
                proximity = distance / _group_pixel_scale(height_left, height_right) <= settings.group_proximity_heights
                pair = (left, right)
                sustained_overlap = frame in qualified_overlap_frames.get(pair, set())
                if not proximity and not sustained_overlap:
                    continue
                adjacency[left].add(right)
                adjacency[right].add(left)

        # Every accepted edge can seed a dyad. A new member may expand that core
        # only when connected to at least two current members, avoiding transitive chains.
        for left in ids:
            for right in sorted(adjacency[left]):
                if left >= right:
                    continue
                motion_agreement = _co_movement_agrees(
                    velocity_by_frame.get(left, {}).get(frame),
                    velocity_by_frame.get(right, {}).get(frame),
                )
                if motion_agreement is False:
                    continue
                core = {left, right}
                active_members.setdefault(tuple(sorted(core)), []).append(frame)
                expanded = set(core)
                while True:
                    additions = []
                    for candidate in ids:
                        if candidate in expanded:
                            continue
                        if len(adjacency[candidate] & expanded) < 2:
                            continue
                        if not co_moving_with_core(candidate, expanded, frame):
                            continue
                        additions.append(candidate)
                    if not additions:
                        break
                    expanded.update(additions)
                if len(expanded) > 2:
                    active_members.setdefault(tuple(sorted(expanded)), []).append(frame)

    groups = []
    pair_evidence: Dict[Tuple[str, str], dict] = {}
    observed_seconds = {
        visitor_id: _observed_presence_seconds(observations, fps, sample_stride)
        for visitor_id, observations in ordered_observations.items()
    }
    for members, raw_frames in active_members.items():
        all_member_frames = sorted(set(raw_frames))
        total_together_seconds = _duration_from_frames(
            all_member_frames, fps, sample_stride, max_gap_frames
        )
        member_observed = {member: observed_seconds.get(member, 0.0) for member in members}
        member_shares = {
            member: min(1.0, total_together_seconds / max(member_observed[member], 1e-6))
            for member in members
        }
        min_member_share = min(member_shares.values(), default=0.0)
        segments = _split_frame_runs(all_member_frames, max_gap_frames)
        for segment in segments:
            start, end = segment[0], segment[-1]
            duration = max(sample_period, (end - start) / max(fps, 1e-6) + sample_period)
            pair = tuple(sorted(members)) if len(members) == 2 else None
            overlap_in_segment = bool(
                pair and any(frame in qualified_overlap_frames.get(pair, set()) for frame in segment)
            )
            # Sustained overlap has its own adjustable time floor; proximity-only
            # dyads and expanded groups continue to use Minimum together time.
            minimum_duration = (
                settings.group_min_box_overlap_sec if overlap_in_segment
                else settings.group_min_together_sec
            )
            if duration < minimum_duration:
                continue
            common_frames = set.intersection(*[
                {obs.frame for obs in ordered_observations.get(member, []) if start <= obs.frame <= end}
                for member in members
            ]) if members else set()
            close_count = len(set(segment))
            close_ratio = close_count / max(1, len(common_frames))
            if close_ratio < settings.group_min_close_ratio:
                continue
            base_confidence = 0.50 * close_ratio + 0.50 * min(1.0, duration / 6.0)
            confidence = float(np.clip(base_confidence + 0.04 * min(2, len(members) - 2), 0, 1))
            group_id = f"{role_prefix}-G{len(groups) + 1:03d}"
            group = {
                "group_id": group_id,
                "members": list(members),
                "member_count": len(members),
                "start_frame": int(start),
                "end_frame": int(end),
                "start_seconds": round(start / max(fps, 1e-6), 2),
                "end_seconds": round(end / max(fps, 1e-6), 2),
                "together_seconds": round(duration, 2),
                "close_ratio": round(close_ratio, 3),
                "confidence_heuristic": round(confidence, 3),
                "expansion_bonus": round(0.04 * min(2, len(members) - 2), 3),
                "evidence_basis": "sustained_box_overlap" if overlap_in_segment else "image_space_proximity",
                "evidence": (
                    "Expanded from a dyad core; each additional member was near/overlapped at least two existing members, with motion agreement when measurable."
                    if len(members) > 2 else
                    ("Sustained bounding-box overlap and repeated co-presence; relationship is not inferred."
                     if overlap_in_segment else
                     "Repeated close co-presence in image-space; relationship is not inferred.")
                ),
                "_together_frames": list(segment),
                "_member_observed_seconds": member_observed,
            }
            _set_group_time_share_fields(
                group, total_together_seconds, member_observed, settings.group_likely_share_ratio
            )
            groups.append(group)
            if len(members) == 2:
                overlap_frames = [
                    frame for frame in segment
                    if frame in qualified_overlap_frames.get(pair, set())
                ]
                raw_ratios = [
                    overlap_ratios.get(pair, {}).get(frame, 0.0)
                    for frame in segment if frame in overlap_ratios.get(pair, {})
                ]
                pair_evidence[pair] = {
                    "shared_frames_in_interval": len(common_frames),
                    "close_frames": close_count,
                    "close_ratio": round(close_ratio, 3),
                    "together_seconds": round(duration, 2),
                    "together_seconds_total": round(total_together_seconds, 2),
                    "box_overlap_frames": len(overlap_frames),
                    "box_overlap_ratio_mean": round(float(np.mean(raw_ratios)), 3) if raw_ratios else 0.0,
                    "member_observed_seconds": {key: round(value, 2) for key, value in member_observed.items()},
                    "member_together_share": {key: round(value, 4) for key, value in member_shares.items()},
                    "minimum_member_together_share": round(min_member_share, 4),
                    "likely_group_share_threshold": settings.group_likely_share_ratio,
                    "meets_likely_group_share": min_member_share >= settings.group_likely_share_ratio,
                    "qualifies_via_sustained_overlap": overlap_in_segment,
                    "qualifies": True,
                }
    groups.sort(key=lambda group: (group["start_frame"], -group["confidence_heuristic"], -len(group["members"]), group["members"]))
    return groups, pair_evidence, frame_people

def _resolve_exclusive_groups(groups: List[dict], fps: float, sample_stride: int,
                             settings: PipelineSettings, role_prefix: str) -> List[dict]:
    """Keep member preferences sticky; hand off only after sustained, stronger evidence."""
    candidates = sorted(
        [dict(group) for group in groups],
        key=lambda group: (group["start_frame"], -group["confidence_heuristic"], -len(group["members"]), group["members"]),
    )
    accepted: List[dict] = []
    member_preferences: Dict[str, Tuple[Tuple[str, ...], float]] = {}
    sample_period = sample_stride / max(fps, 1e-6)

    def duration(start, end):
        return max(sample_period, (end - start) / max(fps, 1e-6) + sample_period)

    def minimum_group_duration(group):
        if group.get("evidence_basis") == "sustained_box_overlap" and len(group.get("members", [])) == 2:
            return settings.group_min_box_overlap_sec
        return settings.group_min_together_sec

    def refresh(group):
        group["start_seconds"] = round(group["start_frame"] / max(fps, 1e-6), 2)
        group["end_seconds"] = round(group["end_frame"] / max(fps, 1e-6), 2)
        group["together_seconds"] = round(duration(group["start_frame"], group["end_frame"]), 2)
        group["confidence_heuristic"] = round(float(np.clip(
            0.50 * group["close_ratio"] + 0.50 * min(1.0, group["together_seconds"] / 6.0)
            + float(group.get("expansion_bonus", 0.0)), 0, 1
        )), 3)
        if "minimum_member_together_share" in group:
            group["classification"] = (
                "likely_group"
                if group["minimum_member_together_share"] >= settings.group_likely_share_ratio
                else "observed_group"
            )
        else:
            group["classification"] = "likely_group" if group["close_ratio"] >= 0.80 and group["together_seconds"] >= 5.0 else "observed_group"

    for candidate in candidates:
        start, end = candidate["start_frame"], candidate["end_frame"]
        original_start = start
        candidate_members = tuple(sorted(candidate["members"]))
        candidate_score = float(candidate["confidence_heuristic"])
        candidate_duration = float(candidate.get("together_seconds", duration(start, end)))
        previous_preferences = [
            member_preferences[member]
            for member in candidate_members
            if member in member_preferences and member_preferences[member][0] != candidate_members
        ]
        if previous_preferences:
            established_score = max(score for _members, score in previous_preferences)
            sustained = candidate_duration >= settings.group_reassign_min_sec
            materially_stronger = candidate_score >= established_score + settings.group_reassign_margin
            if not sustained or not materially_stronger:
                # Do not create a brief alternative membership and then switch
                # back. Keep the person unassigned rather than changing groups
                # on proximity alone.
                continue

        for current in list(accepted):
            shared_members = set(candidate["members"]) & set(current["members"])
            overlaps = start <= current["end_frame"] and current["start_frame"] <= end
            if not shared_members or not overlaps:
                continue
            clearly_better = candidate_score >= current["confidence_heuristic"] + settings.group_reassign_margin
            if clearly_better:
                # The new candidate wins the contested interval. End the old
                # assignment before this one begins; never double-book a member.
                old_end = min(current["end_frame"], start - sample_stride)
                if duration(current["start_frame"], old_end) < minimum_group_duration(current):
                    accepted.remove(current)
                else:
                    current["end_frame"] = old_end
                    current["assignment_note"] = "Ended early when a sustained, more convincing group candidate appeared."
                    refresh(current)
            else:
                # Keep the established assignment for the full competing interval.
                start = max(start, current["end_frame"] + sample_stride)
        if duration(start, end) < minimum_group_duration(candidate):
            continue
        candidate["start_frame"] = start
        if start > original_start:
            candidate["assignment_note"] = "Assignment delayed until the existing group interval ended."
        else:
            candidate["assignment_note"] = "Sticky exclusive assignment; alternatives need sustained, stronger evidence."
        refresh(candidate)
        accepted.append(candidate)
        final_score = float(candidate["confidence_heuristic"])
        for member in candidate_members:
            previous = member_preferences.get(member)
            if previous is None or previous[0] == candidate_members or final_score >= previous[1] + settings.group_reassign_margin:
                member_preferences[member] = (candidate_members, max(final_score, previous[1]) if previous and previous[0] == candidate_members else final_score)

    # Recompute the cumulative share after exclusivity/trimming, aggregating
    # repeated intervals for the same membership without counting overlaps twice.
    frames_by_members = {}
    observed_by_members = {}
    for group in accepted:
        if "_together_frames" not in group:
            continue
        member_key = tuple(sorted(group["members"]))
        assigned_frames = {
            frame for frame in group["_together_frames"]
            if group["start_frame"] <= frame <= group["end_frame"]
        }
        frames_by_members.setdefault(member_key, set()).update(assigned_frames)
        observed_by_members[member_key] = group.get("_member_observed_seconds", {})
    share_gap_frames = int(round(settings.group_max_gap_sec * fps)) + sample_stride
    for group in accepted:
        member_key = tuple(sorted(group["members"]))
        if member_key in frames_by_members:
            together_seconds = _duration_from_frames(
                frames_by_members[member_key], fps, sample_stride, share_gap_frames
            )
            _set_group_time_share_fields(
                group, together_seconds, observed_by_members.get(member_key, {}),
                settings.group_likely_share_ratio,
            )

    accepted.sort(key=lambda group: (group["start_frame"], group["members"]))
    stable_group_ids = {}
    for group in accepted:
        member_key = tuple(sorted(group["members"]))
        if member_key not in stable_group_ids:
            stable_group_ids[member_key] = f"{role_prefix}-G{len(stable_group_ids) + 1:03d}"
        group["group_id"] = stable_group_ids[member_key]
        group.pop("_together_frames", None)
        group.pop("_member_observed_seconds", None)
    return accepted


def _detect_entry_events(visitor_observations: Dict[str, List[Observation]], fps: float,
                         line_y: float, entry_direction: str, height: int,
                         clip_start_window_sec: float = 1.0):
    """Directional line events with a deadband and stable-side confirmation."""
    line_px = line_y * height
    events, preexisting, ambiguous, late_inside = [], [], [], []

    def stable_side(obs: Observation) -> Optional[str]:
        margin = max(6.0, 0.12 * obs.height)
        if entry_direction == "down":
            if obs.foot[1] < line_px - margin:
                return "outside"
            if obs.foot[1] > line_px + margin:
                return "inside"
        else:
            if obs.foot[1] > line_px + margin:
                return "outside"
            if obs.foot[1] < line_px - margin:
                return "inside"
        return None

    for visitor_id, observations in visitor_observations.items():
        ordered = sorted(observations, key=lambda o: o.frame)
        if not ordered:
            continue
        first = ordered[0]
        initial_side = stable_side(first)
        if initial_side is None:
            ambiguous.append({"visitor_id": visitor_id, "reason": "first_observation_inside_entrance_line_deadband"})
        elif initial_side == "inside":
            record = {
                "visitor_id": visitor_id,
                "first_seen_seconds": round(first.timestamp, 2),
                "status": "already_inside_at_clip_start",
            }
            if first.timestamp <= clip_start_window_sec:
                preexisting.append(record)
            else:
                late_inside.append({**record, "status": "first_detected_inside_after_clip_start"})

        committed_side = initial_side
        pending_side = None
        pending_observations: List[Observation] = []
        for current in ordered[1:]:
            side = stable_side(current)
            if side is None:
                pending_side, pending_observations = None, []
                continue
            if committed_side is None:
                committed_side = side
                continue
            if side == committed_side:
                pending_side, pending_observations = None, []
                continue
            if side != pending_side:
                pending_side, pending_observations = side, [current]
            else:
                pending_observations.append(current)
            # Require two consecutive samples firmly on the new side. This
            # hysteresis suppresses box jitter and line-edge oscillation.
            if len(pending_observations) < 2:
                continue
            direction = "entry" if committed_side == "outside" and side == "inside" else "exit"
            crossing = pending_observations[0]
            events.append({
                "type": direction,
                "visitor_id": visitor_id,
                "frame": crossing.frame,
                "timestamp_seconds": round(crossing.timestamp, 2),
                "line_y_percent": round(line_y * 100, 1),
                "confidence_heuristic": 0.78,
                "evidence": "Footpoint moved between stable sides of the configured line; directional event is an estimate.",
            })
            committed_side = side
            pending_side, pending_observations = None, []
    events.sort(key=lambda event: event["timestamp_seconds"])
    for index, event in enumerate(events, start=1):
        event["event_id"] = f"ENT-{index:04d}"
    return events, preexisting, ambiguous, late_inside


def _zone_name(foot: Tuple[float, float], width: int, height: int, layout: str) -> Optional[str]:
    if layout == "off":
        return None
    x, y = foot
    if layout == "columns":
        return "left" if x < width / 3 else ("center" if x < 2 * width / 3 else "right")
    return ("top" if y < height / 2 else "bottom") + ("_left" if x < width / 2 else "_right")


def _visitor_metrics(visitor_observations: Dict[str, List[Observation]], fps: float,
                     stride: int, width: int, height: int, role: str,
                     zone_layout: str):
    sample_period = stride / max(fps, 1e-6)
    metrics, stop_events, checkout_events = {}, [], []
    for visitor_id, raw_obs in visitor_observations.items():
        obs = sorted(raw_obs, key=lambda o: o.frame)
        if not obs:
            continue
        path_px = 0.0
        normalized_speeds = []
        zone_dwell: Dict[str, float] = {}
        for idx, current in enumerate(obs):
            if idx + 1 < len(obs):
                nxt = obs[idx + 1]
                dt = (nxt.frame - current.frame) / max(fps, 1e-6)
                if 0 < dt <= max(sample_period * 2.0, 0.75):
                    distance = float(np.linalg.norm(np.asarray(nxt.foot) - np.asarray(current.foot)))
                    path_px += distance
                    normalized_speeds.append(distance / max(1.0, 0.5 * (current.height + nxt.height)) / dt)
                    dwell_weight = min(dt, sample_period * 1.5)
                else:
                    dwell_weight = 0.0
            else:
                dwell_weight = sample_period
            if role == "store" and dwell_weight > 0:
                zone = _zone_name(current.foot, width, height, zone_layout)
                if zone:
                    zone_dwell[zone] = zone_dwell.get(zone, 0.0) + dwell_weight
        first, last = obs[0], obs[-1]
        duration = max(0.0, last.timestamp - first.timestamp)
        metrics[visitor_id] = {
            "visitor_id": visitor_id,
            "identity_status": "camera_local_anonymous_estimate",
            "first_seen_seconds": round(first.timestamp, 2),
            "last_seen_seconds": round(last.timestamp, 2),
            "observed_duration_seconds": round(duration, 2),
            "observation_count": len(obs),
            "mean_detection_confidence": round(float(np.mean([o.confidence for o in obs])), 3),
            "path_length_pixels": round(path_px, 1),
            "mean_speed_body_lengths_per_second": round(float(np.mean(normalized_speeds)), 3) if normalized_speeds else 0.0,
            "zone_dwell_seconds": {k: round(v, 2) for k, v in zone_dwell.items() if v >= 0.5},
        }
        # Simple trajectory-stability events. This is a descriptive stop heuristic,
        # not evidence of intent or interest.
        threshold_speed = 0.12  # body lengths / second
        run_start = None
        run_last = None
        low_run = []
        for idx in range(1, len(obs)):
            prev, current = obs[idx - 1], obs[idx]
            dt = (current.frame - prev.frame) / max(fps, 1e-6)
            if dt <= 0 or dt > max(sample_period * 2.0, 0.75):
                if low_run:
                    _append_stop(low_run, visitor_id, stop_events, fps)
                    low_run = []
                continue
            speed = float(np.linalg.norm(np.asarray(current.foot) - np.asarray(prev.foot))) / max(1.0, current.height) / dt
            if speed <= threshold_speed:
                if not low_run:
                    low_run = [prev]
                low_run.append(current)
            else:
                if low_run:
                    _append_stop(low_run, visitor_id, stop_events, fps)
                    low_run = []
        if low_run:
            _append_stop(low_run, visitor_id, stop_events, fps)
        if role == "checkout":
            checkout_events.append({
                "event_id": f"CHK-{len(checkout_events) + 1:04d}",
                "visitor_id": visitor_id,
                "start_seconds": round(first.timestamp, 2),
                "end_seconds": round(last.timestamp, 2),
                "observed_seconds": round(duration, 2),
                "event_type": "visual_checkout_camera_presence",
                "purchase_verified": False,
            })
    return metrics, stop_events, checkout_events


def _append_stop(observations: List[Observation], visitor_id: str, stop_events: list, fps: float):
    if len(observations) < 2:
        return
    start, end = observations[0], observations[-1]
    duration = max(0.0, end.timestamp - start.timestamp)
    if duration < 2.0:
        return
    stop_events.append({
        "visitor_id": visitor_id,
        "start_seconds": round(start.timestamp, 2),
        "end_seconds": round(end.timestamp, 2),
        "duration_seconds": round(duration, 2),
        "location_px": [round(float(np.mean([o.foot[0] for o in observations])), 1),
                        round(float(np.mean([o.foot[1] for o in observations])), 1)],
        "event_type": "low_movement_interval",
    })


def _short_group_id(group_id: str) -> str:
    """Display role-prefixed report IDs (E-G001) as a compact in-video G1."""
    suffix = str(group_id).rsplit("-", 1)[-1]
    if suffix.startswith("G") and suffix[1:].isdigit():
        return f"G{int(suffix[1:])}"
    return str(group_id)


def _track_color(key) -> Tuple[int, int, int]:
    """Stable dark-theme box colors, avoiding the previous green cast."""
    palette = [
        (205, 132, 64),   # blue
        (174, 105, 190),  # lavender
        (80, 132, 226),   # warm coral
        (180, 174, 92),   # slate blue
        (55, 126, 225),   # amber
    ]
    seed = sum(ord(char) for char in str(key))
    return palette[seed % len(palette)]


def _draw_overlay(frame: np.ndarray, records, track_to_visitor: Dict[int, str],
                  frame_idx: int, role: str, groups: list, line_y: float,
                  store_zones: str, occupancy: int, peak: int, visitor_count: int,
                  group_count: int, trajectory_points: Dict[str, List[Tuple[int, Tuple[float, float]]]]):
    visual = frame.copy()
    height, width = visual.shape[:2]
    if role == "entrance":
        y = int(line_y * height)
        cv2.line(visual, (0, y), (width, y), (34, 164, 235), 2)
        cv2.putText(visual, "ENTRY / EXIT LINE", (16, max(24, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (34, 164, 235), 2, cv2.LINE_AA)
    elif role == "store" and store_zones != "off":
        if store_zones == "columns":
            for x in (width // 3, 2 * width // 3):
                cv2.line(visual, (x, 0), (x, height), (34, 164, 235), 1)
        else:
            cv2.line(visual, (width // 2, 0), (width // 2, height), (34, 164, 235), 1)
            cv2.line(visual, (0, height // 2), (width, height // 2), (34, 164, 235), 1)
    active_group = {}
    active_groups = []
    group_boxes = {}
    for group in groups:
        if group["start_frame"] <= frame_idx <= group["end_frame"]:
            active_groups.append(group)
            for member in group["members"]:
                active_group[member] = group
    for track_id, bbox in records:
        visitor_id = track_to_visitor.get(track_id)
        color = _track_color(visitor_id if visitor_id else track_id)
        x1, y1, x2, y2 = bbox
        cv2.rectangle(visual, (x1, y1), (x2, y2), color, 2)
        label = f"T{track_id:03d} → {visitor_id}" if visitor_id else f"T{track_id:03d} · pending"
        if visitor_id in active_group:
            group = active_group[visitor_id]
            label += f" ({_short_group_id(group['group_id'])})"
            group_boxes.setdefault(group["group_id"], []).append(bbox)
        text_y = max(20, y1 - 8)
        cv2.putText(visual, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        if visitor_id:
            trajectory = trajectory_points.get(visitor_id, [])
            end_index = bisect_right(trajectory, (frame_idx, (float("inf"), float("inf"))))
            points = [point for _frame, point in trajectory[max(0, end_index - 28):end_index]]
            for p1, p2 in zip(points, points[1:]):
                cv2.line(visual, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, 2)
    # A separate group enclosure makes the group assignment visible, rather
    # than relying only on a small G-label beside each person's box.
    group_color = (34, 164, 235)  # amber in BGR
    for group in active_groups:
        boxes = group_boxes.get(group["group_id"], [])
        if len(boxes) < 2:
            continue
        gx1 = max(0, min(box[0] for box in boxes) - 8)
        gy1 = max(0, min(box[1] for box in boxes) - 28)
        gx2 = min(width - 1, max(box[2] for box in boxes) + 8)
        gy2 = min(height - 1, max(box[3] for box in boxes) + 8)
        cv2.rectangle(visual, (gx1, gy1), (gx2, gy2), group_color, 3)
        group_label = f"{_short_group_id(group['group_id'])} · GROUP ({group['member_count']})"
        (label_w, label_h), _ = cv2.getTextSize(group_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        label_top = max(0, gy1 - label_h - 8)
        cv2.rectangle(visual, (gx1, label_top), (min(width - 1, gx1 + label_w + 10), gy1), group_color, -1)
        cv2.putText(visual, group_label, (gx1 + 5, max(label_h + 2, gy1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 22, 24), 2, cv2.LINE_AA)
    # Keep the frame clean of report-style HUD text; individual track/group
    # labels and the separate group enclosure remain visible for visual review.
    return visual


def _draw_live_preview(frame: np.ndarray, records: list, role: str, frame_idx: int,
                       occupancy: int, line_y: float, zone_layout: str,
                       group_candidates: Optional[list] = None,
                       group_assignments: Optional[dict] = None) -> np.ndarray:
    """Current-frame debug preview; tracker IDs are provisional until final reconciliation."""
    visual = frame.copy()
    height, width = visual.shape[:2]
    if role == "entrance":
        y = int(line_y * height)
        cv2.line(visual, (0, y), (width, y), (34, 164, 235), 2)
    elif role == "store" and zone_layout == "columns":
        cv2.line(visual, (width // 3, 0), (width // 3, height), (34, 164, 235), 1)
        cv2.line(visual, (2 * width // 3, 0), (2 * width // 3, height), (34, 164, 235), 1)
    live_group_by_track = {
        track_id: assignment.get("display_group_id", assignment["group_id"])
        for track_id, assignment in (group_assignments or {}).items()
    }
    if not group_assignments:
        for candidate in sorted(group_candidates or [], key=lambda item: -item.get("duration_seconds", 0.0)):
            candidate_label = candidate.get("display_group_id", candidate.get("group_id", "G?"))
            for track_id in candidate.get("track_ids", []):
                live_group_by_track.setdefault(track_id, candidate_label)
    for track_id, bbox in records:
        color = _track_color(track_id)
        x1, y1, x2, y2 = bbox
        cv2.rectangle(visual, (x1, y1), (x2, y2), color, 2)
        group_tag = f" ({live_group_by_track[track_id]})" if track_id in live_group_by_track else ""
        cv2.putText(visual, f"T{track_id:03d}{group_tag} · tracking", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    group_color = (34, 164, 235)
    for candidate in group_candidates or []:
        boxes = candidate.get("boxes", [])
        if len(boxes) < 2:
            continue
        x1 = max(0, min(box[0] for box in boxes) - 8)
        y1 = max(0, min(box[1] for box in boxes) - 28)
        x2 = min(width - 1, max(box[2] for box in boxes) + 8)
        y2 = min(height - 1, max(box[3] for box in boxes) + 8)
        cv2.rectangle(visual, (x1, y1), (x2, y2), group_color, 3)
        display_id = candidate.get("display_group_id", candidate.get("group_id", "G?"))
        shares = list(candidate.get("member_together_share", {}).values())
        minimum_share = min(shares, default=0.0)
        threshold = float(candidate.get("likely_group_share_threshold", 0.40))
        status = "LIKELY SO FAR" if candidate.get("likely_group_so_far") else "PENDING SHARE"
        label = f"{display_id} · {status} · {minimum_share:.0%}/{threshold:.0%} · {candidate['duration_seconds']:.1f}s"
        cv2.putText(visual, label, (x1, max(18, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, group_color, 2, cv2.LINE_AA)
    # No role/title/HUD text is baked into the frame; tracker and candidate
    # labels remain so the live preview can still be inspected.
    return cv2.cvtColor(visual, cv2.COLOR_BGR2RGB)


def _apply_live_preview_zoom(rgb_frame: np.ndarray, target: Optional[str],
                             records: list, group_candidates: list,
                             group_assignments: Optional[dict] = None) -> np.ndarray:
    """Crop/upscale a UI preview only; the source and annotated MP4 are unchanged."""
    if not target:
        return rgb_frame
    normalized = str(target).strip().upper()
    boxes = []
    if normalized.startswith("T") and normalized[1:].isdigit():
        track_id = int(normalized[1:])
        boxes = [bbox for current_id, bbox in records if int(current_id) == track_id]
    elif normalized.startswith(("G", "P")) and normalized[1:].isdigit():
        candidate = next((
            item for item in group_candidates
            if normalized in {
                str(item.get("group_id", "")).upper(),
                str(item.get("display_group_id", "")).upper(),
            }
        ), None)
        if candidate:
            boxes = list(candidate.get("boxes", []))
        else:
            assigned_tracks = [
                track_id for track_id, assignment in (group_assignments or {}).items()
                if normalized in {
                    str(assignment.get("group_id", "")).upper(),
                    str(assignment.get("display_group_id", "")).upper(),
                }
            ]
            boxes = [bbox for track_id, bbox in records if int(track_id) in assigned_tracks]
    if not boxes:
        return rgb_frame

    height, width = rgb_frame.shape[:2]
    x1 = max(0, min(box[0] for box in boxes))
    y1 = max(0, min(box[1] for box in boxes))
    x2 = min(width, max(box[2] for box in boxes))
    y2 = min(height, max(box[3] for box in boxes))
    box_width, box_height = max(1, x2 - x1), max(1, y2 - y1)
    aspect = width / max(height, 1)
    crop_height = min(height, max(box_height * 2.4, height / 2.8))
    crop_width = min(width, max(box_width * 2.4, width / 2.8))
    if crop_width / max(crop_height, 1) < aspect:
        crop_width = min(width, crop_height * aspect)
    else:
        crop_height = min(height, crop_width / max(aspect, 1e-6))
    center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    left = int(np.clip(center_x - crop_width / 2.0, 0, max(0, width - crop_width)))
    top = int(np.clip(center_y - crop_height / 2.0, 0, max(0, height - crop_height)))
    right = min(width, int(round(left + crop_width)))
    bottom = min(height, int(round(top + crop_height)))
    crop = rgb_frame[top:bottom, left:right]
    if crop.size == 0:
        return rgb_frame
    return cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)


def _write_track_csv(path: Path, tracklets: List[Tracklet], track_to_visitor: Dict[int, str],
                     fps: float, min_track_sec: float):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["tracklet_id", "source_tracker_id", "visitor_id", "start_seconds", "end_seconds",
                         "duration_seconds", "observations", "mean_detection_confidence", "track_status"])
        for track in sorted(tracklets, key=lambda t: t.first_frame):
            duration = (track.last_frame - track.first_frame) / max(fps, 1e-6)
            visitor = track_to_visitor.get(track.track_id, "")
            status = "short_track" if not visitor else "part_of_camera_local_identity"
            writer.writerow([f"T{track.track_id:04d}",
                             track.source_tracker_id if track.source_tracker_id is not None else track.track_id, visitor,
                             round(track.first_frame / fps, 3), round(track.last_frame / fps, 3),
                             round(duration, 3), len(track.observations), round(track.mean_confidence, 4), status])


def _update_live_group_candidates(current_tracks: List[Tracklet], frame_idx: int, fps: float,
                                  sample_stride: int, settings: PipelineSettings,
                                  pair_runs: dict, group_ids: Optional[dict] = None,
                                  member_assignments: Optional[dict] = None,
                                  pair_together_time: Optional[dict] = None) -> list:
    """Build provisional live pairs from proximity or sustained box overlap."""
    candidates = []
    group_ids = group_ids if group_ids is not None else {}
    member_assignments = member_assignments if member_assignments is not None else {}
    pair_together_time = pair_together_time if pair_together_time is not None else {}
    excluded_temp_ids = {
        int(value[1:]) for value in settings.exclude_group_visitor_ids
        if len(value) > 1 and value[0].upper() == "T" and value[1:].isdigit()
    }
    sample_period = sample_stride / max(fps, 1e-6)
    max_gap_frames = int(round(settings.group_max_gap_sec * fps)) + sample_stride

    def latest_velocity(track):
        if len(track.observations) < 2:
            return None
        current = track.observations[-1]
        previous = track.observations[-2]
        for candidate_previous in reversed(track.observations[:-2]):
            elapsed_candidate = current.timestamp - candidate_previous.timestamp
            frame_gap = current.frame - candidate_previous.frame
            if elapsed_candidate <= 0.35 and frame_gap <= max_gap_frames:
                previous = candidate_previous
            else:
                break
        elapsed = current.timestamp - previous.timestamp
        if elapsed <= 0:
            return None
        return (np.asarray(current.foot, dtype=np.float32) - np.asarray(previous.foot, dtype=np.float32)) / (elapsed * max(current.height, 1.0))

    def update_run(run, condition):
        if not condition:
            return None  # Both people were observed now, so this is a real break.
        if run is None or frame_idx - run["last_frame"] > max_gap_frames:
            return {"start_frame": frame_idx, "last_frame": frame_idx}
        run["last_frame"] = frame_idx
        return run

    for left, right in combinations(current_tracks, 2):
        if left.track_id in excluded_temp_ids or right.track_id in excluded_temp_ids:
            continue
        a, b = left.observations[-1], right.observations[-1]
        key = tuple(sorted((left.track_id, right.track_id)))
        distance = float(np.linalg.norm(np.asarray(a.foot) - np.asarray(b.foot)))
        proximity = distance / _group_pixel_scale(a.height, b.height) <= settings.group_proximity_heights
        overlap_ratio = _box_overlap_ratio(a.bbox, b.bbox)
        overlap = overlap_ratio >= settings.group_box_overlap_min_ratio

        state = pair_runs.get(key, {"proximity": None, "overlap": None, "last_seen_frame": frame_idx})
        state["last_seen_frame"] = frame_idx
        state["proximity"] = update_run(state.get("proximity"), proximity)
        state["overlap"] = update_run(state.get("overlap"), overlap)
        if state["proximity"] is None and state["overlap"] is None:
            pair_runs.pop(key, None)
            continue
        pair_runs[key] = state

        motion_agreement = _co_movement_agrees(latest_velocity(left), latest_velocity(right))
        if motion_agreement is False:
            pair_runs.pop(key, None)
            continue
        if proximity or overlap:
            pair_together_time[key] = pair_together_time.get(key, 0.0) + sample_period

        qualifying_durations = []
        if proximity and state["proximity"] is not None:
            proximity_duration = max(
                sample_period,
                (frame_idx - state["proximity"]["start_frame"]) / max(fps, 1e-6) + sample_period,
            )
            required_proximity_time = (
                settings.group_min_together_sec if motion_agreement is True
                else max(6.0, 2.0 * settings.group_min_together_sec)
            )
            if proximity_duration >= required_proximity_time:
                qualifying_durations.append((proximity_duration, "image_space_proximity"))
        if overlap and state["overlap"] is not None:
            overlap_duration = max(
                sample_period,
                (frame_idx - state["overlap"]["start_frame"]) / max(fps, 1e-6) + sample_period,
            )
            if overlap_duration >= settings.group_min_box_overlap_sec:
                qualifying_durations.append((overlap_duration, "sustained_box_overlap"))
        if not qualifying_durations:
            continue

        duration, evidence_basis = max(qualifying_durations, key=lambda item: item[0])
        group_id = group_ids.setdefault(key, f"P{len(group_ids) + 1}")
        display_group_id = f"G{int(group_id[1:])}" if group_id.startswith("P") and group_id[1:].isdigit() else group_id
        together_seconds = pair_together_time.get(key, duration)
        observed_left = _observed_presence_seconds(left.observations, fps, sample_stride)
        observed_right = _observed_presence_seconds(right.observations, fps, sample_stride)
        share_left = min(1.0, together_seconds / max(observed_left, 1e-6))
        share_right = min(1.0, together_seconds / max(observed_right, 1e-6))
        likely_so_far = min(share_left, share_right) >= settings.group_likely_share_ratio
        # The visible G# is provisional until the offline pass measures the
        # complete visit; the candidate status is updated as the share changes.
        score = 0.50 + 0.50 * min(1.0, duration / 6.0)
        can_assign = True
        for track_id in key:
            current = member_assignments.get(track_id)
            if current and current["group_id"] != group_id:
                sustained = duration >= settings.group_reassign_min_sec
                stronger = score >= current["score"] + settings.group_reassign_margin
                if not sustained or not stronger:
                    can_assign = False
                    break
        if not can_assign:
            continue
        for track_id in key:
            current = member_assignments.get(track_id)
            if current and current["group_id"] == group_id:
                current["score"] = max(current["score"], score)
                current["display_group_id"] = display_group_id
                current["likely_group_so_far"] = likely_so_far
            else:
                member_assignments[track_id] = {
                    "group_id": group_id, "display_group_id": display_group_id,
                    "likely_group_so_far": likely_so_far, "score": score,
                }
        candidates.append({
            "group_id": group_id,
            "display_group_id": display_group_id,
            "track_ids": list(key),
            "boxes": [a.bbox, b.bbox],
            "duration_seconds": duration,
            "evidence_basis": evidence_basis,
            "box_overlap_ratio": round(overlap_ratio, 3),
            "together_seconds_total": round(together_seconds, 2),
            "observed_seconds": {str(left.track_id): round(observed_left, 2), str(right.track_id): round(observed_right, 2)},
            "member_together_share": {str(left.track_id): round(share_left, 4), str(right.track_id): round(share_right, 4)},
            "likely_group_so_far": likely_so_far,
            "likely_group_share_threshold": settings.group_likely_share_ratio,
            "confidence_heuristic": score,
        })

    for key, state in list(pair_runs.items()):
        if frame_idx - state.get("last_seen_frame", frame_idx) > max_gap_frames:
            pair_runs.pop(key, None)
    return candidates

def _make_appearance_crops(frame: np.ndarray, detections) -> List[np.ndarray]:
    height, width = frame.shape[:2]
    crops = []
    for item in detections:
        x1, y1, x2, y2 = item[0]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        crops.append(frame[y1:y2, x1:x2].copy())
    return crops


def _frame_stride(source_fps: float, frame_skip: int,
                  legacy_analysis_fps: Optional[float] = None) -> int:
    """Convert the UI's number-of-frames-to-skip setting into a sample stride."""
    if legacy_analysis_fps is not None:
        return max(1, int(round(source_fps / max(legacy_analysis_fps, 1e-6))))
    return max(1, int(frame_skip) + 1)


def _load_reid_extractor(settings: PipelineSettings, progress=None):
    if settings.reid_backend == "hsv":
        return None, None
    if progress:
        progress(0.015, "Checking optional OSNet ReID encoder")
    try:
        from osnet_reid import OSNetReID
        extractor = OSNetReID(allow_download=(settings.reid_backend == "osnet"))
        return extractor, None
    except Exception as exc:
        if settings.reid_backend == "osnet":
            message = f"OSNet was requested but could not be initialized ({exc}); using the HSV fallback."
        else:
            message = f"Cached OSNet weights/dependencies were unavailable ({exc}); using the HSV fallback. Choose OSNet ReID in settings to download/load the optional weights."
        return None, message


def analyze_video(video_path: str, role: str, output_dir: str, raw_settings: Optional[dict] = None,
                  progress: Optional[Callable[[float, str], None]] = None,
                  preview_callback: Optional[Callable[[np.ndarray], None]] = None,
                  camera_id: Optional[str] = None, camera_name: Optional[str] = None,
                  preview_zoom_getter: Optional[Callable[[], Optional[str]]] = None) -> dict:
    """Analyze one recorded video and write a final annotated MP4 + track CSV."""
    if role not in ROLE_PREFIX:
        raise ValueError(f"Unsupported camera role: {role}")
    camera_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(camera_id or role)).strip("_")[:80] or role
    camera_name = str(camera_name or ROLE_LABEL[role])[:120]
    raw_settings = raw_settings or {}
    settings = PipelineSettings.from_dict(raw_settings)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    source_path = Path(video_path)
    if source_path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported video extension: {source_path.suffix}")
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open uploaded video: {source_path.name}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps < 1.0 or fps > 240.0:
        fps = 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Video metadata did not include a valid frame size.")
    stride = _frame_stride(fps, settings.frame_skip, settings.analysis_fps)
    actual_analysis_fps = fps / stride
    detector = PersonDetector(settings)
    reid_extractor, reid_warning = _load_reid_extractor(settings, progress)
    tracker = TrackletManager(fps, settings)
    frame_records: Dict[int, list] = {}
    live_pair_runs = {}
    live_group_ids = {}
    live_member_assignments = {}
    live_pair_together_time = {}
    occupancy_timeline = []
    peak_occupancy = 0
    processed_samples = 0
    frame_idx = 0
    preview_sample_rate = max(1, int(round(actual_analysis_fps * 0.5)))
    appearance_sample_rate = max(1, int(round(actual_analysis_fps / 5.0)))
    start_clock = time.time()
    if progress:
        progress(0.02, f"Loading {ROLE_LABEL[role]} video ({width}×{height}, {fps:.1f} FPS)")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % stride == 0:
                timestamp = frame_idx / fps
                detections = detector.detect(frame)
                appearance_features = None
                if reid_extractor is not None:
                    if (processed_samples + 1) % appearance_sample_rate == 0 and detections:
                        appearance_features = reid_extractor.extract(_make_appearance_crops(frame, detections))
                    else:
                        appearance_features = [None] * len(detections)
                tracker.update(frame, frame_idx, timestamp, detections, appearance_features)
                current_records = []
                current_track_objs = []
                for track in tracker.tracks.values():
                    if track.observations and track.last_frame == frame_idx:
                        current_records.append((track.track_id, track.observations[-1].bbox))
                        current_track_objs.append(track)
                frame_records[frame_idx] = current_records
                live_group_candidates = (
                    _update_live_group_candidates(
                        current_track_objs, frame_idx, fps, stride, settings,
                        live_pair_runs, live_group_ids, live_member_assignments,
                        live_pair_together_time,
                    )
                    if settings.detect_groups else []
                )
                occupancy = len(current_records)
                peak_occupancy = max(peak_occupancy, occupancy)
                if not occupancy_timeline or timestamp - occupancy_timeline[-1]["seconds"] >= 1.0:
                    occupancy_timeline.append({"seconds": round(timestamp, 2), "people_detected": occupancy})
                processed_samples += 1
                if preview_callback and processed_samples % preview_sample_rate == 0:
                    try:
                        preview_image = _draw_live_preview(
                            frame, current_records, role, frame_idx, occupancy,
                            settings.entrance_line_y, settings.zone_layout,
                            group_candidates=live_group_candidates,
                            group_assignments=live_member_assignments,
                        )
                        if preview_zoom_getter:
                            preview_image = _apply_live_preview_zoom(
                                preview_image, preview_zoom_getter(), current_records,
                                live_group_candidates, live_member_assignments,
                            )
                        # Keep the frequently refreshed UI preview responsive on
                        # high-resolution clips; this is not used by the export.
                        max_dimension = max(preview_image.shape[:2])
                        if max_dimension > 1280:
                            scale = 1280.0 / max_dimension
                            preview_image = cv2.resize(
                                preview_image,
                                (max(1, int(round(preview_image.shape[1] * scale))),
                                 max(1, int(round(preview_image.shape[0] * scale)))),
                                interpolation=cv2.INTER_AREA,
                            )
                        preview_callback(preview_image)
                    except Exception:
                        # A preview widget failure should not discard the analysis.
                        pass
            frame_idx += 1
            if progress and frame_idx % max(1, int(fps * 2)) == 0:
                fraction = min(0.72, frame_idx / max(frame_count, frame_idx, 1) * 0.72)
                progress(fraction, f"Detecting and tracking · frame {frame_idx:,}" + (f" / {frame_count:,}" if frame_count else ""))
    finally:
        cap.release()
    if frame_idx == 0:
        raise RuntimeError("The uploaded file contains no decodable frames.")
    tracks = list(tracker.tracks.values())
    appearance_threshold = settings.reid_match_threshold if reid_extractor is not None else max(0.94, settings.reid_match_threshold)
    components, accepted_links, link_audit = _union_find_reconcile(
        tracks, fps, settings, appearance_threshold=appearance_threshold
    )
    eligible_track_ids = {
        track.track_id for track in tracks
        if len(track.observations) >= settings.min_track_observations
        and (track.last_frame - track.first_frame) / max(fps, 1e-6) >= settings.min_track_sec
        and track.mean_confidence >= settings.min_track_confidence
    }
    usable_components = []
    for component in components:
        usable = [track_id for track_id in component if track_id in eligible_track_ids]
        if usable:
            usable_components.append(usable)
    usable_components.sort(key=lambda component: min(next(t.first_frame for t in tracks if t.track_id == tid) for tid in component))
    track_to_visitor: Dict[int, str] = {}
    visitor_observations: Dict[str, List[Observation]] = {}
    visitor_features: Dict[str, Optional[np.ndarray]] = {}
    visitor_tracklets: Dict[str, list] = {}
    prefix = ROLE_PREFIX[role]
    for idx, component in enumerate(usable_components, start=1):
        visitor_id = f"{prefix}-V{idx:03d}"
        visitor_tracklets[visitor_id] = sorted(component)
        merged_observations = []
        features = []
        for track_id in component:
            track = next(t for t in tracks if t.track_id == track_id)
            track_to_visitor[track_id] = visitor_id
            merged_observations.extend(track.observations)
            if track.feature is not None:
                features.append(track.feature)
        # Components are only linked where time does not conflict; remove any
        # duplicate sample frame defensively, retaining the stronger observation.
        by_frame = {}
        for obs in merged_observations:
            prior = by_frame.get(obs.frame)
            if prior is None or obs.confidence > prior.confidence:
                by_frame[obs.frame] = obs
        visitor_observations[visitor_id] = [by_frame[f] for f in sorted(by_frame)]
        if features:
            feature = np.mean(features, axis=0).astype(np.float32)
            if np.any(feature < -1e-6):
                feature /= max(float(np.linalg.norm(feature)), 1e-8)
            else:
                feature /= max(float(feature.sum()), 1e-8)
            visitor_features[visitor_id] = feature
        else:
            visitor_features[visitor_id] = None
    duration_sec = frame_idx / fps
    stationary_candidates = _stationary_role_candidates(
        visitor_observations, duration_sec, actual_analysis_fps, settings
    )
    manual_group_exclusion_tokens = set(settings.exclude_group_visitor_ids)
    manually_excluded_group_ids = set()
    unmatched_group_exclusion_tokens = []
    for token in manual_group_exclusion_tokens:
        if len(token) > 1 and token[0].upper() == "T" and token[1:].isdigit():
            visitor_id = track_to_visitor.get(int(token[1:]))
            if visitor_id:
                manually_excluded_group_ids.add(visitor_id)
            else:
                unmatched_group_exclusion_tokens.append(token)
        elif token in visitor_observations:
            manually_excluded_group_ids.add(token)
        else:
            unmatched_group_exclusion_tokens.append(token)
    likely_fixed_post_ids = {
        visitor_id for visitor_id, candidate in stationary_candidates.items()
        if candidate.get("role_candidate") == "likely_fixed_post_candidate"
    }
    role_groups, pair_evidence = [], {}
    if settings.detect_groups:
        group_observations = visitor_observations
        if settings.exclude_stationary_from_groups or manually_excluded_group_ids:
            group_observations = {
                visitor_id: observations
                for visitor_id, observations in visitor_observations.items()
                if (not settings.exclude_stationary_from_groups or visitor_id not in likely_fixed_post_ids)
                and visitor_id not in manually_excluded_group_ids
            }
        role_groups, pair_evidence, _frame_people = _close_group_frames(
            group_observations, settings, fps, stride, prefix
        )
        role_groups = _resolve_exclusive_groups(role_groups, fps, stride, settings, prefix)
        resolved_pair_groups = {
            tuple(sorted(group["members"])): group
            for group in role_groups if len(group["members"]) == 2
        }
        accepted_pairs = set(resolved_pair_groups)
        for pair, evidence in pair_evidence.items():
            pair_key = tuple(sorted(pair))
            evidence["accepted_after_exclusivity"] = pair_key in accepted_pairs
            resolved_group = resolved_pair_groups.get(pair_key)
            if resolved_group:
                for field in (
                    "together_seconds_total", "member_observed_seconds", "member_together_share",
                    "minimum_member_together_share", "minimum_member_together_share_percent",
                    "likely_group_share_threshold", "meets_likely_group_share", "classification",
                ):
                    if field in resolved_group:
                        evidence[field] = resolved_group[field]
    visitor_metrics, stop_events, checkout_events = _visitor_metrics(
        visitor_observations, fps, stride, width, height, role, settings.zone_layout
    )
    entry_events, preexisting, ambiguous, late_inside = ([], [], [], [])
    if role == "entrance":
        entry_events, preexisting, ambiguous, late_inside = _detect_entry_events(
            visitor_observations, fps, settings.entrance_line_y,
            settings.entry_direction, height
        )
    # Group membership is interval-based. Stationary staff-like patterns are
    # flagged for review and can be excluded from group formation by setting.
    for visitor_id, metrics in visitor_metrics.items():
        metrics["role_candidate"] = stationary_candidates.get(visitor_id, {}).get(
            "role_candidate", "unclassified_customer_or_staff"
        )
        if visitor_id in manually_excluded_group_ids:
            metrics["role_candidate"] = "manually_excluded_from_groups"
        metrics["group_intervals"] = sorted({
            group["group_id"] for group in role_groups if visitor_id in group["members"]
        })
    for stop_idx, event in enumerate(stop_events, start=1):
        event["event_id"] = f"STOP-{stop_idx:04d}"
    detector_warnings = [detector.warning] if detector.warning else []
    if detector.backend == "opencv_hog" and detector.warning is None:
        detector_warnings.append(
            "OpenCV HOG fallback was selected. It is less reliable than a modern person detector, particularly for small, overhead, occluded, or crowded people."
        )
    if reid_warning:
        detector_warnings.append(reid_warning)
    distinct_group_memberships = len({tuple(sorted(group["members"])) for group in role_groups})
    report = {
        "schema_version": "2.1",
        "pipeline_version": "2.1.0",
        "camera_role": role,
        "camera_id": camera_id,
        "camera_name": camera_name,
        "camera_label": ROLE_LABEL[role],
        "source_filename": source_path.name,
        "analysis_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "video": {
            "width": width,
            "height": height,
            "source_fps": round(fps, 3),
            "analysis_fps": round(actual_analysis_fps, 3),
            "frame_skip": stride - 1,
            "frames_processed": frame_idx,
            "sampled_frames": processed_samples,
            "duration_seconds": round(duration_sec, 2),
        },
        "detector": {
            "backend": detector.backend,
            "tracker_backend": getattr(detector, "tracker_backend", "test_or_custom_tracker"),
            "model": settings.detector_model if detector.backend == "ultralytics_yolo" else None,
            "confidence_setting": settings.confidence,
            "confidence_note": "Detector scores are not calibrated probabilities; OpenCV HOG scores are only ranking proxies.",
        },
        "reid": {
            "backend": getattr(reid_extractor, "backend", "hsv_clothing_histogram"),
            "match_threshold": settings.reid_match_threshold if reid_extractor is not None else appearance_threshold,
            "confidence_calibrated": False,
            "note": "Appearance is an anonymous association cue only; never a confirmed physical identity.",
        },
        "warnings": detector_warnings,
        "identity_policy": {
            "scope": "camera_local_anonymous_estimates",
            "tracklets_reconciled_offline": True,
            "appearance_signal": ("OSNet MSMT17 learned pedestrian embedding" if reid_extractor is not None else "coarse HSV clothing histogram; not face recognition and not learned ReID"),
            "automatic_cross_camera_merge": False,
            "confidence_calibrated": False,
            "note": "A person seen in multiple uploaded camera videos can be counted once per camera. Cross-camera candidate links do not merge these IDs.",
        },
        "group_settings": {
            "group_detection_enabled": settings.detect_groups,
            "proximity_scene_scale_ratio": settings.group_proximity_heights,
            "minimum_together_seconds": settings.group_min_together_sec,
            "minimum_continuous_box_overlap_seconds": settings.group_min_box_overlap_sec,
            "minimum_box_overlap_ratio": settings.group_box_overlap_min_ratio,
            "likely_group_together_share_ratio": settings.group_likely_share_ratio,
            "minimum_close_ratio": settings.group_min_close_ratio,
            "max_gap_seconds": settings.group_max_gap_sec,
            "reassignment_margin": settings.group_reassign_margin,
            "alternative_min_together_seconds": settings.group_reassign_min_sec,
            "manually_excluded_group_visitor_ids": sorted(manually_excluded_group_ids),
            "exclusive_temporal_membership": True,
            "exclude_likely_fixed_post_candidates": settings.exclude_stationary_from_groups,
            "automatically_excluded_fixed_post_ids": sorted(likely_fixed_post_ids) if settings.detect_groups and settings.exclude_stationary_from_groups else [],
            "distance_note": "Bounding-box height is only an image-space perspective scale proxy; box-overlap ratio is intersection area divided by the smaller box area. No biological height is inferred.",
            "together_share_denominator": "Each member's total observed in-camera time, estimated as the sum of visible track intervals with short gaps bridged.",
            "group_meaning": "Temporary observed proximity, box overlap, and co-presence only; no family or relationship inference.",
        },
        "summary": {
            "camera_local_visitors_estimated": len(visitor_observations),
            "tracklets": len(tracks),
            "short_or_unconfirmed_tracks": sum(1 for t in tracks if t.track_id not in eligible_track_ids),
            "tracklet_links_accepted": len(accepted_links),
            "group_intervals": len(role_groups),
            "distinct_group_memberships": distinct_group_memberships,
            "likely_group_intervals": sum(1 for group in role_groups if group["classification"] == "likely_group"),
            "stationary_role_candidates": len(stationary_candidates),
            "peak_detected_occupancy": peak_occupancy,
            "entry_events": sum(1 for e in entry_events if e["type"] == "entry"),
            "exit_events": sum(1 for e in entry_events if e["type"] == "exit"),
            "preexisting_occupants_at_clip_start": len(preexisting),
            "first_detected_inside_after_clip_start": len(late_inside),
            "ambiguous_entrance_tracks": len(ambiguous),
            "low_movement_intervals": len(stop_events),
            "checkout_camera_presence_events": len(checkout_events),
            "payments_verified": 0,
        },
        "tracklets": [
            {
                "tracklet_id": f"T{track.track_id:04d}",
                "source_tracker_id": track.source_tracker_id if track.source_tracker_id is not None else track.track_id,
                "analysis_tracklet_id": f"T{track.track_id:04d}",
                "visitor_id": track_to_visitor.get(track.track_id),
                "start_frame": track.first_frame,
                "end_frame": track.last_frame,
                "start_seconds": round(track.first_frame / fps, 2),
                "end_seconds": round(track.last_frame / fps, 2),
                "duration_seconds": round((track.last_frame - track.first_frame) / fps, 2),
                "observations": len(track.observations),
                "mean_detection_confidence": round(track.mean_confidence, 3),
                "status": "camera_local_identity_tracklet" if track.track_id in track_to_visitor else "short_track_unconfirmed",
            }
            for track in sorted(tracks, key=lambda t: t.first_frame)
        ],
        "tracklet_link_audit": link_audit[:500],
        "accepted_tracklet_links": accepted_links,
        "visitors": list(visitor_metrics.values()),
        "stationary_role_candidates": list(stationary_candidates.values()),
        "groups": role_groups,
        "group_assignment_policy": {
            "group_detection_enabled": settings.detect_groups,
            "reassignment_margin": settings.group_reassign_margin,
            "alternative_min_together_seconds": settings.group_reassign_min_sec,
            "manually_excluded_visitor_ids": sorted(manually_excluded_group_ids),
            "manual_exclusion_inputs": sorted(manual_group_exclusion_tokens),
            "unmatched_manual_exclusion_inputs": sorted(unmatched_group_exclusion_tokens),
            "note": (
                "Group detection is disabled for this feed; visitor tracking and other metrics continue."
                if not settings.detect_groups else
                "An existing group preference is retained unless a competing group is sustained and materially stronger."
            )
        },
        "pair_evidence": [
            {"pair": list(pair), **evidence}
            for pair, evidence in sorted(pair_evidence.items())
        ],
        "entrance_events": entry_events,
        "preexisting_occupants": preexisting,
        "first_detected_inside_after_clip_start": late_inside,
        "ambiguous_entrance_tracks": ambiguous,
        "stop_events": stop_events,
        "checkout_presence_events": checkout_events,
        "occupancy_timeline": occupancy_timeline,
    }
    # A second pass can render the final offline visitor and group IDs without
    # holding source frames in RAM.
    annotated_name = f"{camera_id}_annotated.mp4"
    annotated_path = output_path / annotated_name
    if progress:
        progress(0.75, f"Finalizing identities and rendering {ROLE_LABEL[role]} annotation")
    _render_final_video(
        str(source_path), str(annotated_path), fps, frame_records,
        track_to_visitor, visitor_observations, role_groups, role,
        settings, width, height, peak_occupancy, len(visitor_observations),
    )
    tracks_csv = output_path / f"{camera_id}_tracklets.csv"
    _write_track_csv(tracks_csv, tracks, track_to_visitor, fps, settings.min_track_sec)
    report["outputs"] = {"annotated_video": annotated_name, "tracklets_csv": tracks_csv.name}
    report["processing_seconds"] = round(time.time() - start_clock, 2)
    if progress:
        progress(1.0, f"Finished {ROLE_LABEL[role]}")
    return {"report": report, "visitor_features": visitor_features}


def _render_final_video(source: str, destination: str, fps: float, frame_records: Dict[int, list],
                        track_to_visitor: Dict[int, str], visitor_observations: Dict[str, List[Observation]],
                        groups: list, role: str, settings: PipelineSettings, width: int, height: int,
                        peak: int, visitor_count: int):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError("Could not reopen source video for annotation render.")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(destination, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not initialize MP4 output writer. Check OpenCV video codec support.")
    trajectory_points = {
        visitor_id: [(o.frame, o.foot) for o in observations]
        for visitor_id, observations in visitor_observations.items()
    }
    idx, last_records = 0, []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx in frame_records:
                last_records = frame_records[idx]
            occupancy = len(last_records)
            annotated = _draw_overlay(
                frame, last_records, track_to_visitor, idx, role, groups,
                settings.entrance_line_y, settings.zone_layout,
                occupancy, peak, visitor_count, len(groups), trajectory_points,
            )
            writer.write(annotated)
            idx += 1
    finally:
        cap.release()
        writer.release()
    if not Path(destination).exists() or Path(destination).stat().st_size < 1000:
        raise RuntimeError("Annotated video output was empty; the source codec may not be supported.")


def build_cross_camera_candidates(role_results: Dict[str, dict], offsets: Dict[str, Optional[float]]) -> List[dict]:
    """Build review-only identity candidates between independently analyzed feeds."""
    role_order = {"entrance": 0, "store": 1, "checkout": 2}

    def camera_role(camera_key):
        report = role_results[camera_key].get("report", {})
        role = report.get("camera_role", camera_key)
        return role if role in role_order else camera_key

    order = sorted(
        role_results,
        key=lambda camera_key: (role_order.get(camera_role(camera_key), 99), str(camera_key)),
    )
    candidates = []
    for left_idx, source_camera_id in enumerate(order):
        for destination_camera_id in order[left_idx + 1:]:
            source_offset = offsets.get(source_camera_id)
            destination_offset = offsets.get(destination_camera_id)
            if source_offset is None or destination_offset is None:
                continue
            source_result = role_results[source_camera_id]
            destination_result = role_results[destination_camera_id]
            source_report = source_result["report"]
            destination_report = destination_result["report"]
            source_role = camera_role(source_camera_id)
            destination_role = camera_role(destination_camera_id)
            source_features = source_result.get("visitor_features", {})
            destination_features = destination_result.get("visitor_features", {})
            source_visitors = {v["visitor_id"]: v for v in source_report.get("visitors", [])}
            dest_visitors = {v["visitor_id"]: v for v in destination_report.get("visitors", [])}
            source_entries = {}
            for event in source_report.get("entrance_events", []):
                if event["type"] == "entry":
                    source_entries.setdefault(event["visitor_id"], event["timestamp_seconds"])
            for source_id, source_visitor in source_visitors.items():
                local_time = source_entries.get(source_id, source_visitor["first_seen_seconds"])
                absolute_time = float(source_offset) + float(local_time)
                source_feature = source_features.get(source_id)
                ranked = []
                for destination_id, destination_visitor in dest_visitors.items():
                    dest_time = float(destination_offset) + float(destination_visitor["first_seen_seconds"])
                    delta = dest_time - absolute_time
                    if delta < -2.0 or delta > 900.0:
                        continue
                    appearance = feature_similarity(source_feature, destination_features.get(destination_id))
                    # Time and coarse clothing are weak evidence; outputs are
                    # explicitly candidate-only and require review.
                    time_score = math.exp(-max(0.0, delta) / 420.0)
                    score = 0.55 * appearance + 0.45 * time_score
                    if score >= 0.58:
                        ranked.append((score, destination_id, appearance, delta))
                ranked.sort(reverse=True)
                for score, destination_id, appearance, delta in ranked[:3]:
                    candidates.append({
                        "source_camera_role": source_role,
                        "source_camera_id": source_camera_id,
                        "source_camera_name": source_report.get("camera_name", source_camera_id),
                        "source_visitor_id": source_id,
                        "destination_camera_role": destination_role,
                        "destination_camera_id": destination_camera_id,
                        "destination_camera_name": destination_report.get("camera_name", destination_camera_id),
                        "destination_visitor_id": destination_id,
                        "time_delta_seconds": round(delta, 2),
                        "coarse_appearance_similarity": round(appearance, 3),
                        "candidate_score_heuristic": round(score, 3),
                        "status": "review_candidate_only",
                        "automatic_identity_merge": False,
                        "warning": "Coarse clothing color and time order are insufficient to confirm a person match.",
                    })
    return candidates


def write_json(path: str, value: dict) -> None:
    def clean(obj):
        if isinstance(obj, dict):
            return {str(k): clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [clean(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            value = float(obj)
            return value if math.isfinite(value) else None
        if isinstance(obj, float) and not math.isfinite(obj):
            return None
        return obj
    Path(path).write_text(json.dumps(clean(value), indent=2, ensure_ascii=False), encoding="utf-8")


def write_summary_csv(path: str, global_report: dict) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["camera_role", "camera_id", "camera_name", "camera_local_visitors_estimated", "tracklets", "distinct_group_memberships",
                         "group_intervals", "peak_occupancy", "entry_events", "exit_events",
                         "checkout_presence_events", "payments_verified"])
        for camera_key, report in global_report.get("camera_reports", {}).items():
            summary = report.get("summary", {})
            writer.writerow([report.get("camera_role", camera_key), report.get("camera_id", camera_key),
                             report.get("camera_name", camera_key), summary.get("camera_local_visitors_estimated", 0), summary.get("tracklets", 0),
                             summary.get("distinct_group_memberships", 0), summary.get("group_intervals", 0),
                             summary.get("peak_detected_occupancy", 0), summary.get("entry_events", 0),
                             summary.get("exit_events", 0), summary.get("checkout_camera_presence_events", 0), 0])
