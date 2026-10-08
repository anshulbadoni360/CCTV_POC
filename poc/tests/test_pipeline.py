import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import pipeline


class DeterministicDetector:
    """Test-only detector that finds the two colored person-shaped patches."""
    def __init__(self, settings):
        self.backend = "test_stub"
        self.warning = None

    def detect(self, frame):
        results = []
        masks = [
            cv2.inRange(frame, np.array([0, 0, 130], dtype=np.uint8), np.array([90, 90, 255], dtype=np.uint8)),
            cv2.inRange(frame, np.array([130, 0, 0], dtype=np.uint8), np.array([255, 90, 90], dtype=np.uint8)),
        ]
        for mask in masks:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if h >= 45 and w >= 18:
                    results.append(((x, y, x + w, y + h), 0.95))
        return sorted(results, key=lambda item: item[0][0])


class PipelineSmokeTest(unittest.TestCase):
    def make_video(self, path: Path):
        width, height, fps, frames = 320, 240, 10.0, 70
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
        self.assertTrue(writer.isOpened(), "OpenCV MJPG video writer is required for this smoke test")
        for i in range(frames):
            frame = np.full((height, width, 3), (35, 42, 39), dtype=np.uint8)
            top = i
            cv2.rectangle(frame, (58, top), (98, top + 89), (0, 0, 230), -1)
            cv2.rectangle(frame, (132, top), (172, top + 89), (230, 0, 0), -1)
            writer.write(frame)
        writer.release()
        return fps, frames

    def test_frame_skip_zero_processes_every_frame(self):
        self.assertEqual(pipeline._frame_stride(29.97, 0), 1)
        self.assertEqual(pipeline._frame_stride(29.97, 3), 4)
        legacy = pipeline.PipelineSettings.from_dict({"analysis_fps": 5})
        self.assertEqual(pipeline._frame_stride(10.0, legacy.frame_skip, legacy.analysis_fps), 2)
        explicit_skip = pipeline.PipelineSettings.from_dict({"analysis_fps": 5, "frame_skip": 0})
        self.assertEqual(pipeline._frame_stride(10.0, explicit_skip.frame_skip, explicit_skip.analysis_fps), 1)

    def test_single_clip_pipeline_tracks_groups_and_entry_events(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sample.avi"
            self.make_video(source)
            settings = {
                "detector": "hog",
                "analysis_fps": 5,
                "min_track_sec": 0.3,
                "group_min_together_sec": 2,
                "group_proximity_heights": 1.4,
                "exclude_stationary_from_groups": False,
                "entrance_line_y": 0.52,
                "entry_direction": "down",
            }
            with patch("pipeline.PersonDetector", DeterministicDetector):
                result = pipeline.analyze_video(str(source), "entrance", str(root / "out"), settings)
            report = result["report"]
            self.assertEqual(report["video"]["frames_processed"], 70)
            self.assertEqual(report["summary"]["camera_local_visitors_estimated"], 2)
            self.assertGreaterEqual(report["summary"]["entry_events"], 1)
            self.assertGreaterEqual(report["summary"]["group_intervals"], 1)
            self.assertTrue((root / "out" / "entrance_annotated.mp4").stat().st_size > 1000)
            self.assertTrue((root / "out" / "entrance_tracklets.csv").exists())
            self.assertIn("camera_local_anonymous_estimates", report["identity_policy"]["scope"])
            self.assertEqual(report["summary"]["payments_verified"], 0)
            self.assertGreaterEqual(report["video"]["sampled_frames"], 30)

    def test_entrance_hysteresis_does_not_count_late_appearance_as_preexisting(self):
        def obs(frame, y):
            return pipeline.Observation(frame, frame / 10.0, (30, int(y - 80), 70, int(y)), 0.9, (50, float(y)), 80.0)
        observations = {
            "E-V001": [obs(0, 70), obs(10, 91), obs(20, 105), obs(30, 120), obs(40, 130),
                       obs(50, 120), obs(60, 126), obs(70, 130)],
            "E-V002": [obs(30, 130), obs(40, 130), obs(50, 130)],
        }
        events, preexisting, ambiguous, late_inside = pipeline._detect_entry_events(
            observations, 10.0, 0.5, "down", 200
        )
        self.assertEqual([event["type"] for event in events], ["entry"])
        self.assertEqual(preexisting, [])
        self.assertEqual(len(late_inside), 1)
        self.assertEqual(late_inside[0]["visitor_id"], "E-V002")

    def test_final_video_draws_a_separate_group_enclosure(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        group = {"group_id":"S-G001", "members":["S-V001", "S-V002"],
                 "member_count":2, "start_frame":0, "end_frame":10}
        self.assertEqual(pipeline._short_group_id(group["group_id"]), "G1")
        annotated = pipeline._draw_overlay(
            frame,
            [(1, (25, 40, 55, 120)), (2, (70, 40, 100, 120))],
            {1:"S-V001", 2:"S-V002"}, 1, "store", [group], 0.52, "off",
            2, 2, 2, 1, {},
        )
        self.assertEqual(tuple(annotated[128, 50]), (34, 164, 235))

    def test_live_preview_zoom_is_ui_only_and_supports_person_and_group_targets(self):
        yy, xx = np.indices((400, 600))
        frame = np.stack((xx % 255, yy % 255, (xx + yy) % 255), axis=2).astype(np.uint8)
        records = [(1, (260, 160, 300, 240)), (2, (320, 160, 360, 240))]
        candidates = [{
            "group_id": "P1", "display_group_id": "G1",
            "boxes": [(260, 160, 300, 240), (320, 160, 360, 240)],
        }]
        unchanged = pipeline._apply_live_preview_zoom(frame, None, records, candidates)
        zoom_person = pipeline._apply_live_preview_zoom(frame, "T001", records, candidates)
        zoom_group = pipeline._apply_live_preview_zoom(frame, "G1", records, candidates)
        self.assertEqual(zoom_person.shape, frame.shape)
        self.assertEqual(zoom_group.shape, frame.shape)
        self.assertFalse(np.array_equal(zoom_person, frame))
        self.assertFalse(np.array_equal(zoom_group, frame))
        self.assertTrue(np.array_equal(unchanged, frame))

    def test_live_zoom_never_changes_saved_annotated_video(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sample.avi"
            self.make_video(source)
            settings = {
                "detector": "hog", "frame_skip": 0, "min_track_sec": 0.3,
                "min_track_observations": 2, "detect_groups": False,
                "store_zone_layout": "off",
            }
            normal_previews, zoomed_previews = [], []
            with patch("pipeline.PersonDetector", DeterministicDetector):
                normal = pipeline.analyze_video(
                    str(source), "store", str(root / "normal"), settings,
                    preview_callback=lambda frame: normal_previews.append(frame.copy()),
                )
                zoomed = pipeline.analyze_video(
                    str(source), "store", str(root / "zoomed"), settings,
                    preview_callback=lambda frame: zoomed_previews.append(frame.copy()),
                    preview_zoom_getter=lambda: "T001",
                )
            self.assertTrue(normal_previews and zoomed_previews)
            self.assertFalse(np.array_equal(normal_previews[0], zoomed_previews[0]))
            normal_path = root / "normal" / normal["report"]["outputs"]["annotated_video"]
            zoomed_path = root / "zoomed" / zoomed["report"]["outputs"]["annotated_video"]
            normal_capture, zoomed_capture = cv2.VideoCapture(str(normal_path)), cv2.VideoCapture(str(zoomed_path))
            normal_frames, zoomed_frames = [], []
            while True:
                normal_ok, normal_frame = normal_capture.read()
                zoomed_ok, zoomed_frame = zoomed_capture.read()
                if not normal_ok or not zoomed_ok:
                    self.assertEqual(normal_ok, zoomed_ok)
                    break
                normal_frames.append(normal_frame)
                zoomed_frames.append(zoomed_frame)
            normal_capture.release()
            zoomed_capture.release()
            self.assertEqual(len(normal_frames), len(zoomed_frames))
            self.assertTrue(all(np.array_equal(a, b) for a, b in zip(normal_frames, zoomed_frames)))

    def test_live_group_box_appears_after_three_seconds_with_different_box_heights(self):
        settings = pipeline.PipelineSettings.from_dict({"group_min_together_sec": 3.0, "group_proximity_heights": 1.35})
        pair_runs = {}
        live_group_ids = {}
        pair_together_time = {}
        candidates = []
        left = pipeline.Tracklet(1)
        right = pipeline.Tracklet(2)
        for frame in range(31):
            left.observations.append(pipeline.Observation(
                frame, frame / 10.0, (30, 40 + 2*frame, 80, 190 + 2*frame), 0.95, (55.0, 190.0 + 2*frame), 150.0
            ))
            right.observations.append(pipeline.Observation(
                frame, frame / 10.0, (195, 120 + 2*frame, 235, 190 + 2*frame), 0.95, (215.0, 190.0 + 2*frame), 70.0
            ))
            candidates = pipeline._update_live_group_candidates(
                [left, right], frame, 10.0, 1, settings, pair_runs, live_group_ids,
                pair_together_time=pair_together_time,
            )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["group_id"], "P1")
        self.assertGreaterEqual(candidates[0]["duration_seconds"], 3.0)
        preview = pipeline._draw_live_preview(
            np.zeros((240, 320, 3), dtype=np.uint8),
            [(1, (30, 40, 80, 190)), (2, (195, 120, 235, 190))],
            "store", 30, 2, 0.52, "off", candidates,
        )
        self.assertEqual(preview.shape, (240, 320, 3))

    def test_live_sustained_box_overlap_creates_provisional_candidate(self):
        settings = pipeline.PipelineSettings.from_dict({
            "group_min_together_sec": 8.0,
            "group_min_box_overlap_sec": 3.0,
            "group_box_overlap_min_ratio": 0.15,
            "group_proximity_heights": 0.50,
        })
        pair_runs, group_ids, assignments, pair_together_time = {}, {}, {}, {}
        left, right = pipeline.Tracklet(1), pipeline.Tracklet(2)
        candidates = []
        for frame in range(31):
            left.observations.append(pipeline.Observation(
                frame, frame / 10.0, (0, 0, 100, 100), 0.95, (50.0, 100.0), 100.0
            ))
            right.observations.append(pipeline.Observation(
                frame, frame / 10.0, (80, 0, 180, 100), 0.95, (130.0, 100.0), 100.0
            ))
            candidates = pipeline._update_live_group_candidates(
                [left, right], frame, 10.0, 1, settings, pair_runs, group_ids, assignments,
                pair_together_time,
            )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["group_id"], "P1")
        self.assertEqual(candidates[0]["display_group_id"], "G1")
        self.assertEqual(candidates[0]["evidence_basis"], "sustained_box_overlap")
        self.assertTrue(candidates[0]["likely_group_so_far"])
        self.assertGreaterEqual(candidates[0]["duration_seconds"], 3.0)

    def test_live_group_label_stays_sticky_against_cashier_pair(self):
        settings = pipeline.PipelineSettings.from_dict({
            "group_min_together_sec": 3, "group_reassign_min_sec": 8,
            "group_reassign_margin": 0.30,
        })
        pair_runs, group_ids, assignments, pair_together_time = {}, {}, {}, {}
        histories = {}

        def track(track_id, frame, foot, height=90.0):
            x, y = foot
            moved_y = float(y + 2 * frame)
            obs = pipeline.Observation(frame, frame / 10.0,
                                       (int(x - 20), int(moved_y - height), int(x + 20), int(moved_y)),
                                       0.95, (float(x), moved_y), height)
            histories.setdefault(track_id, []).append(obs)
            return pipeline.Tracklet(track_id, observations=list(histories[track_id]))

        for frame in range(31):
            pipeline._update_live_group_candidates(
                [track(1, frame, (50, 190)), track(2, frame, (105, 190))],
                frame, 10.0, 1, settings, pair_runs, group_ids, assignments, pair_together_time,
            )
        self.assertEqual(assignments[1]["group_id"], "P1")

        candidates = []
        for frame in range(31, 112):
            candidates = pipeline._update_live_group_candidates(
                [track(1, frame, (50, 190)), track(3, frame, (80, 190))],
                frame, 10.0, 1, settings, pair_runs, group_ids, assignments, pair_together_time,
            )
        self.assertEqual(assignments[1]["group_id"], "P1")
        self.assertFalse(any(candidate["group_id"] == "P2" for candidate in candidates))

    def test_group_membership_is_exclusive_and_reassignment_needs_margin(self):
        settings = pipeline.PipelineSettings.from_dict({"group_min_together_sec": 3, "group_reassign_margin": 0.15, "group_reassign_min_sec": 6})
        groups = [
            {"group_id":"old", "members":["A","B"], "start_frame":0, "end_frame":50,
             "together_seconds":5.1, "close_ratio":0.55, "confidence_heuristic":0.70},
            {"group_id":"new", "members":["A","C"], "start_frame":40, "end_frame":100,
             "together_seconds":6.1, "close_ratio":1.0, "confidence_heuristic":0.98},
        ]
        resolved = pipeline._resolve_exclusive_groups(groups, 10.0, 1, settings, "E")
        self.assertEqual(len(resolved), 2)
        old = next(group for group in resolved if group["members"] == ["A", "B"])
        new = next(group for group in resolved if group["members"] == ["A", "C"])
        self.assertLess(old["end_frame"], new["start_frame"])

        weak = [
            {"group_id":"old", "members":["A","B"], "start_frame":0, "end_frame":100,
             "together_seconds":10.1, "close_ratio":0.9, "confidence_heuristic":0.80},
            {"group_id":"new", "members":["A","C"], "start_frame":40, "end_frame":100,
             "together_seconds":6.1, "close_ratio":1.0, "confidence_heuristic":0.90},
        ]
        resolved_weak = pipeline._resolve_exclusive_groups(weak, 10.0, 1, settings, "E")
        self.assertEqual(len(resolved_weak), 1)

    def test_low_movement_alone_is_not_classified_as_cashier(self):
        settings = pipeline.PipelineSettings.from_dict({"group_min_together_sec": 3})
        stationary = [
            pipeline.Observation(frame, frame / 10.0, (20, 0, 80, 100), 0.95, (50.0, 100.0), 100.0)
            for frame in range(101)
        ]
        candidates = pipeline._stationary_role_candidates({"S-V001": stationary}, 10.0, 10.0, settings)
        self.assertEqual(candidates["S-V001"]["role_candidate"], "stationary_candidate_needs_review")

        visitors = {"S-V001": stationary}
        for visitor_id, first in (("S-V002", 0), ("S-V003", 30), ("S-V004", 60)):
            visitors[visitor_id] = [
                pipeline.Observation(frame, frame / 10.0, (22, 0, 82, 100), 0.95, (52.0, 100.0), 100.0)
                for frame in range(first, first + 20)
            ]
        candidates = pipeline._stationary_role_candidates(visitors, 10.0, 10.0, settings)
        self.assertEqual(candidates["S-V001"]["role_candidate"], "likely_fixed_post_candidate")
        self.assertEqual(candidates["S-V001"]["distinct_close_visitors"], 3)

    def test_sustained_box_overlap_is_an_alternative_dyad_signal(self):
        settings = pipeline.PipelineSettings.from_dict({
            "group_min_together_sec": 8.0,
            "group_min_box_overlap_sec": 3.0,
            "group_box_overlap_min_ratio": 0.15,
            "group_proximity_heights": 0.50,
            "group_likely_share_ratio": 0.40,
        })
        observations = {
            "A": [
                pipeline.Observation(frame, frame / 10.0, (0, 0, 100, 100), 0.95, (50.0, 100.0), 100.0)
                for frame in range(31)
            ],
            "B": [
                pipeline.Observation(frame, frame / 10.0, (80, -50, 180, 150), 0.95, (130.0, 150.0), 200.0)
                for frame in range(31)
            ],
        }
        self.assertAlmostEqual(pipeline._box_overlap_ratio((0, 0, 100, 100), (80, -50, 180, 150)), 0.20)
        groups, evidence, _people = pipeline._close_group_frames(observations, settings, 10.0, 1, "S")
        resolved = pipeline._resolve_exclusive_groups(groups, 10.0, 1, settings, "S")
        self.assertEqual(len(resolved), 1)
        group = resolved[0]
        self.assertEqual(group["members"], ["A", "B"])
        self.assertEqual(group["evidence_basis"], "sustained_box_overlap")
        self.assertTrue(group["meets_likely_group_share"])
        self.assertGreaterEqual(group["minimum_member_together_share"], 0.40)
        self.assertTrue(evidence[("A", "B")]["qualifies_via_sustained_overlap"])

    def test_box_overlap_shorter_than_threshold_does_not_seed_a_dyad(self):
        settings = pipeline.PipelineSettings.from_dict({
            "group_min_together_sec": 8.0,
            "group_min_box_overlap_sec": 3.0,
            "group_box_overlap_min_ratio": 0.15,
            "group_proximity_heights": 0.50,
        })
        observations = {
            visitor_id: [
                pipeline.Observation(frame, frame / 10.0, bbox, 0.95, foot, 100.0)
                for frame in range(29)
            ]
            for visitor_id, bbox, foot in (
                ("A", (0, 0, 100, 100), (50.0, 100.0)),
                ("B", (80, 0, 180, 100), (130.0, 100.0)),
            )
        }
        groups, _evidence, _people = pipeline._close_group_frames(observations, settings, 10.0, 1, "S")
        self.assertEqual(groups, [])

    def test_observed_nonoverlap_break_resets_continuous_overlap_timer(self):
        settings = pipeline.PipelineSettings.from_dict({
            "group_min_together_sec": 8.0,
            "group_min_box_overlap_sec": 3.0,
            "group_box_overlap_min_ratio": 0.15,
            "group_proximity_heights": 0.50,
        })
        observations = {"A": [], "B": []}
        for frame in range(35):
            observations["A"].append(pipeline.Observation(
                frame, frame / 10.0, (0, 0, 100, 100), 0.95, (50.0, 100.0), 100.0
            ))
            box, foot = ((80, 0, 180, 100), (130.0, 100.0)) if frame != 17 else ((200, 0, 300, 100), (250.0, 100.0))
            observations["B"].append(pipeline.Observation(
                frame, frame / 10.0, box, 0.95, foot, 100.0
            ))
        groups, _evidence, _people = pipeline._close_group_frames(observations, settings, 10.0, 1, "S")
        self.assertEqual(groups, [])

    def test_likely_group_share_uses_each_persons_total_observed_time(self):
        settings = pipeline.PipelineSettings.from_dict({
            "group_min_together_sec": 8.0,
            "group_min_box_overlap_sec": 3.0,
            "group_box_overlap_min_ratio": 0.15,
            "group_proximity_heights": 0.50,
            "group_likely_share_ratio": 0.40,
        })

        def pair_observations(longer_person_frames):
            observations = {"A": [], "B": []}
            for frame in range(longer_person_frames + 1):
                timestamp = frame / 10.0
                observations["A"].append(pipeline.Observation(
                    frame, timestamp, (0, 0, 100, 100), 0.95, (50.0, 100.0), 100.0
                ))
                if frame <= 280:
                    if frame <= 120:
                        bbox, foot = (80, 0, 180, 100), (130.0, 100.0)
                    else:
                        bbox, foot = (200, 0, 300, 100), (250.0, 100.0)
                    observations["B"].append(pipeline.Observation(
                        frame, timestamp, bbox, 0.95, foot, 100.0
                    ))
            return observations

        groups, _evidence, _people = pipeline._close_group_frames(pair_observations(300), settings, 10.0, 1, "S")
        resolved = pipeline._resolve_exclusive_groups(groups, 10.0, 1, settings, "S")
        likely = next(group for group in resolved if group["members"] == ["A", "B"])
        self.assertAlmostEqual(likely["member_observed_seconds"]["A"], 30.1, places=1)
        self.assertAlmostEqual(likely["member_observed_seconds"]["B"], 28.1, places=1)
        self.assertGreaterEqual(likely["together_seconds_total"], 12.0)
        self.assertGreaterEqual(likely["member_together_share"]["A"], 0.40)
        self.assertGreaterEqual(likely["member_together_share"]["B"], 0.40)
        self.assertEqual(likely["classification"], "likely_group")

        groups_longer, _evidence, _people = pipeline._close_group_frames(pair_observations(400), settings, 10.0, 1, "S")
        resolved_longer = pipeline._resolve_exclusive_groups(groups_longer, 10.0, 1, settings, "S")
        still_observed = next(group for group in resolved_longer if group["members"] == ["A", "B"])
        self.assertLess(still_observed["member_together_share"]["A"], 0.40)
        self.assertEqual(still_observed["classification"], "observed_group")

    def test_close_people_moving_in_opposite_directions_do_not_form_a_group(self):
        settings = pipeline.PipelineSettings.from_dict({
            "group_min_together_sec": 3, "group_proximity_heights": 1.35,
        })
        observations = {}
        for visitor_id, x, y_sign in (("A", 50.0, 1.0), ("B", 90.0, -1.0)):
            observations[visitor_id] = [
                pipeline.Observation(frame, frame / 10.0, (int(x-20), 0, int(x+20), 100),
                                    0.95, (x, 100.0 + y_sign * 2 * frame), 100.0)
                for frame in range(31)
            ]
        groups, _evidence, _people = pipeline._close_group_frames(observations, settings, 10.0, 1, "S")
        self.assertEqual(groups, [])

    def test_core_expansion_does_not_merge_a_transitive_chain(self):
        settings = pipeline.PipelineSettings.from_dict({
            "group_min_together_sec": 3, "group_proximity_heights": 0.85,
            "group_min_close_ratio": 0.55,
        })
        observations = {}
        positions = {"A":50.0, "B":100.0, "C":180.0, "D":230.0}
        for visitor_id, x in positions.items():
            observations[visitor_id] = [
                pipeline.Observation(frame, frame / 10.0, (int(x-20), 0, int(x+20), 100),
                                    0.95, (x, 100.0 + frame), 100.0)
                for frame in range(31)
            ]
        groups, _evidence, _people = pipeline._close_group_frames(observations, settings, 10.0, 1, "S")
        memberships = {tuple(group["members"]) for group in groups}
        self.assertIn(("A", "B"), memberships)
        self.assertIn(("C", "D"), memberships)
        self.assertFalse(any(len(members) > 2 for members in memberships))

    def test_core_expansion_can_include_a_member_near_multiple_core_members(self):
        settings = pipeline.PipelineSettings.from_dict({
            "group_min_together_sec": 3, "group_proximity_heights": 0.75,
            "group_min_close_ratio": 0.55,
        })
        observations = {}
        positions = {"A":0.0, "B":70.0, "C":35.0, "D":105.0}
        for visitor_id, x in positions.items():
            observations[visitor_id] = [
                pipeline.Observation(frame, frame / 10.0, (int(x-20), 0, int(x+20), 100),
                                    0.95, (x, 100.0 + 2*frame), 100.0)
                for frame in range(31)
            ]
        groups, _evidence, _people = pipeline._close_group_frames(observations, settings, 10.0, 1, "S")
        self.assertTrue(any(set(group["members"]) == {"A", "B", "C", "D"} for group in groups))

    def test_person_does_not_flip_to_cashier_group_and_back(self):
        settings = pipeline.PipelineSettings.from_dict({
            "group_min_together_sec": 3,
            "group_reassign_margin": 0.30,
            "group_reassign_min_sec": 8,
        })
        groups = [
            {"group_id":"g1a", "members":["A","B"], "start_frame":0, "end_frame":100,
             "together_seconds":10.1, "close_ratio":0.9, "confidence_heuristic":0.95},
            {"group_id":"cashier", "members":["A","CASHIER"], "start_frame":50, "end_frame":140,
             "together_seconds":9.1, "close_ratio":1.0, "confidence_heuristic":1.0},
            {"group_id":"g1b", "members":["A","B"], "start_frame":150, "end_frame":200,
             "together_seconds":5.1, "close_ratio":0.9, "confidence_heuristic":0.9},
        ]
        resolved = pipeline._resolve_exclusive_groups(groups, 10.0, 1, settings, "S")
        self.assertEqual(len(resolved), 2)
        self.assertTrue(all(group["members"] == ["A", "B"] for group in resolved))
        self.assertEqual(resolved[0]["group_id"], resolved[1]["group_id"])

    def test_repeat_intervals_keep_the_same_group_id(self):
        settings = pipeline.PipelineSettings.from_dict({"group_min_together_sec": 1})
        groups = [
            {"group_id":"a", "members":["A","B"], "start_frame":0, "end_frame":30,
             "together_seconds":3.1, "close_ratio":0.9, "confidence_heuristic":0.8},
            {"group_id":"b", "members":["A","B"], "start_frame":50, "end_frame":80,
             "together_seconds":3.1, "close_ratio":0.9, "confidence_heuristic":0.8},
        ]
        resolved = pipeline._resolve_exclusive_groups(groups, 10.0, 1, settings, "E")
        self.assertEqual(len(resolved), 2)
        self.assertEqual(resolved[0]["group_id"], resolved[1]["group_id"])

    def test_botsort_external_ids_keep_tracklet_assignment(self):
        manager = pipeline.TrackletManager(10.0, pipeline.PipelineSettings.from_dict({"max_track_age_sec": 2}))
        frame = np.zeros((160, 120, 3), dtype=np.uint8)
        box = (30, 10, 65, 130)
        manager.update(frame, 0, 0.0, [(box, 0.9, 17)])
        manager.update(frame, 5, 0.5, [(box, 0.9, 17)])
        self.assertEqual(len(manager.tracks), 1)
        track = next(iter(manager.tracks.values()))
        self.assertEqual(track.source_tracker_id, 17)
        self.assertEqual(len(track.observations), 2)

    def test_multiple_same_role_feeds_keep_distinct_review_only_ids(self):
        feature = np.asarray([0.5, 0.5], dtype=np.float32)
        role_results = {
            "entrance_1": {
                "report": {"camera_role":"entrance", "camera_id":"entrance_1", "camera_name":"Entrance · Cam 1",
                           "visitors":[{"visitor_id":"E-V001", "first_seen_seconds":0.0}], "entrance_events":[]},
                "visitor_features": {"E-V001": feature},
            },
            "entrance_2": {
                "report": {"camera_role":"entrance", "camera_id":"entrance_2", "camera_name":"Entrance · Cam 2",
                           "visitors":[{"visitor_id":"E-V001", "first_seen_seconds":0.0}], "entrance_events":[]},
                "visitor_features": {"E-V001": feature},
            },
        }
        candidates = pipeline.build_cross_camera_candidates(role_results, {"entrance_1":0.0, "entrance_2":10.0})
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["source_camera_id"], "entrance_1")
        self.assertEqual(candidates[0]["destination_camera_id"], "entrance_2")
        self.assertFalse(candidates[0]["automatic_identity_merge"])

    def test_manual_cashier_exclusion_removes_camera_id_from_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sample.avi"
            self.make_video(source)
            settings = {
                "detector": "hog",
                "analysis_fps": 5,
                "min_track_sec": 0.3,
                "group_min_together_sec": 2,
                "group_proximity_heights": 1.4,
                "exclude_stationary_from_groups": False,
                "exclude_group_visitor_ids": ["S-V001", "T001"],
            }
            with patch("pipeline.PersonDetector", DeterministicDetector):
                result = pipeline.analyze_video(str(source), "store", str(root / "out"), settings)
            report = result["report"]
            self.assertEqual(report["groups"], [])
            excluded = next(visitor for visitor in report["visitors"] if visitor["visitor_id"] == "S-V001")
            self.assertEqual(excluded["role_candidate"], "manually_excluded_from_groups")
            self.assertEqual(report["group_assignment_policy"]["manually_excluded_visitor_ids"], ["S-V001"])
            self.assertEqual(report["group_assignment_policy"]["unmatched_manual_exclusion_inputs"], [])

    def test_group_detection_can_be_disabled_without_disabling_tracking(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sample.avi"
            self.make_video(source)
            settings = {
                "detector": "hog",
                "analysis_fps": 5,
                "min_track_sec": 0.3,
                "detect_groups": False,
                "exclude_stationary_from_groups": False,
            }
            with patch("pipeline.PersonDetector", DeterministicDetector):
                with patch("pipeline._update_live_group_candidates", side_effect=AssertionError("group stage should be skipped")):
                    result = pipeline.analyze_video(str(source), "store", str(root / "out"), settings)
            report = result["report"]
            self.assertFalse(report["group_settings"]["group_detection_enabled"])
            self.assertFalse(report["group_assignment_policy"]["group_detection_enabled"])
            self.assertEqual(report["summary"]["group_intervals"], 0)
            self.assertEqual(report["summary"]["camera_local_visitors_estimated"], 2)
            self.assertTrue((root / "out" / "store_annotated.mp4").exists())
            self.assertFalse(pipeline.PipelineSettings.from_dict({"detect_groups":"false"}).detect_groups)

    def test_one_clip_is_sufficient_and_checkout_is_not_payment_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sample.avi"
            self.make_video(source)
            with patch("pipeline.PersonDetector", DeterministicDetector):
                result = pipeline.analyze_video(
                    str(source), "checkout", str(root / "out"), {
                        "analysis_fps": 5,
                        "min_track_sec": 0.3,
                        "group_min_together_sec": 2,
                    }, camera_id="checkout_2_test", camera_name="Checkout · Cam 2",
                )
            report = result["report"]
            self.assertEqual(report["summary"]["camera_local_visitors_estimated"], 2)
            self.assertEqual(report["summary"]["checkout_camera_presence_events"], 2)
            self.assertEqual(report["camera_id"], "checkout_2_test")
            self.assertEqual(report["camera_name"], "Checkout · Cam 2")
            self.assertTrue((root / "out" / "checkout_2_test_annotated.mp4").exists())
            self.assertTrue((root / "out" / "checkout_2_test_tracklets.csv").exists())
            self.assertFalse(report["checkout_presence_events"][0]["purchase_verified"])


if __name__ == "__main__":
    unittest.main()
