import pprint
import time

from pilot_monitor.qt_compat import configure_qt_before_cv2_import

configure_qt_before_cv2_import()

import cv2
import mediapipe as mp

from pilot_monitor.calibration import PersonalProfile, default_profile_path
from pilot_monitor.explain import explain
from pilot_monitor.eye_detector import EyeDetector
from pilot_monitor.logging_export import FrameRecord, ResearchLogger
from pilot_monitor.overlays import draw_dashboard
from pilot_monitor.parser import get_args
from pilot_monitor.pipeline import MonitoringPipeline, frame_timestamp_ms, make_detector
from pilot_monitor.pose_estimation import HeadPoseEstimator
from pilot_monitor.qt_compat import configure_qt_fonts_after_cv2_import
from pilot_monitor.utils import get_landmarks, load_camera_parameters

configure_qt_fonts_after_cv2_import()


def _load_profile_and_model(args):
    """Load a pilot's calibration profile and, if available, their trained
    GRU temporal model. Returns ``(profile_or_None, gru_bundle_or_None)``.

    Running without ``--pilot-id`` (or before that pilot has been
    calibrated) is a deliberate, supported fallback: the personalization
    and GRU layers are optional enhancements over the event/risk pipeline,
    not a hard requirement to run at all (see the design document, Part 8).
    """
    if not args.pilot_id:
        print(
            "No --pilot-id given: running uncalibrated. Personalization and the "
            "GRU anomaly model are disabled; run `pilot-monitor-calibrate` first "
            "for a personalized session."
        )
        return None, None

    profile_path = default_profile_path(args.profiles_dir, args.pilot_id)
    if not profile_path.is_file():
        print(
            f"No calibration profile found for pilot '{args.pilot_id}' at "
            f"{profile_path}. Running uncalibrated for this pilot; run "
            f"`pilot-monitor-calibrate --pilot-id {args.pilot_id}` first."
        )
        return None, None

    profile = PersonalProfile.load(profile_path)

    gru_bundle = None
    if args.use_gru:
        try:
            from pilot_monitor.temporal_model import TORCH_AVAILABLE, load_pilot_model

            if TORCH_AVAILABLE:
                meta_json = args.profiles_dir / f"{args.pilot_id}_gru.json"
                if meta_json.is_file():
                    gru_bundle = load_pilot_model(args.pilot_id, args.profiles_dir)
                else:
                    print(f"No trained GRU model found for pilot '{args.pilot_id}'; continuing without it.")
            else:
                print("PyTorch not installed: continuing without the GRU anomaly model.")
        except Exception as error:  # pragma: no cover - defensive, environment-dependent
            print(f"Could not load GRU model ({error}); continuing without it.")

    return profile, gru_bundle


def main(argv=None):
    """Run monitoring (webcam or replay) and own the lifetime of camera/GUI resources."""
    args = get_args(argv)
    if not args.model_path.is_file():
        raise SystemExit(
            f"Face Landmarker model not found at {args.model_path}. Run "
            "`pilot-monitor-download-model` first."
        )

    if not cv2.useOptimized():
        cv2.setUseOptimized(True)

    try:
        camera_matrix, dist_coeffs, camera_image_size = (
            load_camera_parameters(args.camera_params)
            if args.camera_params
            else (None, None, None)
        )
    except (OSError, KeyError, ValueError) as error:
        raise SystemExit(f"Invalid camera parameters: {error}") from error

    if args.verbose:
        print("Arguments and parameters:")
        pprint.pp(vars(args), indent=4)

    profile, gru_bundle = _load_profile_and_model(args)
    pipeline = MonitoringPipeline(profile=profile, gru_bundle=gru_bundle) if profile else None
    logger = ResearchLogger() if args.log_path else None

    video_source = str(args.video) if args.video else args.camera
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        cap.release()
        raise SystemExit(f"Cannot open video source {video_source}")

    is_replay = args.video is not None

    try:
        with make_detector(
            args.model_path,
            args.min_detection_confidence,
            args.min_face_presence_confidence,
            args.min_tracking_confidence,
            args.max_faces,
        ) as detector:
            head_pose_kwargs = dict(
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
                camera_image_size=camera_image_size,
                show_axis=args.show_axis,
            )
            if pipeline is not None:
                pipeline.feature_extractor.head_pose = HeadPoseEstimator(**head_pose_kwargs)

            now = time.perf_counter()
            previous_frame_time = now
            stream_start = now
            previous_timestamp_ms = -1
            frame_index = 0
            fps = 0.0

            while True:
                frame_time = time.perf_counter()
                elapsed = frame_time - previous_frame_time
                previous_frame_time = frame_time
                if elapsed > 0:
                    fps = 1.0 / elapsed

                received, frame = cap.read()
                if not received:
                    print("Cannot receive frame from camera/video or stream ended")
                    break
                if not is_replay and args.camera == 0:
                    frame = cv2.flip(frame, 1)

                processing_start = cv2.getTickCount()
                frame_size = (frame.shape[1], frame.shape[0])
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = frame_timestamp_ms(
                    previous_timestamp_ms, frame_time - stream_start
                )
                previous_timestamp_ms = timestamp_ms
                result = detector.detect_for_video(mp_image, timestamp_ms)

                alerts = []
                ear = gaze_mag = None
                roll = pitch = yaw = None

                if pipeline is not None:
                    (
                        frame,
                        features,
                        normalized,
                        risk_result,
                        fatigue_update,
                        distraction_update,
                        gru_anomaly,
                    ) = pipeline.process_frame(frame, result, frame_size, frame_time)

                    ear, roll, pitch, yaw = features.ear, features.roll, features.pitch, features.yaw
                    if features.gaze_dx is not None:
                        gaze_mag = (features.gaze_dx**2 + features.gaze_dy**2) ** 0.5

                    for name, update in (("fatigue", fatigue_update), ("distraction", distraction_update)):
                        text = explain(name, update)
                        if text:
                            alerts.append(f"{name.upper()}: {update.state.value}")

                    if features.quality.value == "INVALID":
                        alerts.insert(0, "DATA INVALID")

                    if logger is not None:
                        logger.log(
                            FrameRecord(
                                timestamp=frame_time - stream_start,
                                frame_index=frame_index,
                                pilot_id=args.pilot_id,
                                quality=features.quality.value,
                                quality_reasons=features.quality_reasons,
                                ear=features.ear,
                                gaze_dx=features.gaze_dx,
                                gaze_dy=features.gaze_dy,
                                mar=features.mar,
                                roll=features.roll,
                                pitch=features.pitch,
                                yaw=features.yaw,
                                ear_z=normalized.zscores.get("ear"),
                                gaze_dx_z=normalized.zscores.get("gaze_dx"),
                                gaze_dy_z=normalized.zscores.get("gaze_dy"),
                                mar_z=normalized.zscores.get("mar"),
                                roll_z=normalized.zscores.get("roll"),
                                pitch_z=normalized.zscores.get("pitch"),
                                yaw_z=normalized.zscores.get("yaw"),
                                gru_anomaly=gru_anomaly,
                                fatigue_risk=risk_result.fatigue_risk,
                                distraction_risk=risk_result.distraction_risk,
                                fatigue_state=fatigue_update.state.value,
                                distraction_state=distraction_update.state.value,
                                fatigue_reasons=risk_result.fatigue_reasons,
                                distraction_reasons=risk_result.distraction_reasons,
                            )
                        )
                    face_detected = features.face_detected
                else:
                    # Uncalibrated fallback: raw features only, no personalization/events/risk.
                    face_detected = bool(result.face_landmarks)
                    if face_detected:
                        landmarks = get_landmarks(result.face_landmarks)
                        eye_detector = EyeDetector()
                        ear = eye_detector.get_EAR(landmarks, frame_size)
                        gaze_mag = eye_detector.get_Gaze_Score(frame, landmarks, frame_size)

                processing_ms = (
                    (cv2.getTickCount() - processing_start) / cv2.getTickFrequency()
                ) * 1000
                draw_dashboard(
                    frame,
                    ear=ear,
                    gaze=gaze_mag,
                    perclos=None,
                    perclos_ready=False,
                    roll=roll,
                    pitch=pitch,
                    yaw=yaw,
                    ear_threshold=0.15,
                    gaze_threshold=0.2,
                    perclos_threshold=0.2,
                    roll_threshold=args.roll_thresh,
                    pitch_threshold=args.pitch_thresh,
                    yaw_threshold=args.yaw_thresh,
                    roll_limit=args.roll_bar_limit,
                    pitch_limit=args.pitch_bar_limit,
                    yaw_limit=args.yaw_bar_limit,
                    alerts=alerts,
                    face_detected=face_detected,
                    fps=fps if args.show_fps else None,
                    processing_ms=processing_ms if args.show_proc_time else None,
                    show_panels=args.show_dashboard,
                )

                if not is_replay:
                    cv2.imshow("Press 'q' to terminate", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                frame_index += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if logger is not None and len(logger) > 0:
            logger.to_csv(args.log_path)
            print(f"Wrote {len(logger)} research-log rows to {args.log_path}")


if __name__ == "__main__":
    main()
