"""
ProxiSense - Real-Time Pedestrian & Obstacle Intent Prediction for ADAS

Entry point for running the pipeline without Streamlit dashboard.
Processes video and shows OpenCV window with overlays.

Usage:
    python main.py                   
    python main.py --video path/to/video.mp4
    python main.py --video path/to/video.mp4 --no-display
"""

import argparse
import cv2
import time
from pathlib import Path
from src.pipeline import ProxiSensePipeline


def parse_args():
    parser = argparse.ArgumentParser(description="ProxiSense ADAS Intent Prediction")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to video file (default: webcam)")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="Path to config file")
    parser.add_argument("--no-display", action="store_true",
                        help="Run without display (headless)")
    parser.add_argument("--record", type=str, default=None,
                        help="Record output to file")
    parser.add_argument("--benchmark", action="store_true",
                        help="Benchmark mode (measure FPS)")
    parser.add_argument("--detect-interval", type=int, default=2,
                        help="Run YOLO detection every N frames (default: 2)")
    return parser.parse_args()


def main():
    args = parse_args()

    pipeline = ProxiSensePipeline(
        config_path=args.config,
        use_webcam=args.video is None,
        video_path=args.video
    )

    writer = None
    if args.record:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.record, fourcc, 30.0,
                                 (640, 480))

    frame_count = 0
    total_inference_time = 0
    detect_interval = args.detect_interval
    print(f"[ProxiSense] Pipeline started. Press 'q' to quit.")
    print(f"[ProxiSense] Detection interval: every {detect_interval} frame(s)")

    try:
        while True:
            run_detection = (frame_count % detect_interval == 0)
            frame, alerts, metrics = pipeline.process_frame(run_detection=run_detection)

            if frame is None:
                print("[ProxiSense] End of input stream")
                break

            frame = pipeline.render_alerts(frame, alerts)
            frame = pipeline.render_trajectories(frame)

            if writer:
                writer.write(frame)

            frame_count += 1
            total_inference_time += metrics.get("inference_time", 0)

            if not args.no_display:
                cv2.imshow("ProxiSense - ADAS Intent Prediction", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if frame_count % 30 == 0:
                avg_fps = metrics.get("fps", 0)
                agents = len(alerts)
                print(f"[Frame {frame_count}] FPS: {avg_fps:.1f} | "
                      f"Agents: {agents} | "
                      f"Inference: {metrics.get('inference_time', 0):.1f}ms | "
                      f"Status: {pipeline.alert_engine.get_system_status(alerts)['status']}")

    except KeyboardInterrupt:
        print("\n[ProxiSense] Interrupted by user")

    finally:
        pipeline.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    if frame_count > 0:
        avg_inference = total_inference_time / frame_count
        print(f"\n[ProxiSense] Summary: {frame_count} frames processed")
        print(f"  Avg inference: {avg_inference:.1f}ms/frame")

    print("[ProxiSense] Shutdown complete")


if __name__ == "__main__":
    main()