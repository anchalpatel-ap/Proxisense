"""Quick test script for ProxiSense pipeline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import ProxiSensePipeline


def test_webcam(frames=60):
    print("Testing pipeline with webcam...")
    pipeline = ProxiSensePipeline(use_webcam=True)

    try:
        for i in range(frames):
            frame, alerts, metrics = pipeline.process_frame()
            if frame is None:
                print("No frame received")
                break

            status = "SAFE"
            for a in alerts:
                if a.zone == "RED":
                    status = "RED"
                    break
                elif a.zone == "AMBER" and status != "RED":
                    status = "AMBER"

            if i % 10 == 0:
                print(
                    f"Frame {i+1:3d} | Agents: {len(alerts):2d} | "
                    f"FPS: {metrics['fps']:5.1f} | "
                    f"Infer: {metrics['inference_time']:5.1f}ms | "
                    f"Status: {status}"
                )
    finally:
        pipeline.release()

    print("Test complete!")


def test_video(path="data/sample_videos/test.mp4"):
    from pathlib import Path as P
    if not P(path).exists():
        print(f"Video not found: {path}")
        return False

    print(f"Testing with video: {path}")
    pipeline = ProxiSensePipeline(video_path=path)

    total_frames = 0
    try:
        while True:
            frame, alerts, metrics = pipeline.process_frame()
            if frame is None:
                break
            total_frames += 1

            if total_frames % 30 == 0:
                print(
                    f"Frame {total_frames:4d} | Agents: {len(alerts):2d} | "
                    f"FPS: {metrics['fps']:5.1f} | "
                    f"Infer: {metrics['inference_time']:6.1f}ms"
                )
    finally:
        pipeline.release()

    print(f"Processed {total_frames} frames. Test complete!")
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--frames", type=int, default=60)
    args = parser.parse_args()

    if args.video:
        test_video(args.video)
    else:
        test_webcam(args.frames)
