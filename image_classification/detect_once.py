"""One-shot object detector.

Runs the live YOLO detector until it produces one confident prediction,
prints it to stdout as a `RESULT\t<class_id>\t<class_name>` line, then exits.

Because this runs as its own process, exiting releases /dev/video2
unconditionally — even if an OpenCV read() is stuck — which is the only
reliable way to hand the camera to a lerobot rollout.
"""

import os
import sys

import object_detector as od

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "best.pt")
DETECTOR_CAMERA = 2

EXIT_RESULT = 0   # printed a RESULT line
EXIT_STOPPED = 1  # predictor stopped (e.g. 'q') without a result
EXIT_CAMERA = 3   # detector failed to start (camera busy / not found)


def main():
    try:
        predictor = od.start_live_prediction(
            model_path=MODEL_PATH,
            camera=DETECTOR_CAMERA,
            imgsz=640,
            conf=0.25,
            width=1280,
            height=720,
            save=None,
        )
    except (RuntimeError, FileNotFoundError) as err:
        print(f"detector startup failed: {err}", file=sys.stderr)
        return EXIT_CAMERA

    try:
        while predictor.is_running():
            result = predictor.get_weighted_class(
                seconds=3,
                threshold_average_confidence_per_frame=0.20,
            )
            if result is not None:
                class_id, class_name = result
                # Result goes to stdout (parsed by the parent); progress and
                # library chatter go to stderr so they don't pollute it.
                print(f"RESULT\t{class_id}\t{class_name}", flush=True)
                return EXIT_RESULT
            print("No confident prediction yet. Waiting...", file=sys.stderr)
        return EXIT_STOPPED
    finally:
        predictor.stop()


if __name__ == "__main__":
    raise SystemExit(main())
