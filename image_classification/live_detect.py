import argparse
import os
import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Run live YOLO object detection from a webcam.")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained YOLO model (.pt file)"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index (default: 0)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (default: 640)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Camera capture width (default: 1280)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Camera capture height (default: 720)"
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional output video path, e.g. output.mp4"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.model):
        print(f"Error: model file not found: {args.model}")
        return

    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: could not open camera {args.camera}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    writer = None

    if args.save is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = cv2.VideoWriter(args.save, fourcc, fps, (frame_width, frame_height))

        if not writer.isOpened():
            print(f"Error: could not open video writer for {args.save}")
            cap.release()
            return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: failed to read frame from camera.")
            break

        results = model(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)
        annotated_frame = results[0].plot()

        cv2.imshow("YOLO Live Detection", annotated_frame)

        if writer is not None:
            writer.write(annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()