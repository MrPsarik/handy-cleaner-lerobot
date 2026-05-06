import os
import time
import threading
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Optional

import cv2
from ultralytics import YOLO


@dataclass
class FramePrediction:
    timestamp: float
    class_id: int
    confidence: float


class LivePredictor:
    def __init__(
        self,
        model_path: str,
        camera: int = 0,
        imgsz: int = 640,
        conf: float = 0.25,
        width: int = 1280,
        height: int = 720,
        save: Optional[str] = None,
        window_name: str = "YOLO Live Detection",
    ):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = YOLO(model_path)

        self.camera = camera
        self.imgsz = imgsz
        self.conf = conf
        self.width = width
        self.height = height
        self.save = save
        self.window_name = window_name

        self.cap = None
        self.writer = None
        self.thread = None

        self.running = False
        self.stop_event = threading.Event()
        self.ready_event = threading.Event()
        self.lock = threading.Lock()
        

        # Keep recent predictions in memory
        self.predictions = deque()

    def start(self):
        """Start live prediction in a background thread."""
        if self.running:
            return

        self.thread = threading.Thread(target=self._run, daemon=False)
        self.thread.start()

        # Wait until first frame is shown
        started = self.ready_event.wait(timeout=20)
        if not started:
            raise RuntimeError("Prediction window did not start in time.")

        self.running = True

    def stop(self):
        """Stop prediction and clean up."""
        self.stop_event.set()

        if self.thread is not None:
            self.thread.join(timeout=5)

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        if self.writer is not None:
            self.writer.release()
            self.writer = None

        cv2.destroyAllWindows()
        self.running = False

    def is_running(self) -> bool:
        return (
            self.thread is not None
            and self.thread.is_alive()
            and not self.stop_event.is_set()
        )

    def wait_until_running(self, timeout: Optional[float] = None) -> bool:
        """Wait until the prediction window is active."""
        return self.ready_event.wait(timeout=timeout)

    def get_weighted_class(self, seconds: float, threshold_average_confidence_per_frame: float = 0.5) -> Optional[int]:
        """
        Wait `seconds`, then return the class id with the highest
        total confidence over that period.

        Returns None if no predictions were made.
        """
        if seconds <= 0:
            raise ValueError("seconds must be > 0")

        if not self.wait_until_running(timeout=10):
            raise RuntimeError("Prediction is not running.")

        start_time = time.time()
        end_time = start_time + seconds

        while time.time() < end_time:
            if not self.thread.is_alive():
                break
            time.sleep(0.02)

        cutoff = time.time() - seconds

        with self.lock:
            recent = [p for p in self.predictions if p.timestamp >= cutoff]

        if not recent:
            return None

        scores = defaultdict(float)
        for pred in recent:
            scores[pred.class_id] += pred.confidence

        best_class = max(scores, key=scores.get)
        best_score = scores[best_class]

        #Calculate if the best score is above the required threshold
        min_total_confidence = threshold_average_confidence_per_frame * 30 * seconds
        print(min_total_confidence)

        if best_score < min_total_confidence:
            return None
        
        return best_class, self.model.names[best_class]

    def _run(self):
        self.cap = cv2.VideoCapture(self.camera)
        if not self.cap.isOpened():
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if self.save is not None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30.0

            frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            self.writer = cv2.VideoWriter(
                self.save,
                fourcc,
                fps,
                (frame_width, frame_height),
            )

        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                break

            results = self.model(frame, imgsz=self.imgsz, conf=self.conf, verbose=False)
            result = results[0]

            self._store_best_prediction(result)

            annotated_frame = result.plot()
            cv2.imshow(self.window_name, annotated_frame)

            if not self.ready_event.is_set():
                self.ready_event.set()

            if self.writer is not None:
                self.writer.write(annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self.stop_event.set()
                break

    def _store_best_prediction(self, result):
        """Store only the highest-confidence detection from this frame."""
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return

        best_index = 0
        best_conf = float(boxes.conf[0].item())

        for i in range(1, len(boxes)):
            conf = float(boxes.conf[i].item())
            if conf > best_conf:
                best_conf = conf
                best_index = i

        class_id = int(boxes.cls[best_index].item())

        with self.lock:
            self.predictions.append(
                FramePrediction(
                    timestamp=time.time(),
                    class_id=class_id,
                    confidence=best_conf,
                )
            )

            # Cleanup old predictions so memory stays small
            cutoff = time.time() - 30
            while self.predictions and self.predictions[0].timestamp < cutoff:
                self.predictions.popleft()


def start_live_prediction(
    model_path: str = "best.pt",
    camera: int = 0,
    imgsz: int = 640,
    conf: float = 0.25,
    width: int = 1280,
    height: int = 720,
    save: Optional[str] = None,
) -> LivePredictor:
    """
    Start live YOLO prediction and show it in a window.
    Returns a predictor object you can query later.
    """
    predictor = LivePredictor(
        model_path=model_path,
        camera=camera,
        imgsz=imgsz,
        conf=conf,
        width=width,
        height=height,
        save=save,
    )
    predictor.start()
    return predictor
