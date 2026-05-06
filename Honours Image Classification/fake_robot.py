import time

import object_detector as od
predictor = od.start_live_prediction(
    model_path="best.pt",
    camera=0,
    imgsz=640,
    conf=0.25,
    width=1280,
    height=720,
    save=None,
)

predictor.wait_until_running()

try:
    while predictor.is_running():
        result = predictor.get_weighted_class(
            seconds=3,
            threshold_average_confidence_per_frame=0.5
        )

        if result is None:
            print("No confident prediction.")
            continue

        class_id, class_name = result
        print("Predicted class id:", class_id)
        print("Predicted class name:", class_name)

finally:
    predictor.stop()