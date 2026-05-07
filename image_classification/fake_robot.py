import time
import policy_manager as pm
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

result = None

try:
    while predictor.is_running():

        while result is None:
            if not predictor.is_running():
                print("Predictor stopped while waiting for result.")
                break

            result = predictor.get_weighted_class(
                seconds=3,
                threshold_average_confidence_per_frame=0.20
            )
            if result is None:
                print("No confident prediction yet. Waiting...")

        if not predictor.is_running():
            break

        #If result is not None:
        class_id, class_name = result
        print("Predicted class id:", class_id)
        print("Predicted class name:", class_name)

        match class_id:
            case 0:
                predictor.clear_predictions()
                pm.run_act_policy_blocking(0)
                predictor.clear_predictions()

            case 1:
                predictor.clear_predictions()
                pm.run_act_policy_blocking(1)
                predictor.clear_predictions()

            case 2:
                predictor.clear_predictions()
                pm.run_act_policy_blocking(2)
                predictor.clear_predictions()

        result = None  # Reset result for the next prediction

finally:
    predictor.stop()
