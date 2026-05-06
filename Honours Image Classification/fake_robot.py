import time

import object_detector as od
predictor = od.start_live_prediction(
    model_path="Honours Image Classification/best.pt",
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
                threshold_average_confidence_per_frame=0.5
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
            case 0: #Recycle
                time.sleep(2)  # Simulate some processing time
            case 1: #Scrap_Paper
                time.sleep(2)  # Simulate some processing time
            case 2: #Toy
                time.sleep(2)  # Simulate some processing time        

        result = None  # Reset result for the next prediction

finally:
    predictor.stop()