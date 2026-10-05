"""Main toy-sorting loop.

Each cycle runs the detector as a SEPARATE process (detect_once.py). When it
exits, /dev/video2 is released unconditionally by the OS, so the lerobot
rollout always gets a free camera. This sidesteps the fact that a stuck
OpenCV read() can't be interrupted from within the detector's own process.
"""

import os
import sys
import time
import subprocess

import policy_manager as pm

HERE = os.path.dirname(os.path.abspath(__file__))
DETECT_SCRIPT = os.path.join(HERE, "detect_once.py")

# Exit codes from detect_once.py
EXIT_RESULT = 0
EXIT_STOPPED = 1
EXIT_CAMERA = 3

CAMERA_RETRY_S = 1.0
CAMERA_MAX_RETRIES = 10


def detect_class():
    """Spawn the detector, wait for one prediction, return (class_id, name).

    Returns None if the detector stopped without a prediction or the camera
    never became available.
    """
    retries = 0
    while True:
        proc = subprocess.run(
            [sys.executable, DETECT_SCRIPT],
            stdout=subprocess.PIPE,
            text=True,
        )

        if proc.returncode == EXIT_RESULT:
            for line in proc.stdout.splitlines():
                if line.startswith("RESULT\t"):
                    _, class_id, class_name = line.split("\t", 2)
                    return int(class_id), class_name
            return None  # exited 0 but no RESULT line (shouldn't happen)

        if proc.returncode == EXIT_CAMERA and retries < CAMERA_MAX_RETRIES:
            retries += 1
            print(
                f"Camera not ready, retrying detector "
                f"({retries}/{CAMERA_MAX_RETRIES})..."
            )
            time.sleep(CAMERA_RETRY_S)
            continue

        return None  # stopped, or gave up waiting for the camera


def main():
    while True:
        result = detect_class()  # detector runs, then EXITS -> camera freed
        if result is None:
            print("Detector stopped without a prediction. Exiting.")
            break

        class_id, class_name = result
        print("Predicted class id:", class_id)
        print("Predicted class name:", class_name)

        pm.run_act_policy_blocking(class_id)  # camera is free here


if __name__ == "__main__":
    main()
