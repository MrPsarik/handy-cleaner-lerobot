import subprocess
import uuid
from pathlib import Path


ROBOT_PORT = "/dev/ttyACM0"
ROBOT_ID = "my_so101_follower"

CAMERA_CONFIG = "{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}"

POLICIES = {
    0: {
        "name": "recycle",
        "policy_path": "YOUR_HF_USERNAME/recycle_act_policy", # Ommited for now
        "task": "Put the recyclable object in the recycle bin",
    },
    1: {
        "name": "scrap_paper",
        "policy_path": "MrPsarik/movement1", # Dummy movement 1
        "task": "Put the scrap paper in the paper bin",
    },
    2: {
        "name": "toy",
        "policy_path": "MrPsarik/movement2", # Dummt movement 2
        "task": "Put the toy in the toy bin",
    },
}


def run_act_policy_blocking(class_id: int):
    """
    Runs one ACT policy on the SO-ARM101 / SO101 follower robot.
    This function blocks until LeRobot finishes.
    """

    if class_id not in POLICIES:
        print(f"No policy configured for class_id={class_id}")
        return

    policy = POLICIES[class_id]

    LEROBOT_REPLAY = Path(sys.executable).parent / "lerobot-replay"

    command = [
        str(LEROBOT_REPLAY), 
        "--robot.type=so101_follower",
        "--robot.port=/dev/ttyACM0",
        "--robot.id=follower_arm",
        f"--dataset.repo_id=MrPsarik/movement{class_id}", 
        "--dataset.episode=0"
    ]

    print(f"Running ACT policy: {policy['name']}")
    subprocess.run(command, check=True)
    print(f"Finished ACT policy: {policy['name']}")
