import subprocess
import uuid
from pathlib import Path


ROBOT_PORT = "COM3"  # Windows example. Linux example: "/dev/ttyACM0"
ROBOT_ID = "my_so101_follower"

CAMERA_CONFIG = "{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}"

POLICIES = {
    0: {
        "name": "recycle",
        "policy_path": "YOUR_HF_USERNAME/recycle_act_policy",
        "task": "Put the recyclable object in the recycle bin",
    },
    1: {
        "name": "scrap_paper",
        "policy_path": "YOUR_HF_USERNAME/scrap_paper_act_policy",
        "task": "Put the scrap paper in the paper bin",
    },
    2: {
        "name": "toy",
        "policy_path": "YOUR_HF_USERNAME/toy_act_policy",
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

    eval_repo_id = f"local/eval_{policy['name']}_{uuid.uuid4().hex[:8]}"

    command = [
        "python", "-m", "lerobot.record",
        "--robot.type=so101_follower",
        f"--robot.port={ROBOT_PORT}",
        f"--robot.id={ROBOT_ID}",
        f"--robot.cameras={CAMERA_CONFIG}",
        "--display_data=false",
        f"--dataset.repo_id={eval_repo_id}",
        "--dataset.num_episodes=1",
        "--dataset.push_to_hub=false",
        f"--dataset.single_task={policy['task']}",
        "--dataset.episode_time_s=10",
        "--dataset.reset_time_s=1",
        f"--policy.path={policy['policy_path']}",
    ]

    print(f"Running ACT policy: {policy['name']}")
    subprocess.run(command, check=True)
    print(f"Finished ACT policy: {policy['name']}")