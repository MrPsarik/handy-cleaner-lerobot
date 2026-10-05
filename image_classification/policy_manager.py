"""LeRobot ACT policy dispatch for the toy-sorting robot.

Import and call `run_act_policy_blocking(class_id)` from the main script;
it builds and runs the matching `lerobot-rollout` command (sentry strategy)
and blocks until the rollout finishes.

    from run_policies import run_act_policy_blocking
    run_act_policy_blocking(class_id)   # e.g. class_id from the classifier
"""

from __future__ import annotations

import subprocess
import sys

# --- Shared robot / camera configuration ---------------------------------

ROBOT_TYPE = "so101_follower"
ROBOT_PORT = "/dev/ttyACM0"
ROBOT_ID = "follower_arm"
POLICY_DEVICE = "cuda"
DISPLAY_DATA = False

# subprocess.run() with a list passes each element as one literal argv entry,
# so the braces/spaces need NO shell quoting here (unlike the shell version).
CAMERAS = (
    "{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, "
    "top: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}"
)

# --- Policy table: class_id -> policy definition -------------------------
# 0 = recycle (add once the policy is trained/uploaded)
POLICIES = {
    1: {
        "name": "scrap_paper",
        "policy_path": "MrPsarik/act_sortpaper_subset40",
        "task": "Put paper into the left bin",
        "repo_id": "MrPsarik/rollout_SortPaper_20260529_145930",
    },
    2: {
        "name": "toy",
        "policy_path": "MrPsarik/act_sorttoy",
        "task": "Put toy into the right bin",
        "repo_id": "MrPsarik/rollout_SortToy_20260527_143413",
    },
}


def build_command(policy: dict) -> list[str]:
    return [
        "lerobot-rollout",
        "--strategy.type=sentry",
        f"--policy.path={policy['policy_path']}",
        f"--policy.device={POLICY_DEVICE}",
        f"--robot.type={ROBOT_TYPE}",
        f"--robot.port={ROBOT_PORT}",
        f"--robot.id={ROBOT_ID}",
        f"--robot.cameras={CAMERAS}",
        f"--task={policy['task']}",
        f"--dataset.repo_id={policy['repo_id']}",
        f"--display_data={str(DISPLAY_DATA).lower()}",
        "--duration=20"
    ]


def run_act_policy_blocking(class_id: int) -> int:
    if class_id not in POLICIES:
        print(f"No policy configured for class_id={class_id}", file=sys.stderr)
        return 1

    policy = POLICIES[class_id]
    command = build_command(policy)

    print(f"Running policy: {policy['name']}")
    print("  " + " ".join(command))
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError:
        print(
            "Error: `lerobot-rollout` not found on PATH. "
            "Activate your lerobot environment first.",
            file=sys.stderr,
        )
        return 127
    except subprocess.CalledProcessError as exc:
        print(
            f"Policy '{policy['name']}' exited with code {exc.returncode}",
            file=sys.stderr,
        )
        return exc.returncode

    print(f"Finished policy: {policy['name']}")
    return 0
