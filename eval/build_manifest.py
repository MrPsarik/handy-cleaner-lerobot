"""Build manifest.json: one entry per recorded policy rollout on the Hub.

Reads the metadata downloaded by fetch_meta.py (meta_cache/). Videos are not downloaded;
the labelling page streams them from huggingface.co.

Skipped on purpose: rollout_SortToy_20260527_143413 and rollout_SortPaper_20260529_145930
are Hub duplicates of the demonstration datasets ("Duplicate from ..."), not policy runs.
"""
import glob
import json
import os
import re
import urllib.request
from datetime import datetime, timedelta

import pyarrow.parquet as pq

USER = "MrPsarik"
CAMS = ["observation.images.top", "observation.images.front"]
DUPLICATES = {"rollout_SortToy_20260527_143413", "rollout_SortPaper_20260529_145930"}
SESSION_GAP = timedelta(minutes=30)

# The eval_* datasets on the Hub (eval_t, eval_PaperBalls, eval_PaperBallsExt, eval_movement1)
# come from earlier design iterations and mix policy runs with teleoperated recordings under
# the same name. Nothing in the recordings says which is which, so they are left out.
EVAL_GROUPS = {}


def episodes(name):
    rows = []
    for f in sorted(glob.glob(f"meta_cache/{name}/meta/episodes/*/*.parquet")):
        for r in pq.read_table(f).to_pylist():
            rows.append({k: v for k, v in r.items() if not k.startswith("stats/")})
    return rows


def first_commit(name):
    url = f"https://huggingface.co/api/datasets/{USER}/{name}/commits/main"
    commits = json.load(urllib.request.urlopen(url))
    return datetime.fromisoformat(commits[-1]["date"].replace("Z", "+00:00")).replace(tzinfo=None)


def video(name, cam, ep):
    chunk = ep[f"videos/{cam}/chunk_index"]
    file = ep[f"videos/{cam}/file_index"]
    return {
        "url": f"https://huggingface.co/datasets/{USER}/{name}/resolve/main/videos/{cam}/"
               f"chunk-{chunk:03d}/file-{file:03d}.mp4",
        "from": ep[f"videos/{cam}/from_timestamp"],
        "to": ep[f"videos/{cam}/to_timestamp"],
    }


items = []
for path in sorted(glob.glob("meta_cache/*")):
    name = os.path.basename(path)
    if name in DUPLICATES:
        continue
    m = re.match(r"rollout_Sort(Toy|Paper)_\d{8}_\d{6}_(\d{8}_\d{6})$", name)
    if m:
        cls = m.group(1).lower()
        when = datetime.strptime(m.group(2), "%Y%m%d_%H%M%S")
        stage = "final"
    elif name in EVAL_GROUPS:
        cls = EVAL_GROUPS[name][1]
        when = first_commit(name)
        stage = EVAL_GROUPS[name][0]
    else:
        continue
    for ep in episodes(name):
        items.append({
            "id": f"{name}#{ep['episode_index']}",
            "dataset": name,
            "episode": ep["episode_index"],
            "class": cls,
            "stage": stage,
            "task": ep["tasks"][0] if ep["tasks"] else "",
            "frames": ep["length"],
            "seconds": round(ep["length"] / 30, 1),
            "recorded": when.isoformat(timespec="minutes"),
            "videos": {cam.split(".")[-1]: video(name, cam, ep) for cam in CAMS
                       if f"videos/{cam}/chunk_index" in ep},
        })

# Sessions: final-stage rollouts split where the gap between recordings exceeds SESSION_GAP;
# each tabletop eval dataset is its own session.
items.sort(key=lambda x: (x["stage"] != "final", x["recorded"], x["dataset"], x["episode"]))
prev, n = None, 0
for it in items:
    if it["stage"] == "final":
        t = datetime.fromisoformat(it["recorded"])
        if prev is None or t - prev > SESSION_GAP:
            n += 1
        prev = t
        it["session"] = f"S{n} {t:%d %b}"
    else:
        it["session"] = f"{it['dataset']} ({it['recorded'][:10]})"

json.dump(items, open("manifest.json", "w"), indent=1)
by = {}
for it in items:
    by.setdefault(it["session"], {}).setdefault(it["class"], 0)
    by[it["session"]][it["class"]] += 1
for s, c in by.items():
    print(s, c)
print(len(items), "episodes")
