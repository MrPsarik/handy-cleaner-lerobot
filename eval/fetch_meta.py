"""Download metadata (no videos) of every rollout dataset on the Hub into meta_cache/."""
import json, urllib.request
from huggingface_hub import snapshot_download

USER = "MrPsarik"
ids = [d["id"] for d in json.load(urllib.request.urlopen(
    f"https://huggingface.co/api/datasets?author={USER}&limit=500"))]
ids = [i for i in ids if "rollout_" in i or i.split("/")[1] in
       ("eval_t", "eval_PaperBalls", "eval_PaperBallsExt", "eval_movement1")]
for i in ids:
    snapshot_download(i, repo_type="dataset", local_dir=f"meta_cache/{i.split('/')[1]}",
                      allow_patterns=["meta/*", "meta/**", "README.md"])
print(len(ids), "datasets")
