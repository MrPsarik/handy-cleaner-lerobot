"""Print the Results tables for the README from manifest.json and labels.json.

    python summarize.py

Counting rules:
- "exclude" and "technical" runs are left out of the rates (empty recordings, no object in
  reach, the wrong class's policy started). They are listed separately.
- "retry_success" counts as a failure: in every such run a person helped by hand.
- A success is the object in the correct bin within the 20 s run, without help.
"""
import collections
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
items = {m["id"]: m for m in json.load(open(os.path.join(HERE, "manifest.json")))}
labels = json.load(open(os.path.join(HERE, "labels.json")))

CLASSES = [("toy", "Toy"), ("paper", "Scrap paper")]
SESSIONS = [("S1 10 Jun", "S1, lab, 10 Jun"), ("S2 12 Jun", "S2, Demo Day stand, 12 Jun")]
REASONS = [("missed_grasp", "Missed grasp"), ("shaking", "Shaking / collision"),
           ("wrong_bin", "Wrong bin"), ("no_attempt", "No attempt"), ("other", "Other"),
           (None, "No reason given")]

runs = collections.defaultdict(list)
left_out = collections.Counter()
for id_, lab in labels.items():
    it = items[id_]
    if lab["outcome"] in ("exclude", "technical"):
        left_out[(it["class"], lab["outcome"], lab.get("note", ""))] += 1
        continue
    runs[(it["session"], it["class"])].append(lab)


def rate(s, n):
    return f"{s}/{n}" if n < 20 else f"{round(100 * s / n)}%"


print("| Class | Session | Runs | Success | Helped by hand | Failure | Success rate |")
print("|---|---|---|---|---|---|---|")
for cls, cname in CLASSES:
    total = collections.Counter()
    for sess, sname in SESSIONS:
        c = collections.Counter(l["outcome"] for l in runs[(sess, cls)])
        n = sum(c.values())
        total.update(c)
        print(f"| {cname} | {sname} | {n} | {c['success']} | {c['retry_success']} | "
              f"{c['failure']} | {rate(c['success'], n)} |")
    n = sum(total.values())
    print(f"| **{cname}** | **both** | **{n}** | **{total['success']}** | "
          f"**{total['retry_success']}** | **{total['failure']}** | "
          f"**{rate(total['success'], n)}** |")

print()
print("| Failure reason | " + " | ".join(c for _, c in CLASSES) + " |")
print("|---|" + "---|" * len(CLASSES))
for key, name in REASONS:
    row = []
    for cls, _ in CLASSES:
        row.append(sum(1 for sess, _ in SESSIONS for l in runs[(sess, cls)]
                       if l["outcome"] == "failure" and l.get("reason") == key))
    if any(row):
        print(f"| {name} | " + " | ".join(map(str, row)) + " |")

print()
print("Left out of the rates:")
for (cls, outcome, note), n in sorted(left_out.items()):
    print(f"  {cls:6} {outcome:10} {n:2}  {note}")
