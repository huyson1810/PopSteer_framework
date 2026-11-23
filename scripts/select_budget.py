"""Budget-based model selection over the CSVs written by steer_eval.py.

For an nDCG budget b, each method keeps the configurations whose VALIDATION nDCG@10 is at least (1 - b) times the
backbone's validation nDCG@10, picks among them the one with the lowest VALIDATION Gini Index, and reports that
configuration's TEST metrics. No test value is used to choose anything.

  python scripts/select_budget.py --backbone runs/sasrec/tune_none.csv runs/sasrec/tune_*.csv \
      [--budgets 0.01,0.05,0.10] [--by_side] [--out selection.csv]

  --backbone   tune_none.csv of the BACKBONE checkpoint (steer_eval.py --method none on the backbone .pth)
  files        tune_*.csv of the variants and baselines; rows of different files are kept apart, and each file is
               one method (identified by its `method` and `rule` columns, plus `side` with --by_side)
Excluded: the unsteered reference rows of the steering variants (param3 = -1), ablation rows, and partial files.
An extra method "PopSteer (best variant)" takes, at each budget, the variant whose selection has the lowest
validation Gini Index.
"""
import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

VARIANTS = ("U-S", "U-R", "I-R", "UI-S", "UI-R")
KEY = ["param1", "param2", "param3", "side", "seed"]

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--backbone", required=True, help="tune_none.csv of the backbone checkpoint")
ap.add_argument("files", nargs="+", help="tune_*.csv files (globs are expanded)")
ap.add_argument("--budgets", default=",".join(f"{b / 100:.2f}" for b in range(1, 16)),
                help="comma-separated relative nDCG budgets (default 0.01 ... 0.15)")
ap.add_argument("--accuracy", default="ndcg@10", help="accuracy column constrained by the budget")
ap.add_argument("--objective", default="giniindex@10", help="column minimised on validation")
ap.add_argument("--by_side", action="store_true", help="joint SAEs: select separately per steered side")
ap.add_argument("--out", default=None, help="write the selections to this CSV")
a = ap.parse_args()


def read(path):
    d = pd.read_csv(path, dtype={k: str for k in KEY + ["rule", "grid", "ablation"]}, keep_default_na=False)
    for c in KEY + ["rule", "ablation"]:
        if c not in d:
            d[c] = ""
    d["file"] = Path(path).name
    return d


bb = read(a.backbone)
bb_valid = bb[bb.split == "valid"]
bb_test = bb[bb.split == "test"]
if bb_valid.empty or bb_test.empty:
    sys.exit(f"{a.backbone}: needs one 'valid' and one 'test' row")
base_v = float(bb_valid[a.accuracy].iloc[0])
print(f"backbone validation {a.accuracy} = {base_v:.5f}")

paths = sorted({p for f in a.files for p in (glob.glob(f) or [f])
                if not p.endswith(".partial.csv") and Path(p).resolve() != Path(a.backbone).resolve()})
frames = []
for p in paths:
    d = read(p)
    d = d[d.ablation.astype(str) == ""]
    steering = d.method.isin(VARIANTS)
    d = d[~(steering & (d.param3.astype(float) < 0))]          # unsteered reference row
    if d.method.eq("none").all():
        continue                                               # plain-model files are not methods
    frames.append(d)
if not frames:
    sys.exit("no method rows found")
data = pd.concat(frames, ignore_index=True)
group_cols = ["file", "method", "rule"] + (["side"] if a.by_side else [])
metric_cols = [c for c in data.columns if "@" in c]


def select(g, b):
    v = g[g.split == "valid"]
    ok = v[v[a.accuracy].astype(float) >= (1 - b) * base_v]
    if ok.empty:
        return None
    ok = ok.assign(_obj=ok[a.objective].astype(float), _acc=-ok[a.accuracy].astype(float))
    best = ok.sort_values(["_obj", "_acc"], kind="stable").iloc[0]
    t = g[(g.split == "test")]
    for k in KEY:
        t = t[t[k] == best[k]]
    if t.empty:
        return None
    t = t.iloc[0]
    row = {"valid_" + a.accuracy: float(best[a.accuracy]), "valid_" + a.objective: float(best[a.objective])}
    row.update({k: best[k] for k in KEY})
    row.update({"test_" + c: float(t[c]) for c in metric_cols if c in t and t[c] != ""})
    return row


out = []
for b in [float(x) for x in a.budgets.split(",")]:
    out.append({"budget": b, "method": "Backbone", **{"test_" + c: float(bb_test[c].iloc[0])
                                                        for c in metric_cols if c in bb_test}})
    best_variant = None
    for keys, g in data.groupby(group_cols, sort=False):
        info = dict(zip(group_cols, keys))
        sel = select(g, b)
        rec = {"budget": b, **info, "selected": sel is not None, **(sel or {})}
        out.append(rec)
        if sel is not None and info["method"] in VARIANTS:
            if best_variant is None or sel["valid_" + a.objective] < best_variant["valid_" + a.objective]:
                best_variant = rec
    if best_variant is not None:
        out.append({**best_variant, "method": "PopSteer (best variant)",
                    "chosen": f"{best_variant['method']} {best_variant['rule']}".strip()})

res = pd.DataFrame(out)
first = list(dict.fromkeys(["budget", "method", "rule", "side", "chosen", "file", "selected"] + KEY))
res = res[[c for c in first if c in res] + [c for c in res if c not in first]]
if a.out:
    res.to_csv(a.out, index=False)
    print(f"{len(res)} rows -> {a.out}")
label = res.method + res.get("rule", pd.Series("", index=res.index)).fillna("").map(lambda r: f" [{r}]" if r else "")
if a.by_side and "side" in res:
    label = label + res.side.fillna("").map(lambda s: f" ({s})" if s else "")
col = "test_" + a.objective
if col in res:
    table = res.assign(label=label).pivot_table(index="label", columns="budget", values=col, aggfunc="first")
    print(f"\ntest {a.objective} of the selected configuration (columns: budget)")
    print(table.round(4).to_string())
