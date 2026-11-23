"""Evaluate a trained checkpoint on the validation AND test splits, for a PopSteer steering grid, a re-ranking
baseline grid, or the plain model. Writes one CSV row per (split, setting) with all metrics.

Run from the working folder that holds ./dataset/<name>/ (the folder the checkpoint was trained in). Neuron labels
are written to ./dataset/<name>/user/ (U-S, U-R, UI-S, UI-R) or ./dataset/<name>/item/ (I-R) before steering, so
run variants that share a label folder one after the other, not in parallel, or from separate working folders.

  python scripts/steer_eval.py --path saved/<ckpt>.pth --variant U-S [--analyze_batches 200]
  python scripts/steer_eval.py --path saved/<ckpt>.pth --method ipr

PopSteer variants (--variant; SAE checkpoint type in brackets)
  U-S   user SAE, labels from SYNTHETIC users (original PopSteer)   [SASRec_SAE; SASRec only]
  U-R   user SAE, labels from REAL users: top vs bottom 10% users by mean head/tail label of their history
                                                                    [SASRec_SAE, LightGCN_SAE]
  I-R   item SAE, labels from REAL items: head vs tail 10% items   [SASRec_ISAE, LightGCN_ISAE]
  UI-S  joint user+item SAE, labels from SYNTHETIC users            [SASRec_JSAE; SASRec only]
  UI-R  joint user+item SAE, labels from REAL users (as U-R)        [SASRec_JSAE, LightGCN_JSAE]
Other methods (--method)
  none                                  plain model; for an SAE checkpoint: reconstruction without steering + cosine
  ipr, fair (FA*IR), pct, min_reg (P-MMF), duor, random       re-ranking baselines on a backbone checkpoint

Steering parameters per row: param1 = alpha_pop, param2 = alpha_unpop, param3 = beta (|d_j| threshold); the
unsteered reference row has param3 = -1. Baselines: param1 / param2 as in GRIDS below.
Paper setting: residual-preserving steering (default), --grid auto (SASRec: "full", LightGCN: "low"; baselines:
"www"), I-R with --align, UI-S / UI-R with --sides both.
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # make the bundled recbole/ importable

from recbole.quick_start import load_data_and_model  # noqa: E402
from recbole.utils import get_trainer  # noqa: E402

VARIANTS = {"U-S": "ps", "U-R": "ur", "I-R": "ir", "UI-S": "uis", "UI-R": "uir"}   # public name -> internal code
NAMES = {v: k for k, v in VARIANTS.items()}
BASELINES = ["ipr", "fair", "pct", "min_reg", "duor", "random"]

METRICS = ["NDCG", "Recall", "Hit", "GiniIndex", "AveragePopularity", "ItemCoverage", "ItemCoverageN", "UPD",
           "NDCGTail", "NDCGHead", "NDCGMid"]
ALPHAS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
LAMBDAS = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
STEER_GRID = [(ap, au, b) for b in [0.0, 0.5, 1.0, 1.5, 2.0] for ap in ALPHAS for au in ALPHAS]
# "www" grid: the grids of the conference paper (also the baseline grids of the extended paper).
GRIDS = {  # (param1, param2, param3) per configuration
    "none": [(0, 0, 0)],
    "ps": STEER_GRID, "ur": STEER_GRID, "ir": STEER_GRID,   # alpha_pop, alpha_unpop, beta
    "uir": STEER_GRID, "uis": STEER_GRID,                   # x sides (user, item, both)
    "ipr": [(a, 0, 0) for a in LAMBDAS],                    # alpha
    "min_reg": [(a, 0, 0) for a in LAMBDAS],                # lambda
    "fair": [(p, a, 0) for p in LAMBDAS for a in [0.01, 0.05, 0.1]],     # target share p, significance alpha
    "pct": [(p, l, 0) for p in LAMBDAS for l in [0.01, 0.05, 0.1]],      # p, alpha (as FA*IR)
    "duor": [(t, 0, 0) for t in [50, 100, 250, 500, 1000]],              # list length t
    "random": [(t, 0, 0) for t in [15, 30, 50, 75, 100]],                # top-t pool
}
# Extension: gentler steering (alpha < 1, beta > 2) and finer baseline grids. `--grid ext` evaluates ONLY the
# configurations that are not in the "www" grid; `--grid full` evaluates both.
ALPHAS_EXT = [0.1, 0.25, 0.5, 0.75] + ALPHAS
STEER_FULL = [(ap, au, b) for b in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0] for ap in ALPHAS_EXT for au in ALPHAS_EXT]
LAMBDAS_EXT = [0.005, 0.02, 0.03, 0.07, 0.15, 0.3, 1.5, 2.0, 3.0]
GRIDS_EXT = {
    "none": [],
    "ps": STEER_FULL, "ur": STEER_FULL, "ir": STEER_FULL, "uir": STEER_FULL, "uis": STEER_FULL,
    "ipr": [(a, 0, 0) for a in LAMBDAS_EXT],
    "min_reg": [(a, 0, 0) for a in LAMBDAS_EXT],
    "fair": [(p, a, 0) for p in [0.005, 0.02, 0.03, 0.3] for a in [0.01, 0.05, 0.1]],
    "pct": [(p, l, 0) for p in [0.005, 0.02, 0.03, 0.3] for l in [0.01, 0.05, 0.1]],
    "duor": [(t, 0, 0) for t in [20, 30, 40, 75, 150, 200]],
    "random": [(t, 0, 0) for t in [12, 20, 25, 40]],
}
# "low" grid (LightGCN): its recommendations respond much more strongly to steering, so the strengths are restricted
# to the gentle region alpha <= 1 (175 settings).
ALPHAS_LOW = [0.1, 0.25, 0.5, 0.75, 1.0]
STEER_LOW = [(ap, au, b) for b in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0] for ap in ALPHAS_LOW for au in ALPHAS_LOW]
# "mid" grid: complements "low" with stronger settings only (at least one alpha > 1, alphas up to 3, same betas);
# together with "low" it forms one grid.
ALPHAS_MID = ALPHAS_LOW + [1.5, 2.0, 3.0]
STEER_MID = [(ap, au, b) for b in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0] for ap in ALPHAS_MID for au in ALPHAS_MID
             if max(ap, au) > 1.0]
STEERING = ("ps", "ur", "ir", "uir", "uis")
JOINT = ("uir", "uis")
SYNTHETIC = ("ps", "uis")
ITEM_RULES = ("ir", "uir", "uis")     # variants with item-side steering (align / gate rules)

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--path", required=True, help="checkpoint (.pth) to evaluate")
who = ap.add_mutually_exclusive_group(required=True)
who.add_argument("--variant", choices=list(VARIANTS), help="PopSteer variant (steering grid)")
who.add_argument("--method", choices=["none"] + BASELINES, help="plain model or re-ranking baseline")
ap.add_argument("--splits", default="valid,test")
ap.add_argument("--out", default=None, help="output CSV (default: tune_<variant or method>[_<rules>][_<grid>].csv)")
ap.add_argument("--analyze_batches", type=int, default=None,
                help="U-S / UI-S: run the synthetic-user neuron analysis first with this many batches of "
                     "train_batch_size users per group (default 200: 200 x 2048 = 409,600); 0 = reuse the existing "
                     "statistics in ./dataset/<name>/user/")
ap.add_argument("--grid", default="auto", choices=["auto", "www", "ext", "full", "low", "mid"],
                help="auto (default): 'full' for steering on SASRec, 'low' for steering on LightGCN, 'www' for "
                     "baselines; 'www' = conference-paper grid, 'ext' = only the extension, 'full' = both, "
                     "'low' / 'mid' = gentle / stronger steering grids (steering only)")
ap.add_argument("--resume", action="store_true",
                help="continue from <out>.partial.csv (rows are appended there as they finish)")
ap.add_argument("--residual", action=argparse.BooleanOptionalAction, default=True,
                help="residual-preserving steering (default on): x' = x + (x_hat_steered - x_hat_unsteered), "
                     "which keeps the reconstruction error; --no-residual returns the steered reconstruction x_hat")
ap.add_argument("--gate", action="store_true",
                help="gate rule on the ITEM side (I-R, UI-S, UI-R): steer only neurons in each item's own TopK support")
ap.add_argument("--align", action="store_true",
                help="align rule on the ITEM side (I-R, UI-S, UI-R): delta_j <- delta_j * sign(p_bar^T w_j), "
                     "p_bar = mean validation-input user embedding")
ap.add_argument("--sides", default="user,item,both",
                help="joint SAE (UI-S, UI-R): steered sides to evaluate, comma-separated from user,item,both")
ap.add_argument("--limit", type=int, default=0, help="smoke tests: evaluate only the first N grid settings")
ap.add_argument("--ablation", choices=["none", "noise", "randsel"], default=None,
                help="ablation at fixed --settings: 'randsel' moves PopSteer's edits to random neurons, 'noise' "
                     "replaces them by zero-mean Gaussian edits of the same per-neuron SD; 'none' = PopSteer itself")
ap.add_argument("--seeds", default="0", help="--ablation: comma-separated seeds of the random draw")
ap.add_argument("--settings", default="",
                help="evaluate only these settings instead of a grid: 'alpha_pop:alpha_unpop:beta[:side],...'")
args = ap.parse_args()
args.method = VARIANTS[args.variant] if args.variant else args.method
NAME = args.variant or args.method
if args.method not in STEERING:
    args.residual = False      # residual-preserving steering applies to the SAE steering variants only
assert not (args.gate and args.method not in ITEM_RULES), "--gate applies to item-side steering (I-R, UI-S, UI-R)"
assert not (args.align and args.method not in ITEM_RULES), "--align applies to item-side steering (I-R, UI-S, UI-R)"
if args.analyze_batches is None:
    args.analyze_batches = 200 if args.method in SYNTHETIC else 0

overrides = {"metrics": METRICS, "topk": [10], "valid_metric": "NDCG@10"}
if args.method in STEERING:
    overrides.update({"steer": True, "alpha_pop": 1.0, "alpha_unpop": 1.0, "D": 0.0})
if args.analyze_batches:
    overrides["analyze_batches"] = args.analyze_batches
config, model, dataset, train_data, valid_data, test_data = load_data_and_model(args.path, dict=overrides)
trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
trainer.eval_collector.data_collect(train_data)
splits = {"valid": valid_data, "test": test_data}

grid_name = args.grid
if grid_name == "auto":
    if args.method in STEERING:
        grid_name = "low" if str(config["model"]).startswith("LightGCN") else "full"
    else:
        grid_name = "www"
if grid_name == "ext":
    GRIDS = {m: [c for c in GRIDS_EXT[m] if c not in set(GRIDS[m])] for m in GRIDS}
elif grid_name == "full":
    GRIDS = {m: GRIDS[m] + [c for c in GRIDS_EXT[m] if c not in set(GRIDS[m])] for m in GRIDS}
elif grid_name == "low":
    assert args.method in STEERING, "--grid low is a steering grid"
    GRIDS = dict(GRIDS, **{args.method: STEER_LOW})
elif grid_name == "mid":
    assert args.method in STEERING, "--grid mid is a steering grid"
    GRIDS = dict(GRIDS, **{args.method: STEER_MID})

sae = getattr(model, "sae_module_i" if args.method == "ir" else "sae_module_u", None)
if args.method in STEERING:
    assert sae is not None, f"{NAME} needs an SAE checkpoint of the matching type"
if args.residual:
    sae.residual = True
if args.gate:
    if args.method == "ir":
        sae.gate = True
    else:
        model.item_gate = True     # joint SAE: the gate is applied to the item side only
if args.align:
    with torch.no_grad():
        model.eval()
        if hasattr(model, "_base"):                      # LightGCN: propagated user embeddings
            p_bar = model._base()[0][1:].mean(0)
        else:                                            # SASRec: validation-input user states (raw, unstandardised)
            from recbole.model.sequential_recommender.sasrec import SASRec
            p_sum, n_u = 0.0, 0
            for batch in valid_data:
                inter = batch[0].to(trainer.device)
                x = SASRec.forward(model, inter[model.ITEM_SEQ], inter[model.ITEM_SEQ_LEN])
                p_sum, n_u = p_sum + x.sum(0), n_u + x.shape[0]
            p_bar = p_sum / n_u
        sae.align_sign = torch.sign(sae.W_dec @ p_bar).detach()
    print(f"align: {int((sae.align_sign < 0).sum())} of {sae.align_sign.numel()} neurons flipped", flush=True)
    if args.method == "ir":
        sae.use_align = True
    else:
        model.item_align = True    # joint SAE: item side only
RULE = "+".join(v for v, on in (("gate", args.gate), ("align", args.align), ("res", args.residual)) if on)
if args.ablation:
    assert args.method in STEERING and args.settings, "--ablation needs a steering variant and --settings"
    sae.ablation = None if args.ablation == "none" else args.ablation
SEEDS = [int(s) for s in args.seeds.split(",")] if args.ablation else [""]
if sae is None and hasattr(model, "sae_module_i") and args.method == "none":
    sae = model.sae_module_i
base = f"./dataset/{config['dataset']}"


def write_stats(side, z_all, z_pop, z_unpop):
    """Neuron statistics (means, SDs, Cohen's d) in the format read by the SAE: ./dataset/<name>/<side>/."""
    os.makedirs(f"{base}/{side}", exist_ok=True)
    for name, z in (("neuron_stats", z_all), ("neuron_stats_pop", z_pop), ("neuron_stats_unpop", z_unpop)):
        pd.DataFrame({"mean": z.mean(0), "sd": z.std(0)}).to_csv(f"{base}/{side}/{name}.csv")
    n1, n2 = len(z_pop), len(z_unpop)
    pooled = np.sqrt(((n1 - 1) * z_pop.var(0) + (n2 - 1) * z_unpop.var(0)) / (n1 + n2 - 2))
    pd.DataFrame({"cohens_d": (z_pop.mean(0) - z_unpop.mean(0)) / pooled}).to_csv(f"{base}/{side}/cohens_d.csv")
    print(f"labels written to {base}/{side}: {n1} pop vs {n2} unpop", flush=True)


@torch.no_grad()
def item_labels():
    """I-R: pre-activations of every real item; head (label 1) vs tail (label -1) items."""
    sae.steer = False
    table = model._base()[1][1:] if hasattr(model, "_base") else model.item_embedding.weight[1:]   # LightGCN: propagated
    z = sae.encoder(table - sae.b_dec).cpu().numpy()            # pre-activations, item ids 1..n
    lab = pd.read_csv(f"{base}/item_popularity_labels.csv", usecols=["item_id:token", "popularity_label"])
    lab = dict(zip(lab["item_id:token"].astype(int), lab["popularity_label"].astype(int)))
    y = np.array([lab.get(i, 0) for i in range(1, len(table) + 1)])
    write_stats("item", z, z[y == 1], z[y == -1])


@torch.no_grad()
def real_user_labels():
    """U-R / UI-R: pre-activations of real users' validation-input states (training history only); groups = the user
    popularity_label (top / bottom 10% of users by mean item label). LightGCN: the propagated user embedding."""
    sae.steer = False
    model.eval()
    lab = pd.read_csv(f"{base}/user_popularity_labels.csv", usecols=["user_id:token", "popularity_label"])
    grp = lab.groupby("user_id:token")["popularity_label"].first()
    if hasattr(model, "base_u"):                     # LightGCN_SAE: embeddings of all users, no sequences
        if hasattr(model, "_base"):                  # LightGCN_JSAE / _ISAE
            model._base()
        elif model.base_u is None:
            model.forward(train_mode=False)
        uids = np.arange(1, model.base_u.shape[0])
        x = model.base_u[1:]
        if hasattr(model, "std_u"):                  # joint SAE: users are standardised before the shared SAE
            x = model.std_u(x)
        z = sae.encoder(x - sae.b_dec).cpu().numpy()
        g = grp.reindex(uids).fillna(0).to_numpy()
        write_stats("user", z, z[g == 1], z[g == -1])
        return
    zs, uids = [], []
    for batch in valid_data:
        inter = batch[0].to(trainer.device)
        if hasattr(model, "_user_rec"):      # joint SAE: user states pass the shared SAE after standardising
            model._user_rec(inter[model.ITEM_SEQ], inter[model.ITEM_SEQ_LEN], steer=False)
        else:
            model.forward(inter[model.ITEM_SEQ], inter[model.ITEM_SEQ_LEN])
        zs.append(sae.last_activations.cpu().numpy())
        uids.append(inter[model.USER_ID].cpu().numpy())
    z, uids = np.concatenate(zs), np.concatenate(uids)
    g = grp.reindex(uids).fillna(0).to_numpy()
    write_stats("user", z, z[g == 1], z[g == -1])


if args.method in SYNTHETIC and args.analyze_batches:   # synthetic users; steering must be off meanwhile
    sae.steer = False
    os.makedirs(f"{base}/user", exist_ok=True)       # save_mean_SD does not create it
    t0 = time.time()
    trainer.analyze_neurons(train_data, model_file=args.path, eval_data=False, show_progress=False)
    print(f"neuron analysis with {args.analyze_batches} batches done in {time.time() - t0:.0f}s", flush=True)
elif args.method == "ir":
    item_labels()
elif args.method in ("ur", "uir"):
    real_user_labels()

cos_acc = {"sum": 0.0, "n": 0}   # cosine(original, reconstructed) on the unsteered pass of an SAE checkpoint


def cos_hook(module, inputs, output):
    c = F.cosine_similarity(output.detach(), inputs[0].detach(), dim=1)
    cos_acc["sum"] += float(c.sum())
    cos_acc["n"] += c.numel()


def configure(p1, p2, p3):
    for cache in ("restore_user_e", "restore_item_e"):   # LightGCN caches propagated embeddings between batches
        if hasattr(model, cache):
            setattr(model, cache, None)
    if args.method == "none" or (args.method in STEERING and p3 == -1):
        if sae is not None:
            sae.steer = False
        return
    if args.method in STEERING:
        sae.steer = True
        sae.alpha_pop, sae.alpha_unpop, sae.d_min = p1, p2, p3
        sae.N = sae.hidden_dim
        sae._steer_ready = False
        return
    for flag in ("fair", "random", "ipr", "pct", "min_reg", "duor"):
        setattr(model, flag, flag == args.method)
    model.param1, model.param2 = p1, p2
    model.recommendation_count = torch.zeros(model.n_items, dtype=torch.long, device=trainer.device)


default_grid = ("low" if str(config["model"]).startswith("LightGCN") else "full") if args.method in STEERING else "www"
out = args.out or ("tune_" + NAME + ("_gate" if args.gate else "") + ("_align" if args.align else "")
                   + ("_nores" if args.method in STEERING and not args.residual else "")
                   + (f"_abl_{args.ablation}" if args.ablation else "")
                   + ("" if grid_name == default_grid else f"_{grid_name}") + ".csv")
partial = out + ".partial.csv"     # one row per finished setting, so a time limit loses nothing (--resume)
rows, done = [], set()
if args.resume and os.path.exists(partial):
    rows = list(csv.DictReader(open(partial, newline="")))
    done = {(r["split"], r["param1"], r["param2"], r["param3"], r["side"], r.get("seed", "")) for r in rows}
    print(f"resume: {len(rows)} rows from {partial}", flush=True)
elif os.path.exists(partial):
    os.remove(partial)
pw = None
for split in args.splits.split(","):
    grid = ([(0, 0, -1)] if args.method in STEERING and grid_name not in ("ext", "mid") else []) + GRIDS[args.method]
    if args.limit:   # before the side expansion, so joint variants still test every side
        grid = grid[:args.limit]
    if args.method in JOINT:
        n0 = 1 if grid and grid[0][2] == -1 else 0   # the unsteered row (absent in the ext / mid grids) runs once
        grid = [(p1, p2, p3, "both") for p1, p2, p3 in grid[:n0]] + \
               [(p1, p2, p3, side) for side in args.sides.split(",") for p1, p2, p3 in grid[n0:]]
    else:
        grid = [(p1, p2, p3, "") for p1, p2, p3 in grid]
    if args.settings:   # explicit settings (ablation): no grid
        grid = []
        for s in args.settings.split(","):
            f = s.split(":")
            grid.append((float(f[0]), float(f[1]), float(f[2]), f[3] if len(f) > 3 else ""))
    grid = [(p1, p2, p3, side, seed) for p1, p2, p3, side in grid for seed in SEEDS]
    for p1, p2, p3, side, seed in grid:
        if (split, str(p1), str(p2), str(p3), side, str(seed)) in done:
            continue
        t0 = time.time()
        if side:
            model.steer_side = side
        if args.ablation:
            sae.ablation_seed = seed
        configure(p1, p2, p3)
        handle = None
        if sae is not None and (args.method == "none" or p3 == -1):
            cos_acc.update(sum=0.0, n=0)
            handle = sae.register_forward_hook(cos_hook)
        res = trainer.evaluate(splits[split], load_best_model=False, model_file=args.path, show_progress=False)
        if handle is not None:
            handle.remove()
        row = {"method": NAME, "rule": RULE, "grid": grid_name,
               "split": split, "param1": p1, "param2": p2, "param3": p3, "side": side,
               "ablation": args.ablation or "", "seed": seed,
               "n_steered": int(sae.N) if args.method in STEERING and p3 >= 0 else "",
               "cosine": round(cos_acc["sum"] / cos_acc["n"], 5) if handle is not None and cos_acc["n"] else "",
               "seconds": round(time.time() - t0, 2)}
        row.update({k: float(v) for k, v in res.items() if not k.startswith("sae_loss")})
        rows.append(row)
        print(row, flush=True)
        new = not os.path.exists(partial)
        with open(partial, "a", newline="") as f:
            pw = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if new:
                pw.writeheader()
            pw.writerow(row)

cols = list(dict.fromkeys(c for r in rows for c in r))
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} rows -> {out}")
