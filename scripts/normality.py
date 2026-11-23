"""Normality diagnostics of SAE neuron activations (skewness and excess kurtosis per neuron).

Run from the working folder that holds ./dataset/<name>/ of the checkpoints:
  python scripts/normality.py --path saved/<sae_1>.pth [saved/<sae_2>.pth ...] --out normality.csv

  - variable: the SAE pre-activations z = W_enc^T (x - b_pre) before TopK and without steering;
  - inputs: real users' training sequences from the TRAIN dataloader (SASRec; capped at --max_samples), or all users'
    propagated embeddings (LightGCN); items: the item embedding table (item and joint SAEs);
  - per neuron: mean, SD with ddof = 1, Z = (x - mean) / SD; skew = mean(Z^3), excess kurtosis = mean(Z^4) - 3
    (population moments, no bias correction); a neuron with SD = 0 gives NaN and counts as NOT passing;
  - reported: % of neurons with |skew| < 0.5 and with |excess kurtosis| < 1, over ALL neurons and over the ACTIVE
    neurons (in the TopK support, with a positive value, of at least one input).
The SAE type (user / item / joint) is read from the checkpoint's model class. Output: one row per SAE and input side.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # make the bundled recbole/ importable

from recbole.model.sequential_recommender.sasrec import SASRec  # noqa: E402
from recbole.quick_start import load_data_and_model  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--path", required=True, nargs="+", help="SAE checkpoints (.pth)")
ap.add_argument("--out", required=True)
ap.add_argument("--max_samples", type=int, default=100000)
a = ap.parse_args()
torch.set_grad_enabled(False)


def sae_kind(model_name):
    if model_name.endswith("_ISAE"):
        return "item"
    if model_name.endswith("_JSAE"):
        return "joint"
    if model_name.endswith("_SAE"):
        return "user"
    raise SystemExit(f"{model_name} is not an SAE checkpoint")


def moments(z):
    """Per-neuron skew and excess kurtosis of a [samples, neurons] array (NaN if SD = 0)."""
    mu, sd = z.mean(0), z.std(0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = (z - mu) / np.where(sd > 0, sd, np.nan)
        return np.nanmean(Z ** 3, axis=0), np.nanmean(Z ** 4, axis=0) - 3.0


def active(z, k):
    zt = torch.tensor(z)
    idx = torch.topk(torch.relu(zt), k, dim=1).indices
    pos = torch.gather(zt, 1, idx) > 0
    m = np.zeros(z.shape[1], dtype=bool)
    m[idx[pos].unique().numpy()] = True
    return m


rows = []
for ckpt in a.path:
    overrides = {"steer": False, "alpha_pop": 1.0, "alpha_unpop": 1.0, "D": 0.0}
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(ckpt, dict=overrides)
    kind = sae_kind(str(config["model"]))
    model.eval()
    sae = model.sae_module_i if kind == "item" else model.sae_module_u
    sae.steer = False
    dev = sae.W_dec.device
    joint = kind == "joint"
    stdz = (lambda x, m: (x - getattr(model, f"mu_{m}")) / getattr(model, f"sd_{m}")) if joint else (lambda x, m: x)
    pre = lambda x: sae.encoder(x - sae.b_dec).cpu().numpy()
    inputs = {}
    lightgcn = hasattr(model, "base_u")
    if lightgcn:
        if hasattr(model, "_base"):
            base_u, base_i = model._base()
        else:
            model.forward(train_mode=False)
            base_u, base_i = model.base_u, model.base_i
        if kind in ("user", "joint"):
            inputs["user"] = pre(stdz(base_u[1:], "u"))
        if kind in ("item", "joint"):
            inputs["item"] = pre(stdz(base_i[1:], "i"))
    else:
        if kind in ("user", "joint"):
            zs, n = [], 0
            for batch in train_data:
                inter = (batch[0] if isinstance(batch, tuple) else batch).to(dev)
                x = SASRec.forward(model, inter[model.ITEM_SEQ], inter[model.ITEM_SEQ_LEN])
                zs.append(pre(stdz(x, "u")))
                n += len(x)
                if n >= a.max_samples:
                    break
            inputs["user"] = np.concatenate(zs)[: a.max_samples]
        if kind in ("item", "joint"):
            inputs["item"] = pre(stdz(model.item_embedding.weight[1:], "i"))
    for side, z in inputs.items():
        sk, ku = moments(z.astype(np.float64))
        act = active(z, sae.k)
        ok_s, ok_k = np.abs(sk) < 0.5, np.abs(ku) < 1      # NaN compares False: dead neurons do not pass
        rows.append(dict(dataset=config["dataset"], model=config["model"], sae=kind, side=side, checkpoint=Path(ckpt).name,
                         n_samples=len(z), n_neurons=z.shape[1], n_active=int(act.sum()),
                         pct_skew_all=100 * ok_s.mean(), pct_kurt_all=100 * ok_k.mean(),
                         pct_skew_active=100 * ok_s[act].mean(), pct_kurt_active=100 * ok_k[act].mean()))
        print(rows[-1], flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
pd.DataFrame(rows).to_csv(a.out, index=False)
print(f"{len(rows)} rows -> {a.out}")
