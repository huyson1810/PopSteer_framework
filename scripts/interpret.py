"""Interpretability analyses of the PopSteer neurons, for every variant (SASRec backbone).

Run from the working folder in which scripts/steer_eval.py labelled the neurons of this checkpoint, i.e. that holds
./dataset/<name>/ with the variant's user/ or item/ cohens_d.csv and neuron_stats.csv:
  python scripts/interpret.py --path saved/<sae>.pth --variant {U-S,U-R,I-R,UI-S,UI-R} [--side {user,item,both}]
         [--rule {original,gate,align}] --alpha_pop A --alpha_unpop B --beta C [--kprime 0,8,16,...]
Steering is residual-preserving throughout. Outputs (in the working folder):
  interp_box.csv    (a) head-share box plot: for every user (item) the head share H (item popularity percentile),
                        and the mean of the top-10 users (items) of the most popularity-aligned (max d_j) and most
                        tail-aligned (min d_j) active neuron. Users: validation-input states; items: the scoring table.
  interp_kprime.csv (b) Gini@10 / nDCG@10 (test) when the activations of the top-K' neurons with d_j > 1 (or < -1)
                        are reduced by one standard deviation, z_j - sigma_j; K' in --kprime.
  interp_umap.csv   (c) 2-D UMAP (cosine, 15 neighbours, min_dist 0.1, seed 42) of SAE pre-activations for four
                        groups of --n_umap: real, reference-popular, reference-unpopular, steered.
                        Reference groups: synthetic popular / unpopular users (U-S, UI-S), the top / bottom 10% real
                        users by mean item label (U-R, UI-R), head / tail items (I-R, or a joint SAE on the item side).
                        Steered = the given setting (--alpha_pop, --alpha_unpop, --beta, --rule).
  interp_wass.csv       Wasserstein distances between the groups in the UMAP plane (mean of the 1-D distances along
                        the two axes): real / steered -> reference popular / unpopular.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import wasserstein_distance

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # make the bundled recbole/ importable

from recbole.model.sequential_recommender.sasrec import SASRec  # noqa: E402
from recbole.quick_start import load_data_and_model  # noqa: E402
from recbole.utils import get_trainer, make_items_popular, make_items_unpopular  # noqa: E402

VARIANTS = {"U-S": "ps", "U-R": "ur", "I-R": "ir", "UI-S": "uis", "UI-R": "uir"}
RULES = {"original": "res", "gate": "gate+res", "align": "align+res"}

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--path", required=True)
ap.add_argument("--variant", required=True, choices=list(VARIANTS))
ap.add_argument("--side", default="user", choices=["user", "item", "both"], help="steered side of a joint SAE")
ap.add_argument("--rule", default="original", choices=list(RULES), help="item-side steering rule")
ap.add_argument("--alpha_pop", type=float, required=True)
ap.add_argument("--alpha_unpop", type=float, required=True)
ap.add_argument("--beta", type=float, required=True)
ap.add_argument("--kprime", default="0,8,16,24,32,40,48,56,64,72,80,88,96")
ap.add_argument("--n_umap", type=int, default=5000, help="points per UMAP group")
ap.add_argument("--umap_only", action="store_true", help="skip (a) and (b); only UMAP + Wasserstein")
ap.add_argument("--skip_umap", action="store_true", help="only (a) and (b)")
a = ap.parse_args()
a.method = VARIANTS[a.variant]
a.rule = RULES[a.rule]
rng = np.random.default_rng(42)
torch.set_grad_enabled(False)   # inference only

overrides = {"metrics": ["NDCG", "GiniIndex"], "topk": [10], "valid_metric": "NDCG@10", "steer": True,
             "alpha_pop": 1.0, "alpha_unpop": 1.0, "D": 0.0}
config, model, dataset, train_data, valid_data, test_data = load_data_and_model(a.path, dict=overrides)
assert not str(config["model"]).startswith("LightGCN"), "interpret.py supports SASRec-based SAEs only"
trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
trainer.eval_collector.data_collect(train_data)
model.eval()
ds = config["dataset"]
base = f"./dataset/{ds}"
sae = model.sae_module_i if a.method == "ir" else model.sae_module_u
sae.residual = True
label_side = "item" if a.method == "ir" else "user"
d = pd.read_csv(f"{base}/{label_side}/cohens_d.csv")["cohens_d"].to_numpy()
sd = pd.read_csv(f"{base}/{label_side}/neuron_stats.csv")["sd"].to_numpy()
joint = a.method in ("uis", "uir")
dev = sae.W_dec.device


# ------------------------------------------------------------------ embeddings and pre-activations
@torch.no_grad()
def user_states(seqs=None):
    """Raw SASRec states of the validation users (or of given synthetic sequences) and their user ids."""
    xs, ids = [], []
    if seqs is None:
        for batch in valid_data:
            inter = batch[0].to(dev)
            xs.append(SASRec.forward(model, inter[model.ITEM_SEQ], inter[model.ITEM_SEQ_LEN]))
            ids.append(inter[model.USER_ID])
        return torch.cat(xs), torch.cat(ids).cpu().numpy()
    for i in range(0, len(seqs), 1024):
        s = seqs[i:i + 1024].to(dev)
        xs.append(SASRec.forward(model, s, torch.full((len(s),), s.shape[1], device=dev)))
    return torch.cat(xs), None


def std(x, m):
    return (x - getattr(model, f"mu_{m}")) / getattr(model, f"sd_{m}") if joint else x


@torch.no_grad()
def pre(x):
    return sae.encoder(x - sae.b_dec)


item_table = model.item_embedding.weight[1:].detach()
pop = np.bincount(train_data.dataset.inter_feat[model.ITEM_ID].numpy(), minlength=item_table.shape[0] + 1)[1:]
lab = pd.read_csv(f"{base}/item_popularity_labels.csv", usecols=["item_id:token", "popularity_label"])
ilab = dict(zip(lab["item_id:token"].astype(int), lab["popularity_label"].astype(int)))
y_item = np.array([ilab.get(i, 0) for i in range(1, item_table.shape[0] + 1)])

# head share H_u of each user's training history
uid_tr = train_data.dataset.inter_feat[model.USER_ID].numpy()
iid_tr = train_data.dataset.inter_feat[model.ITEM_ID].numpy()
head = np.isin(iid_tr, np.where(y_item == 1)[0] + 1)
H = pd.Series(head.astype(float)).groupby(uid_tr).mean()

x_val, uid_val = user_states()
z_users = pre(std(x_val, "u")).cpu().numpy()
z_items = pre(std(item_table, "i")).cpu().numpy() if (joint or a.method == "ir") else None


def active(z):
    """Neurons in the TopK support (positive) of at least one input. Dead neurons are dropped, since they carry
    many of the largest |d| without ever being selected by TopK."""
    zt = torch.tensor(z)
    idx = torch.topk(torch.relu(zt), sae.k, dim=1).indices
    pos = torch.gather(zt, 1, idx) > 0
    return set(idx[pos].unique().tolist())


act_u = active(z_users) if a.method != "ir" else set()
act_i = active(z_items) if z_items is not None else set()


def argext(idx_set, sign):
    """Neuron with the largest (sign=+1) or smallest (sign=-1) d among the given active neurons."""
    c = [j for j in idx_set if not np.isnan(d[j])]
    return max(c, key=lambda j: sign * d[j])


# ------------------------------------------------------------------ (a) box plot data
if not a.umap_only:   # (a) and (b) do not depend on the steering setting
    rows = []
    if a.method != "ir":
        h_users = H.reindex(uid_val).to_numpy()
        for j, kind in ((argext(act_u, 1), "pop_neuron"), (argext(act_u, -1), "unpop_neuron")):
            top = np.argsort(-z_users[:, j])[:10]
            rows.append(dict(side="user", kind=kind, neuron=j, cohens_d=d[j], top10_mean=float(np.nanmean(h_users[top]))))
        rows += [dict(side="user", kind="all", value=float(v)) for v in h_users if not np.isnan(v)]
    if z_items is not None:
        pct = pd.Series(pop).rank(pct=True).to_numpy()            # item popularity percentile in the training data
        for j, kind in ((argext(act_i, 1), "pop_neuron"), (argext(act_i, -1), "unpop_neuron")):
            top = np.argsort(-z_items[:, j])[:10]
            rows.append(dict(side="item", kind=kind, neuron=j, cohens_d=d[j], top10_mean=float(pct[top].mean())))
        rows += [dict(side="item", kind="all", value=float(v)) for v in pct]
    pd.DataFrame(rows).to_csv("interp_box.csv", index=False)
    print("box done", flush=True)

    # ------------------------------------------------------------------ (b) K' neuron reduction
    # a'_j = a_j - sigma_j on the sparse activations after TopAct (SAE.post_shift), residual-preserving.
    # Joint SAE: the shift is applied only on the steered side (user / item / both).
    SHIFT = {"vec": None}
    if joint:
        _orig_u, _orig_i = model._user_rec, model._item_table

        def _user_rec(*args, **kw):
            sae.post_shift = SHIFT["vec"] if a.side in ("user", "both") else None
            try:
                return _orig_u(*args, **kw)
            finally:
                sae.post_shift = None

        def _item_table(*args, **kw):
            sae.post_shift = SHIFT["vec"] if a.side in ("item", "both") else None
            sae.post_shift_active = True     # item side: only where the neuron is active (a shared shift ranks nothing)
            try:
                return _orig_i(*args, **kw)
            finally:
                sae.post_shift, sae.post_shift_active = None, False

        model._user_rec, model._item_table = _user_rec, _item_table

    def evaluate_with(vec):
        for cache in ("restore_user_e", "restore_item_e"):
            if hasattr(model, cache):
                setattr(model, cache, None)
        sae.steer, sae.gate, sae.use_align = False, False, False
        if joint:
            model.item_gate, model.item_align = False, False
        use = vec if (vec is not None and torch.any(vec != 0)) else None
        SHIFT["vec"] = use
        sae.post_shift = None if joint else use
        sae.post_shift_active = a.method == "ir"   # item SAE: only where a_j > 0
        r = trainer.evaluate(test_data, load_best_model=False, model_file=a.path, show_progress=False)
        sae.post_shift, SHIFT["vec"] = None, None
        return float(r["giniindex@10"]), float(r["ndcg@10"])

    side_act = act_i if (a.method == "ir" or a.side == "item") else (act_u | act_i if a.side == "both" else act_u)
    order_pos = [j for j in np.argsort(-d) if d[j] > 1 and j in side_act]
    order_neg = [j for j in np.argsort(d) if d[j] < -1 and j in side_act]
    krows = []
    for kp in [int(k) for k in a.kprime.split(",")]:
        for kind, order in (("pop", order_pos), ("unpop", order_neg)):
            vec = torch.zeros(sae.hidden_dim, device=dev)
            sel = order[:kp]
            if sel:
                vec[sel] = -torch.tensor(sd[sel], dtype=vec.dtype, device=dev)
            g, n = evaluate_with(vec)
            krows.append(dict(kprime=kp, neurons=kind, n_available=len(order), gini=g, ndcg=n))
            print(krows[-1], flush=True)
    pd.DataFrame(krows).to_csv("interp_kprime.csv", index=False)

# ------------------------------------------------------------------ (c) UMAP + Wasserstein
if a.skip_umap:
    print("interp done (no umap)", os.getcwd())
    raise SystemExit(0)
sae.alpha_pop, sae.alpha_unpop, sae.d_min, sae.N = a.alpha_pop, a.alpha_unpop, a.beta, sae.hidden_dim
sae._steer_ready = False
sae._build_steering_vector(dataset=ds)
delta = sae.steer_vec.detach().clone()


def sample(m, n=a.n_umap):
    idx = rng.choice(len(m), size=min(n, len(m)), replace=False)
    return m[idx]


item_mode = a.method == "ir" or (joint and a.side == "item")
if item_mode:
    if a.rule == "align+res":
        p_bar = x_val.mean(0)
        delta = delta * torch.sign(sae.W_dec @ p_bar)
    zi = torch.tensor(z_items, device=dev)
    shift = delta.unsqueeze(0).expand_as(zi).clone()
    if a.rule == "gate+res":
        idx = torch.topk(torch.relu(zi), sae.k, dim=1).indices
        shift = shift * torch.zeros_like(zi).scatter_(1, idx, 1.0) * (zi > 0)
    steered = (zi + shift).cpu().numpy()
    groups = {"Actual": sample(z_items), "Reference popular": sample(z_items[y_item == 1]),
              "Reference unpopular": sample(z_items[y_item == -1]), "Steered": sample(steered)}
else:
    if a.method in ("ps", "uis"):
        n = a.n_umap
        zp = pre(std(user_states(make_items_popular(n, ds, model.max_seq_length))[0], "u")).cpu().numpy()
        zu = pre(std(user_states(make_items_unpopular(n, ds, model.max_seq_length))[0], "u")).cpu().numpy()
    else:
        ul = pd.read_csv(f"{base}/user_popularity_labels.csv", usecols=["user_id:token", "popularity_label"])
        grp = ul.groupby("user_id:token")["popularity_label"].first().reindex(uid_val).fillna(0).to_numpy()
        zp, zu = z_users[grp == 1], z_users[grp == -1]
    steered = z_users + delta.cpu().numpy()[None, :]
    groups = {"Actual": sample(z_users), "Reference popular": sample(zp), "Reference unpopular": sample(zu),
              "Steered": sample(steered)}

import umap  # noqa: E402  (slow import, only needed here)

X = np.concatenate(list(groups.values()))
emb = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42).fit_transform(X)
names = np.concatenate([[k] * len(v) for k, v in groups.items()])
pd.DataFrame({"approach": names, "var1": emb[:, 0], "var2": emb[:, 1]}).to_csv("interp_umap.csv", index=False)


def w2(g1, g2):
    e1, e2 = emb[names == g1], emb[names == g2]
    return float(np.mean([wasserstein_distance(e1[:, k], e2[:, k]) for k in range(2)]))


wrows = [dict(pair=f"{s} -> {r}", distance=w2(s, r)) for r in ("Reference popular", "Reference unpopular")
         for s in ("Actual", "Steered")]
pd.DataFrame(wrows).to_csv("interp_wass.csv", index=False)
print(wrows, flush=True)
print("interp done", os.getcwd())
