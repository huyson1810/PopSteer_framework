"""User Popularity Deviation (UPD), Abdollahpouri et al. 2021, with PopSteer's 10% groups.

Item groups H/M/T = items labelled 1 / 0 / -1 in item_popularity_labels.csv (top 10%, middle, bottom 10% by
training count). User groups = popularity_label in user_popularity_labels.csv (top 10% / middle / bottom 10% of
users by their mean item label). For each user, JSD (base 2, in [0, 1]) between the H/M/T shares of the training
profile and of the top-k list; UPD = mean over user groups of the within-group mean JSD.

Pure numpy so it can be unit-tested without RecBole; the metric class in metrics.py is a thin wrapper.
"""
import numpy as np

LABELS = (1, 0, -1)  # column order of the share vectors: H, M, T


def hmt_shares(labels):
    """labels: int array (n, L) with values in {1, 0, -1} -> (n, 3) shares in H, M, T order."""
    labels = np.asarray(labels)
    return np.stack([(labels == v).mean(axis=1) for v in LABELS], axis=1)


def jsd(p, q):
    """Row-wise Jensen-Shannon divergence, base 2; 0 * log 0 = 0."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    m = (p + q) / 2

    def kl(a, b):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(a > 0, a * np.log2(a / b), 0.0)
        return t.sum(axis=1)

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def upd(profile_shares, list_shares, groups):
    """Returns (UPD, {group: mean JSD}, mean JSD over all users)."""
    d = jsd(profile_shares, list_shares)
    groups = np.asarray(groups)
    per_group = {g: float(d[groups == g].mean()) for g in np.unique(groups)}
    return float(np.mean(list(per_group.values()))), per_group, float(d.mean())


def upd_from_lists(item_matrix, user_ids, item_label, user_profile, user_group, k):
    """item_matrix: (n, >=k) recommended item ids; user_ids: (n,) ids of the users owning each row.
    item_label: dict item id -> {1, 0, -1}; user_profile: dict user id -> (3,) H/M/T profile shares;
    user_group: dict user id -> group label. Items without a label count as mid (0)."""
    lab = np.vectorize(lambda i: item_label.get(int(i), 0), otypes=[int])(np.asarray(item_matrix)[:, :k])
    q = hmt_shares(lab)
    p = np.stack([user_profile[int(u)] for u in user_ids])
    g = np.array([user_group[int(u)] for u in user_ids])
    return upd(p, q, g)
