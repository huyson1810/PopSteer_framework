"""Build the filtered atomic datasets used in the paper from RecBole atomic files. Raw files are never modified.

  python data/prepare_datasets.py [--raw raw] [--out dataset] [name ...]

Input  <raw>/<source folder>/<file> (see SPECS): a RecBole atomic .inter file (tab-separated, typed header such as
       user_id:token), or an Amazon 2018 review file (.json, one JSON object per line: reviewerID, asin, overall,
       unixReviewTime).
Output <out>/<name>/<name>.inter, ready for RecBole (run the pipeline from the folder that contains <out>/), plus
       <out>/summary.json with the counts and md5 of every file.

Filtering per dataset (SPECS):
  - duplicates KEPT or REMOVED: removing keeps the earliest (user, item) pair by timestamp, ties by file order;
  - then iterative k-core: users and items with fewer than k interactions are dropped until none is left.
Rows keep their original file order, because RecBole assigns internal ids by first appearance and breaks timestamp
ties by file order. Only the columns the pipeline loads are written: user_id, item_id, rating, timestamp
(BeerAdvocate has no `rating`; its `overall` score is written as rating). If an <name>.item file sits next to the
raw .inter, the rows of the surviving items are copied (not loaded by the pipeline).
ML-1M is not built here: the pipeline uses ml-1mm.inter as distributed with the original PopSteer code.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

SPECS = {  # name: (raw file under --raw, rating column, k, remove duplicates, expected users/items/interactions)
    "BeerAdvocate": ("BeerAdvocate/BeerAdvocate.inter", "overall", 10, False, (10463, 13906, 1395865)),
    "yelp2018": ("yelp2018/yelp2018.inter", "rating", 20, False, (20798, 16252, 983530)),
    "amazon-musical": ("amazon-musical/amazon-musical.inter", "rating", 5, True, (24780, 9930, 206241)),
    "amazon-game10": ("amazon-game/amazon-game.inter", "rating", 10, True, (5942, 3793, 103778)),
}


def kcore(df, k):
    while True:
        n = len(df)
        df = df[(df["user_id"].map(df["user_id"].value_counts()) >= k)
                & (df["item_id"].map(df["item_id"].value_counts()) >= k)]
        if len(df) == n:
            return df


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path, rcol):
    if path.suffix == ".json":   # Amazon 2018 review dump: one JSON object per line
        df = pd.read_json(path, lines=True, dtype={"reviewerID": str, "asin": str})
        return df.rename(columns={"reviewerID": "user_id", "asin": "item_id", "overall": "rating",
                                  "unixReviewTime": "timestamp"})[["user_id", "item_id", "rating", "timestamp"]]
    head = pd.read_csv(path, sep="\t", nrows=0).columns
    col = {c.split(":")[0]: c for c in head}
    use = [col["user_id"], col["item_id"], col["timestamp"]] + ([col[rcol]] if rcol else [])
    df = pd.read_csv(path, sep="\t", usecols=use, dtype={col["user_id"]: str, col["item_id"]: str},
                     keep_default_na=False)
    df = df.rename(columns={col["user_id"]: "user_id", col["item_id"]: "item_id", col["timestamp"]: "timestamp",
                            **({col[rcol]: "rating"} if rcol else {})})
    if not rcol:
        df["rating"] = 1.0   # ratings are not used by the models
    return df[["user_id", "item_id", "rating", "timestamp"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="raw", help="folder with the raw atomic files")
    ap.add_argument("--out", default="dataset", help="output folder (RecBole's data_path)")
    ap.add_argument("names", nargs="*", help=f"datasets to build (default: all of {list(SPECS)})")
    a = ap.parse_args()
    raw, out_dir = Path(a.raw), Path(a.out)
    summary_path = out_dir / "summary.json"
    summary = json.load(open(summary_path)) if summary_path.exists() else {}
    for name in a.names or list(SPECS):
        rel, rcol, k, dedup, expected = SPECS[name]
        df = load(raw / rel, rcol)
        df["_row"] = range(len(df))
        if dedup:
            df = (df.sort_values(["timestamp", "_row"], kind="stable")
                    .drop_duplicates(["user_id", "item_id"], keep="first").sort_values("_row"))
        out = kcore(df, k).sort_values("_row").drop(columns="_row")
        path = out_dir / name / f"{name}.inter"
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(path, sep="\t", index=False,
                   header=["user_id:token", "item_id:token", "rating:float", "timestamp:float"])
        stats = dict(users=int(out.user_id.nunique()), items=int(out.item_id.nunique()), interactions=len(out),
                     duplicate_pairs_left=int(out.duplicated(["user_id", "item_id"]).sum()), k=k,
                     duplicates_removed=dedup, md5=md5(path))
        stats["matches_paper"] = (stats["users"], stats["items"], stats["interactions"]) == expected
        item_file = (raw / rel).with_suffix(".item")
        if item_file.exists():  # item metadata of the surviving items (not loaded by the pipeline)
            meta = pd.read_csv(item_file, sep="\t", dtype=str, keep_default_na=False, quoting=3)
            meta = meta[meta.iloc[:, 0].isin(set(out.item_id))]
            meta.to_csv(path.with_suffix(".item"), sep="\t", index=False, quoting=3)
            stats["item_metadata_rows"] = len(meta)
        summary[name] = stats
        print(name, stats)
    json.dump(summary, open(summary_path, "w"), indent=1)


if __name__ == "__main__":
    main()
