# PopSteer: Interpreting and Controlling Popularity Signals in Recommender Systems via Sparse Autoencoders

PopSteer is a post hoc framework for interpreting and mitigating popularity bias in embedding-based recommenders.
It leaves the pretrained recommender untouched and works in three steps:

1. **Interpret.** A TopK sparse autoencoder (SAE) is trained to reconstruct the embeddings of a frozen backbone
   (SASRec or LightGCN).
2. **Label.** Each SAE neuron *j* is scored by Cohen's *d<sub>j</sub>* between two reference groups with opposite
   popularity preferences. Neurons with *d<sub>j</sub> ≥ β* are *popularity-aligned*, those with
   *d<sub>j</sub> ≤ −β* are *unpopularity-aligned*.
3. **Steer.** At inference, popularity-aligned neurons are shifted down and unpopularity-aligned neurons up by
   *w<sub>j</sub>·σ<sub>j</sub>*, where *σ<sub>j</sub>* is the neuron's standard deviation and *w<sub>j</sub>* grows
   with |*d<sub>j</sub>*| up to the strength *α<sup>Pop</sup>* or *α<sup>Unpop</sup>*. Steering is
   *residual-preserving*: only the decoded change, *x̂*(steered) − *x̂*(unsteered), is added to the original
   embedding *x*, so the backbone is recovered exactly when nothing is steered.

The code builds on a fork of [RecBole](https://github.com/RUCAIBox/RecBole) 1.2.1 (`recbole/`), in which the SAE
models, the popularity metrics and the re-ranking baselines are implemented.

### Variants

A variant is named X-Y. X is the embedding the SAE is trained on and steers (U = users, I = items, UI = both, with
one joint SAE). Y is the labeling source (S = synthetic users, R = real users or items).

| Variant | SAE (checkpoint class) | Reference groups for Cohen's *d* | Backbones |
|---|---|---|---|
| **U-S** (original PopSteer, WWW '26) | user SAE (`SASRec_SAE`) | synthetic users whose histories hold only popular or only unpopular items | SASRec |
| **U-R** | user SAE (`SASRec_SAE`, `LightGCN_SAE`) | real users: top vs. bottom 10% by the mean head/tail label of their history | SASRec, LightGCN |
| **I-R** | item SAE (`SASRec_ISAE`, `LightGCN_ISAE`) | real items: head (top 10%) vs. tail (bottom 10%) by training popularity | SASRec, LightGCN |
| **UI-S** | joint user+item SAE (`SASRec_JSAE`) | synthetic users (as U-S) | SASRec |
| **UI-R** | joint user+item SAE (`SASRec_JSAE`, `LightGCN_JSAE`) | real users (as U-R) | SASRec, LightGCN |

* U-S and UI-S need synthetic profiles, which only a sequential backbone (SASRec) can embed.
* The joint SAE standardizes each modality with its own mean and scale before encoding. It can steer users, items,
  or both (`--sides`).
* Item-side steering has three rules: `original`, `align` (the sign of each neuron's edit follows its first-order
  effect on the mean user's scores, `--align`) and `gate` (only neurons already active for the item are steered,
  `--gate`).
* Paper setting: residual-preserving steering everywhere. I-R uses `align`, and UI-S and UI-R steer `both` sides
  without an item-side rule.

**Baselines** (re-ranking or score reweighting on the backbone): IPR, FA\*IR (`fair`), P-MMF with the
min-regularizer heuristic (`min_reg`), PCT, DUOR and Random.
**Metrics** (all at cutoff 10, on validation and test): nDCG, Recall, Hit, Gini Index, Item Coverage (N), Average
Recommendation Popularity (ARP, `averagepopularity`), User Popularity Deviation (UPD), and nDCG on head / mid / tail
items.

## Repository layout

```
run.py                     train backbones and SAEs (RecBole entry point)
configs/                   backbone and SAE configurations used in the paper
scripts/steer_eval.py      label neurons + evaluate a steering grid (variants) or a baseline grid; valid + test
scripts/select_budget.py   budget-based model selection over the steer_eval.py outputs
scripts/interpret.py       interpretability analyses (head-share box plots, top-K' reduction, UMAP + Wasserstein)
scripts/normality.py       skewness / kurtosis diagnostics of SAE activations
data/prepare_datasets.py   k-core filtering of the raw atomic files (no data is shipped)
recbole/                   RecBole fork with the PopSteer models:
                           model/sequential_recommender/sasrec_{sae,isae,jsae}.py
                           model/general_recommender/lightgcn_{sae,isae,jsae}.py
```

## Installation

The lock file targets **Linux x86_64 with CUDA 12.8** (Python 3.11, PyTorch 2.11.0 from the cu128 wheel index):

```bash
uv sync --locked          # creates .venv; then prefix commands with `uv run` or activate .venv
```

On other platforms, use pip with Python 3.11. Install a PyTorch build for your hardware first
(see [pytorch.org](https://pytorch.org)), then:

```bash
pip install -r requirements.txt
```

The package is not installed. Run all commands from the repository root, where `run.py` and the scripts find the
bundled `recbole/`.

## Data preparation

RecBole reads `./dataset/<name>/<name>.inter` relative to the working directory, and the scripts write neuron labels
to `./dataset/<name>/`. All commands below therefore run from the repository root with a `dataset/` folder there
(it is git-ignored). No data is distributed with this repository.

| Dataset (`--dataset`) | Source | Filtering in the paper |
|---|---|---|
| `ml-1mm` | [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) | 5-core version shipped with the original PopSteer code (`dataset/ml-1mm/ml-1mm.inter` in [PopSteer2.0](https://github.com/parepic/PopSteer2.0)): 6,040 users, 3,416 items, 999,611 interactions |
| `BeerAdvocate` | [BeerAdvocate reviews](https://cseweb.ucsd.edu/~jmcauley/datasets.html#multi_aspect) (McAuley et al.) | 10-core, duplicates kept: 10,463 / 13,906 / 1,395,865 |
| `yelp2018` | [Yelp Open Dataset](https://www.yelp.com/dataset) | 20-core, duplicates kept: 20,798 / 16,252 / 983,530 |
| `amazon-musical` | [Amazon Review Data 2018](https://nijianmo.github.io/amazon/) (Ni et al., EMNLP 2019), *Musical Instruments* 5-core reviews | duplicates removed, 5-core: 24,780 / 9,930 / 206,241 |
| `amazon-game10` | Amazon Review Data 2018, *Video Games* 5-core reviews | duplicates removed, 10-core: 5,942 / 3,793 / 103,778 |

Counts are users / items / interactions after filtering. "Duplicates removed" keeps the earliest (user, item) pair by
timestamp. The k-core filter is iterative.

1. Obtain the raw data as RecBole atomic files, e.g. from
   [RecSysDatasets](https://github.com/RUCAIBox/RecSysDatasets) or with its conversion scripts. Place them as
   `raw/BeerAdvocate/BeerAdvocate.inter`, `raw/yelp2018/yelp2018.inter`, `raw/amazon-musical/amazon-musical.inter`
   and `raw/amazon-game/amazon-game.inter` (Amazon: columns `user_id, item_id, rating, timestamp` from the review
   file in its original order). `data/prepare_datasets.py` also reads an Amazon 2018 `.json` review file directly
   if you point `SPECS` to it.
2. Build the filtered files:
   ```bash
   python data/prepare_datasets.py            # -> dataset/<name>/<name>.inter and dataset/summary.json
   ```
   `summary.json` reports `matches_paper` (the user / item / interaction counts above) and the md5 of every file.
   Row order is preserved because RecBole breaks timestamp ties by file order, which decides the leave-one-out split.
3. Copy `ml-1mm.inter` to `dataset/ml-1mm/ml-1mm.inter`.

When a dataset is first loaded, the code writes `item_popularity_labels.csv` (head / tail = top / bottom 10% of
items by training interactions) and `user_popularity_labels.csv` to `dataset/<name>/`. Metrics and labels use
these files.

## Pipeline

The examples use SASRec on ML-1M. For LightGCN, replace `SASRec` with `LightGCN` and `*_sasrec_d64.yaml` with
`*_lightgcn_d128.yaml`. Checkpoints are written to `saved/<Model>-<timestamp>.pth`.

### 1. Train the backbone

```bash
python run.py --model=SASRec   --dataset=ml-1mm --config_files=configs/bb_sasrec_d64.yaml   --train
python run.py --model=LightGCN --dataset=ml-1mm --config_files=configs/bb_lightgcn_d128.yaml --train
```

### 2. Train the SAEs on the frozen backbone

`--base_path` is stored in the SAE checkpoint and reloaded at evaluation time, so pass an **absolute** path. Any
config value can be overridden as `--key=value`.

```bash
BB=$PWD/saved/SASRec-<timestamp>.pth
python run.py --model=SASRec_SAE  --dataset=ml-1mm --config_files=configs/sae_sasrec_d64.yaml  --train --base_path=$BB --sae_scale_size=32 --sae_k=44
python run.py --model=SASRec_ISAE --dataset=ml-1mm --config_files=configs/isae_sasrec_d64.yaml --train --base_path=$BB --sae_scale_size=4  --sae_k=8
python run.py --model=SASRec_JSAE --dataset=ml-1mm --config_files=configs/jsae_sasrec_d64.yaml --train --base_path=$BB --sae_scale_size=32 --sae_k=44
```

The SAE has *N = s·d* neurons (`sae_scale_size` = *s*) and keeps the `sae_k` = *K* largest activations. Training
stops early on the validation reconstruction loss. For each backbone, dataset and SAE type, the paper keeps the SAE
with the highest validation nDCG@10 under reconstruction. `scripts/steer_eval.py --method none` on an SAE
checkpoint reports this (plus the cosine between original and reconstructed embeddings).

### 3. Steering and baseline evaluation

`scripts/steer_eval.py` writes one row per (split, setting) with every metric on validation **and** test. For the
steering variants, it first labels the neurons and writes them to `dataset/<name>/user/` or `dataset/<name>/item/`.
Variants that share a label folder (U-S, U-R, UI-S, UI-R all use `user/`) must therefore run one after the other,
not in parallel, or from separate working folders with their own `dataset/` copy.

```bash
OUT=results/SASRec_ml-1mm; mkdir -p $OUT
U=saved/SASRec_SAE-<timestamp>.pth; I=saved/SASRec_ISAE-<timestamp>.pth; J=saved/SASRec_JSAE-<timestamp>.pth

# backbone reference (needed for the budgets)
python scripts/steer_eval.py --path $BB --method none --out $OUT/tune_none.csv

# PopSteer variants (paper setting; U-S / UI-S run the synthetic-user analysis, 200 x 2048 users per group)
python scripts/steer_eval.py --path $U --variant U-S                --out $OUT/tune_U-S.csv
python scripts/steer_eval.py --path $U --variant U-R                --out $OUT/tune_U-R.csv
python scripts/steer_eval.py --path $I --variant I-R  --align       --out $OUT/tune_I-R.csv
python scripts/steer_eval.py --path $J --variant UI-S --sides both  --out $OUT/tune_UI-S.csv
python scripts/steer_eval.py --path $J --variant UI-R --sides both  --out $OUT/tune_UI-R.csv

# baselines on the backbone checkpoint
for m in ipr fair pct min_reg duor random; do
  python scripts/steer_eval.py --path $BB --method $m --out $OUT/tune_$m.csv
done
```

Useful options: `--grid {auto,www,ext,full,low,mid}` (default `auto`: `full` for steering on SASRec, `low` on
LightGCN, `www` for baselines), `--no-residual` (steer the reconstruction instead), `--gate` / `--align`,
`--sides user,item,both`, `--resume` (continue from `<out>.partial.csv` after an interruption), and `--settings
a_pop:a_unpop:beta[:side],...` with `--ablation {none,noise,randsel} --seeds 0,1,2,3,4` for the Noise /
RandomSelect ablation. `--analyze_batches 0` reuses existing synthetic-user statistics. Columns: `param1, param2,
param3` = *α<sup>Pop</sup>, α<sup>Unpop</sup>, β* (steering; `param3 = -1` is the unsteered reference row) or the
baseline's parameters. `rule` records the steering rule, e.g. `align+res`.

### 4. Budget-based model selection

For an nDCG budget *b*, each method keeps the configurations whose validation nDCG@10 is at least *(1 − b)* times
the backbone's validation nDCG@10. Among them, it takes the one with the lowest validation Gini Index and reports
that configuration's test metrics:

```bash
python scripts/select_budget.py --backbone $OUT/tune_none.csv "$OUT/tune_*.csv" \
    --budgets 0.01,0.02,0.03,0.05,0.07,0.10,0.12,0.15 --out $OUT/selection.csv
```

The output also contains "PopSteer (best variant)", which at each budget is the variant whose selection has the
lowest validation Gini Index. `--by_side` selects separately per steered side of a joint SAE.

### Optional analyses

```bash
# interpretability (SASRec SAEs), from the folder in which the variant's labels were written
python scripts/interpret.py --path $U --variant U-S --alpha_pop 1 --alpha_unpop 1 --beta 1
# normality of SAE activations
python scripts/normality.py --path $U $I $J --out $OUT/normality.csv
# quick single-setting check on the validation split
python run.py --test --path=$U --steer --a_pop=1 --a_unpop=1 --D=0
```

## Hyperparameters used in the paper

| | SASRec | LightGCN |
|---|---|---|
| Backbone | *d* = 64, 2 layers, 2 heads, dropout 0.5, max. sequence length 50, full-softmax CE, lr 1e-3, leave-one-out split | *d* = 128, 2 layers, reg. 1e-5, BPR, lr 1e-3, random 80/10/10 split per user |
| Training | batch 2048, early stopping on validation nDCG@10 (patience 10), seed 2020 | same |
| User SAE | *s* ∈ {32, 64, 96}, *K* ∈ {36, 40, …, 56} | *s* ∈ {32, 48}, *K* ∈ {16, 32} |
| Item SAE | *s* ∈ {2, 4, 8, 16} with *K* ∈ {4, 8, 16}; *s* ∈ {32, 64} with *K* ∈ {16, 32, 64} | *s* ∈ {8, 16, 32}, *K* ∈ {16, 32} |
| Joint SAE | *s* ∈ {32, 64}, *K* ∈ {36, 44, 52} | *s* ∈ {16, 32, 48}, *K* ∈ {16, 32} |
| SAE training | Adam, lr 0.01 with ReduceLROnPlateau (patience 10), batch 2048, early stopping on the validation reconstruction loss | same |
| Steering (`--grid`) | `full`: *α<sup>Pop</sup>, α<sup>Unpop</sup>* ∈ {0.1, 0.25, 0.5, 0.75, 1.0, 1.5, …, 5.0}, *β* ∈ {0, 0.5, …, 3.0} | `low`: *α* ∈ {0.1, 0.25, 0.5, 0.75, 1.0}, *β* ∈ {0, 0.5, …, 3.0} |
| Synthetic users | 200 batches × 2048 users per group (fewer if the training data has fewer batches), profiles of *M* = 50 items | n/a |

Baseline grids (`www`): IPR *α* and P-MMF *λ* ∈ {0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0}; FA\*IR and PCT
*p* ∈ {0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0} × *α* ∈ {0.01, 0.05, 0.1}; DUOR list size *t* ∈ {50, 100, 250, 500,
1000}; Random top-*t* ∈ {15, 30, 50, 75, 100}. Smaller subsets of the SAE grids were used on the smallest datasets.

## Extending PopSteer

**New dataset.** Put a RecBole atomic file at `dataset/<name>/<name>.inter`, with columns `user_id:token`,
`item_id:token`, `rating:float` and `timestamp:float`. Then run the pipeline with `--dataset=<name>`.
Dataset-specific RecBole settings can go into `recbole/properties/dataset/<name>.yaml`. The configs in `configs/`
set the split and the maximum sequence length. The popularity labels are created automatically at the first load.

**New backbone.** Add `<Backbone>_SAE` (and, if needed, `_ISAE` / `_JSAE`) in a file named after the class in
lower case under `recbole/model/<type>_recommender/`. Import it in that package's `__init__.py`; RecBole finds a
model by its file name. Use `sasrec_sae.py` (sequential) or `lightgcn_sae.py` (graph-based) as templates. The
wrapper must:

* subclass the backbone, load the frozen weights from `config["base_path"]`
  (`torch.load(...)["state_dict"]`), and freeze all backbone parameters;
* create the SAE as `self.sae_module_u = SAE(config, side="user")` and/or `self.sae_module_i = SAE(config,
  side="item")`, using the `SAE` class from `sasrec_sae.py` or `lightgcn_sae.py`. It reads `input_dim`,
  `sae_scale_size`, `sae_k`, `alpha_pop`, `alpha_unpop`, `D` and `steer` from the config. Only the SAE parameters
  are trainable;
* in `calculate_loss`, pass the backbone embeddings through the SAE with `train_mode=True` and return
  `sae.fvu + sae.auxk_loss / 2`;
* in `predict` / `full_sort_predict`, replace the user embedding (or the item scoring table) by
  `sae(x, train_mode=False)`. This returns the reconstruction, the steered embedding when `sae.steer` is on, or
  *x* + the steering change when `sae.residual` is set. Also accumulate `self.val_fvu_u += sae.fvu`
  (`val_fvu_i` for items; both tensors must exist), which the evaluator reports as `SAE_Loss_u` / `_i` / `_total`;
* add the class name to the two SAE tuples in `recbole/trainer/trainer.py`: the per-epoch dead-latent tracking in
  `_train_epoch`, and the SAE learning-rate scheduler (ReduceLROnPlateau) in `fit`;
* for U-S / UI-S (synthetic users), support `full_sort_predict(interaction, popular=True/False)` and
  `full_sort_predict(interaction, save=True)` as in `SASRec_SAE`. This is used by `trainer.analyze_neurons`;
* for real-user labels on a non-sequential backbone, expose the user embeddings as `self.base_u` (see
  `LightGCN_SAE`). Sequential backbones only need `forward(item_seq, item_seq_len)`, which sets
  `sae.last_activations`;
* for the baselines, the backbone itself must implement the re-ranking flags (`fair`, `ipr`, `pct`, `min_reg`,
  `duor`, `random`, with `param1` / `param2`) as in `sasrec.py` and `lightgcn.py`.

**New labeling source.** A labeling source only has to produce SAE pre-activations for all entities of the steered
type and for two reference groups. Pass them to `write_stats(side, z_all, z_pop, z_unpop)` in
`scripts/steer_eval.py` (see `item_labels()` and `real_user_labels()`). This writes `cohens_d.csv` and
`neuron_stats{,_pop,_unpop}.csv` to `dataset/<name>/<side>/`. The SAE builds its steering vector from these files
(`SAE._build_steering_vector`): *d<sub>j</sub>* from `cohens_d.csv` and *σ<sub>j</sub>* from the `sd` column of
`neuron_stats.csv`.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{ahmadov2026popsteer,
  title     = {From Insight to Intervention: Interpretable Neuron Steering for Controlling Popularity Bias in
               Recommender Systems},
  author    = {Ahmadov, Parviz and Mansoury, Masoud},
  booktitle = {Proceedings of the ACM Web Conference 2026},
  series    = {WWW '26},
  year      = {2026},
  doi       = {10.1145/3774904.3793006}
}
```

Journal version: citation to be added.

## Acknowledgements and license

This repository is built on [RecBole](https://github.com/RUCAIBox/RecBole) (MIT License) and extends the original
PopSteer implementation, [PopSteer2.0](https://github.com/parepic/PopSteer2.0). RecBole's example dataset
(`recbole/dataset_example/ml-100k`), tests, documentation assets and packaging scripts are not included. The code
is released under the MIT License (see `LICENSE`), which keeps RecBole's copyright notice.
