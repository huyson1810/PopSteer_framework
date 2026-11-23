# @Time   : 2020/7/20
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE
# @Time   : 2022/7/8, 2020/10/3, 2020/10/1
# @Author : Zhen Tian, Yupeng Hou, Zihan Lin
# @Email  : chenyuwuxinn@gmail.com, houyupeng@ruc.edu.cn, zhlin@ruc.edu.cn

"""Entry point for training backbones and SAEs (RecBole run script, extended for PopSteer).

Run it from a working folder that holds ./dataset/<name>/<name>.inter; checkpoints go to ./saved/ (``checkpoint_dir``).
Any RecBole parameter can be overridden on the command line as ``--key=value`` (e.g. ``--sae_k=44``).

  Train a backbone:
    python run.py --model=SASRec --dataset=ml-1mm --config_files=configs/bb_sasrec_d64.yaml --train
  Train an SAE on a frozen backbone (SASRec_SAE / SASRec_ISAE / SASRec_JSAE / LightGCN_SAE / LightGCN_ISAE /
  LightGCN_JSAE):
    python run.py --model=SASRec_SAE --dataset=ml-1mm --config_files=configs/sae_sasrec_d64.yaml --train \
        --base_path=/abs/path/to/saved/SASRec-<date>.pth --sae_scale_size=32 --sae_k=44
  Quick evaluation of a checkpoint on the validation split (optionally steered):
    python run.py --test --path=saved/SASRec_SAE-<date>.pth [--steer --a_pop=1 --a_unpop=1 --D=0]
  Synthetic-user neuron analysis of a user SAE (writes ./dataset/<name>/user/):
    python run.py --analyze --path=saved/SASRec_SAE-<date>.pth
The full steering evaluation (all variants, baselines, valid + test) is scripts/steer_eval.py.
"""

import argparse

import torch

from recbole.quick_start import run, load_data_and_model
from recbole.utils import get_trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="BPR", help="name of models")
    parser.add_argument("--train", action="store_true", help="Whether to train model")
    parser.add_argument("--test", action="store_true", help="Whether to test model")
    parser.add_argument("--save_activations", action="store_true", help="Whether to save SAE activations")
    parser.add_argument("--epochs_save", type=int, default=None, help="Number of epochs for saving SAE activations")
    parser.add_argument("--analyze", action="store_true", help="Whether to analyze neurons")

    parser.add_argument(
        "--dataset", "-d", type=str, default="ml-100k", help="name of datasets"
    )

    parser.add_argument("--steer", action="store_true", help="Whether to steer PopSteer or not when testing")
    parser.add_argument("--a_pop", default=None, type=float, help="alpha_pop hyperparameter value")
    parser.add_argument("--a_unpop", default=None, type=float, help="alpha_unpop hyperparameter value")
    parser.add_argument("--D", default=None, type=float, help="Cohens d hyperparameter")

    parser.add_argument("--config_files", type=str, default=None, help="config files")
    parser.add_argument(
        "--nproc", type=int, default=1, help="the number of process in this group"
    )
    parser.add_argument(
        "--ip", type=str, default="localhost", help="the ip of master node"
    )
    parser.add_argument(
        "--port", type=str, default="5678", help="the port of master node"
    )
    parser.add_argument(
        "--world_size", type=int, default=-1, help="total number of jobs"
    )
    parser.add_argument(
        "--base_path", type=str, default='no path', help="base model path"
    )
    parser.add_argument(
        "--path", type=str, default='no path', help="model path"
    )

    parser.add_argument(
        "--group_offset",
        type=int,
        default=0,
        help="the global rank offset of this group",
    )

    args, _ = parser.parse_known_args()
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    config_file_list = (
        args.config_files.strip().split(" ") if args.config_files else None
    )
    config_dict = dict()

    if args.train == True:
        config_dict = {}
        # User SAEs: keep the SAE validation metrics and stop early on the validation reconstruction loss
        # (smaller = better) instead of nDCG@10.
        if args.model in ["LightGCN_SAE", "SASRec_SAE"]:
            config_dict["metrics"] = ["NDCG", "GiniIndex", "ItemCoverageN", "SAE_Loss_u", "SAE_Loss_total"]
            config_dict["valid_metric"] = "SAE_LOSS_u"
            config_dict["valid_metric_bigger"] = False
        if args.model in ["SASRec", "SASRec_SAE", "SASRec_ISAE", "SASRec_JSAE"]:
            config_dict["train_neg_sample_args"] = None
        run(
            args.model,
            args.dataset,
            config_file_list=config_file_list,
            config_dict=config_dict,
            nproc=args.nproc,
            ip=args.ip,
            port=args.port,
            group_offset=args.group_offset,
        )

    elif args.test == True:
        config_dict = {
            "alpha_pop": args.a_pop,
            "alpha_unpop": args.a_unpop,
            "D": args.D,
            "steer": args.steer
            }
        config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
            model_file=args.path, dict=config_dict
        )
        trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
        trainer.eval_collector.data_collect(train_data)

        test_result = trainer.evaluate(
            valid_data, model_file=args.path, load_best_model=False, show_progress=config["show_progress"]
        )

        keys = [
            'ndcg@10',
            'giniindex@10',
            'itemcoveragen@10'
            ]

        max_key_len = max(len(k) for k in keys)
        print(f"{'Metric':<{max_key_len}} | Value")
        print(f"{'-'*max_key_len}-|-------")
        print(test_result)
        for key in keys:
            value = test_result[key]
            print(f"{key:<{max_key_len}} | {value:>7.4f}")

    elif args.analyze:
        config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
            model_file=args.path
        )
        trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
        trainer.eval_collector.data_collect(train_data)

        trainer.analyze_neurons(train_data, model_file=args.path, eval_data=False)
        exit()

    elif args.save_activations:
        config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
            model_file=args.path
        )
        trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
        trainer.eval_collector.data_collect(train_data)

        trainer.save_SAE_activations(train_data, model_file=args.path, eval_data=False, epochs=args.epochs_save if hasattr(args, "epochs_save") else None)
        exit()
