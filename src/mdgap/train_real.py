"""
Grid runner for experiments B and C2 (real image embeddings; docs/analysis_plan_bc.md).

Same training and evaluation as mdgap.train (experiment A), with nodes drawn from a pool of real
embeddings instead of a synthetic manifold. For every (dataset, width, N, seed, loss) in the config:
  1. draws N training images and builds the kNN graph on their features;
  2. trains the GNN full-batch for a fixed number of epochs, on features standardized with the
     pool's mean and SD;
  3. evaluates it on its training graph and on `test_graphs` graphs of N images that are not in the
     training sample;
  4. appends one row to results/<experiment>/<dataset>_<loss>.csv (finished rows are skipped on
     re-launch).

    python -m mdgap.train_real --config configs/real_pca_width.yaml
    python -m mdgap.train_real --config configs/real_full.yaml --dataset cifar10 --dry-run
"""
import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch_geometric.data import Data

from mdgap.graphs import knn_graph
from mdgap.models import GNNModel
from mdgap.real_data import EmbeddingPool
from mdgap.train import append_row, evaluate, seed_list

REPO_ROOT = Path(__file__).resolve().parents[2]
KEY_COLUMNS = ("dataset", "width", "n", "seed", "loss", "k", "epochs")


def existing_keys(path):
    if not path.exists():
        return set()
    with open(path, newline="") as f:
        return {tuple(str(row[c]) for c in KEY_COLUMNS) for row in csv.DictReader(f)}


def loss_fn(name, num_classes):
    if name == "mse":
        return lambda out, y: F.mse_loss(out, F.one_hot(y, num_classes).float())
    if name == "ce":
        return F.cross_entropy
    raise ValueError(f"unknown loss {name!r}")


def make_graph(pool, sample, k, device):
    edge_index, edge_weight, stats = knn_graph(sample.x, k=k)       # graph on the features as given
    data = Data(x=torch.from_numpy(pool.standardize(sample.x)), edge_index=edge_index,
                edge_weight=edge_weight, y=torch.from_numpy(sample.y))
    data.batch_size = data.num_nodes   # GNNModel.forward returns out[:batch.batch_size]
    return data.to(device), stats


def width_label(width):
    return "full" if width is None else int(width)


def run_one(cfg, pool, n, seed, loss_name, device):
    t0 = time.time()
    k, epochs = cfg["k"], cfg["optim"]["epochs"]
    crit = loss_fn(loss_name, pool.num_classes)
    train_data, gstats = make_graph(pool, pool.sample(n, seed, 0), k, device)

    init_seed = int(np.random.default_rng([seed, pool.code, n, 10_000]).integers(2 ** 31))
    torch.manual_seed(init_seed)
    mc = cfg["model"]
    model = GNNModel(in_features=pool.dim, hidden_size=mc["hidden_size"], out_size=pool.num_classes,
                     gnn_conv=mc["gnn_conv"], gnn_conv_args={}, layers=mc["layers"], dropout=mc["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["optim"]["lr"])

    history = []
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = crit(model(train_data), train_data.y)
        loss.backward()
        opt.step()
        history.append(float(loss))

    # convergence check: relative change of the training loss between the last two tenths of training
    w = max(1, epochs // 10)
    prev, last = np.mean(history[-2 * w:-w]), np.mean(history[-w:])
    plateau_rel_change = float((last - prev) / prev) if prev > 0 else float("nan")

    train_loss, train_acc = evaluate(model, train_data, crit)
    test_losses, test_accs, test_degrees = [], [], []
    for t in range(1, cfg["test_graphs"] + 1):
        test_data, tstats = make_graph(pool, pool.sample(n, seed, t), k, device)
        tl, ta = evaluate(model, test_data, crit)
        test_losses.append(tl)
        test_accs.append(ta)
        test_degrees.append(tstats["mean_degree"])

    test_loss, test_acc = float(np.mean(test_losses)), float(np.mean(test_accs))
    return {
        "dataset": pool.dataset, "width": width_label(pool.width), "n": n, "seed": seed, "loss": loss_name,
        "k": k, "epochs": epochs, "features": pool.features, "dim": pool.dim, "pool_size": pool.size,
        "num_classes": pool.num_classes,
        "lr": cfg["optim"]["lr"], "hidden_size": mc["hidden_size"], "layers": mc["layers"], "dropout": mc["dropout"],
        "test_graphs": cfg["test_graphs"],
        "train_loss": train_loss, "test_loss": test_loss,
        "test_loss_sd": float(np.std(test_losses, ddof=1)) if len(test_losses) > 1 else float("nan"),
        "gen_gap_loss": test_loss - train_loss,
        "train_acc": train_acc, "test_acc": test_acc, "gen_gap_acc": train_acc - test_acc,
        "train_loss_trainmode_final": history[-1], "plateau_rel_change": plateau_rel_change,
        "sigma_train": gstats["sigma"], "mean_degree_train": gstats["mean_degree"],
        "isolated_frac_train": gstats["isolated_frac"], "mean_degree_test": float(np.mean(test_degrees)),
        "runtime_s": time.time() - t0, "device": str(device), "torch": torch.__version__,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--dataset", help="comma-separated subset of the config's datasets")
    p.add_argument("--data-dir", default=str(REPO_ROOT / "data" / "real"))
    p.add_argument("--results-root", default=str(REPO_ROOT / "results"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dry-run", action="store_true", help="list the runs that would execute, then stop")
    args = p.parse_args(argv)

    cfg = yaml.safe_load(open(args.config))
    datasets = args.dataset.split(",") if args.dataset else cfg["datasets"]
    widths = cfg.get("widths") or [None]          # None = all columns of the feature file
    out_dir = Path(args.results_root) / cfg["experiment"]
    device = torch.device(args.device)

    planned = skipped = 0
    for dataset in datasets:
        for width in widths:
            pool = None   # loaded lazily: a dry run or a finished cell does not read the data
            for loss_name in cfg["losses"]:
                path = out_dir / f"{dataset}_{loss_name}.csv"
                done = existing_keys(path)
                for n in cfg["n"]:
                    for seed in seed_list(cfg["seeds"]):
                        key = (dataset, width_label(width), n, seed, loss_name, cfg["k"], cfg["optim"]["epochs"])
                        if tuple(str(v) for v in key) in done:
                            skipped += 1
                            continue
                        planned += 1
                        if args.dry_run:
                            print("would run:", dict(zip(KEY_COLUMNS, key)))
                            continue
                        if pool is None:
                            pool = EmbeddingPool(dataset, cfg["features"], width, args.data_dir)
                        row = run_one(cfg, pool, n, seed, loss_name, device)
                        append_row(path, row)
                        print(f"{dataset} width={row['width']} n={n} seed={seed} {loss_name}: "
                              f"gap_loss={row['gen_gap_loss']:.4f} test_acc={row['test_acc']:.1f} "
                              f"plateau={row['plateau_rel_change']:+.3f} deg={row['mean_degree_train']:.1f} "
                              f"{row['runtime_s']:.1f}s", flush=True)
            del pool
    print(f"{'planned' if args.dry_run else 'ran'} {planned} runs, skipped {skipped} already in {out_dir}")


if __name__ == "__main__":
    main()
