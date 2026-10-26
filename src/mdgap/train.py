"""
Grid runner for the synthetic-manifold experiments, one config file per grid.

For every (manifold, m, N, seed, loss) cell in the config it:
  1. samples N training points (stream 0) and builds the kNN graph on them;
  2. trains the GNN full-batch for a fixed number of epochs (no early stopping);
  3. evaluates the trained model, in eval mode, on its own training graph and on
     `test_graphs` fresh graphs of N points each (streams 1..T), built the same way;
  4. appends one row to results/<experiment>/<manifold>_<loss>.csv right away,
     so an interrupted job loses at most the run in progress.

Re-launching the same command skips rows already present (same key columns).

    python -m mdgap.train --config configs/clean_manifolds.yaml --manifold torus --loss mse
    python -m mdgap.train --config configs/pilot.yaml --dry-run
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
from mdgap.manifolds import MANIFOLD_CODES, NUM_CLASSES, sample_manifold
from mdgap.models import GNNModel

REPO_ROOT = Path(__file__).resolve().parents[2]
KEY_COLUMNS = ("manifold", "m", "n", "seed", "loss", "k", "epochs", "noise_sigma", "ambient_dim", "embed_seed")


def seed_list(spec):
    if isinstance(spec, int):
        return list(range(spec))
    if isinstance(spec, dict):
        return list(range(spec.get("offset", 0), spec.get("offset", 0) + spec["count"]))
    return [int(s) for s in spec]


def loss_fn(name):
    if name == "mse":
        # L2 loss on one-hot targets, closest to the bound's Lipschitz (L2) loss
        return lambda out, y: F.mse_loss(out, F.one_hot(y, NUM_CLASSES).float())
    if name == "ce":
        return F.cross_entropy
    raise ValueError(f"unknown loss {name!r}")


def make_graph(sample, k, device):
    edge_index, edge_weight, stats = knn_graph(sample.x, k=k)
    data = Data(x=torch.from_numpy(sample.x), edge_index=edge_index, edge_weight=edge_weight,
                y=torch.from_numpy(sample.y))
    data.batch_size = data.num_nodes   # GNNModel.forward returns out[:batch.batch_size]
    return data.to(device), stats


@torch.no_grad()
def evaluate(model, data, crit):
    model.eval()
    out = model(data)
    return float(crit(out, data.y)), float((out.argmax(1) == data.y).float().mean() * 100)


def run_one(cfg, manifold, m, n, seed, loss_name, device, trace_rows=None):
    t0 = time.time()
    gen = dict(ambient_dim=cfg["ambient_dim"], embed_seed=cfg["embed_seed"], noise_sigma=cfg["noise_sigma"])
    k, epochs = cfg["k"], cfg["optim"]["epochs"]
    crit = loss_fn(loss_name)

    train_data, gstats = make_graph(sample_manifold(manifold, m, n, seed, stream=0, **gen), k, device)

    init_seed = int(np.random.default_rng([seed, MANIFOLD_CODES[manifold], m, n, 10_000]).integers(2 ** 31))
    torch.manual_seed(init_seed)
    mc = cfg["model"]
    model = GNNModel(in_features=cfg["ambient_dim"], hidden_size=mc["hidden_size"], out_size=NUM_CLASSES,
                     gnn_conv=mc["gnn_conv"], gnn_conv_args={}, layers=mc["layers"], dropout=mc["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["optim"]["lr"])

    history = []
    trace_every = cfg.get("trace_every", 0)
    tracing = trace_rows is not None and trace_every
    if tracing:
        # one fresh graph (the same as the first final test graph, stream 1) to follow the gap
        # during training; it is only evaluated, never trained on
        trace_test_data, _ = make_graph(sample_manifold(manifold, m, n, seed, stream=1, **gen), k, device)
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        loss = crit(model(train_data), train_data.y)
        loss.backward()
        opt.step()
        history.append(float(loss))
        if tracing and (epoch % trace_every == 0 or epoch == epochs):
            tr_loss, tr_acc = evaluate(model, train_data, crit)
            te_loss, te_acc = evaluate(model, trace_test_data, crit)
            trace_rows.append({"epoch": epoch, "train_loss_trainmode": history[-1],
                               "train_loss_eval": tr_loss, "train_acc_eval": tr_acc,
                               "test_loss_eval": te_loss, "test_acc_eval": te_acc,
                               "gap_loss_eval": te_loss - tr_loss, "gap_acc_eval": tr_acc - te_acc})

    # plateau check: relative change of the (train-mode) loss between the last two tenths of training
    w = max(1, epochs // 10)
    prev, last = np.mean(history[-2 * w:-w]), np.mean(history[-w:])
    plateau_rel_change = float((last - prev) / prev) if prev > 0 else float("nan")

    train_loss, train_acc = evaluate(model, train_data, crit)
    test_losses, test_accs, test_degrees = [], [], []
    for t in range(1, cfg["test_graphs"] + 1):
        test_data, tstats = make_graph(sample_manifold(manifold, m, n, seed, stream=t, **gen), k, device)
        tl, ta = evaluate(model, test_data, crit)
        test_losses.append(tl)
        test_accs.append(ta)
        test_degrees.append(tstats["mean_degree"])

    test_loss, test_acc = float(np.mean(test_losses)), float(np.mean(test_accs))
    return {
        "manifold": manifold, "m": m, "n": n, "seed": seed, "loss": loss_name,
        "k": k, "epochs": epochs, "noise_sigma": cfg["noise_sigma"],
        "ambient_dim": cfg["ambient_dim"], "embed_seed": cfg["embed_seed"],
        "lr": cfg["optim"]["lr"], "hidden_size": mc["hidden_size"], "layers": mc["layers"], "dropout": mc["dropout"],
        "test_graphs": cfg["test_graphs"],
        "train_loss": train_loss, "test_loss": test_loss, "test_loss_sd": float(np.std(test_losses, ddof=1)) if len(test_losses) > 1 else float("nan"),
        "gen_gap_loss": test_loss - train_loss,
        "train_acc": train_acc, "test_acc": test_acc, "gen_gap_acc": train_acc - test_acc,
        "train_loss_trainmode_final": history[-1], "plateau_rel_change": plateau_rel_change,
        "sigma_train": gstats["sigma"], "mean_degree_train": gstats["mean_degree"],
        "isolated_frac_train": gstats["isolated_frac"], "mean_degree_test": float(np.mean(test_degrees)),
        "runtime_s": time.time() - t0, "device": str(device), "torch": torch.__version__,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def existing_keys(path):
    if not path.exists():
        return set()
    with open(path, newline="") as f:
        return {tuple(str(row[c]) for c in KEY_COLUMNS) for row in csv.DictReader(f)}


def append_row(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    if not new:
        with open(path, newline="") as f:
            header = next(csv.reader(f))
        if header != list(row.keys()):
            raise ValueError(f"{path} has different columns; write to a new results folder")
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--manifold", help="run only this manifold")
    p.add_argument("--loss", help="run only this loss")
    p.add_argument("--m", help="comma-separated subset of m values")
    p.add_argument("--results-root", default=str(REPO_ROOT / "results"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dry-run", action="store_true", help="list the runs that would execute, then stop")
    args = p.parse_args(argv)

    cfg = yaml.safe_load(open(args.config))
    manifolds = [args.manifold] if args.manifold else cfg["manifolds"]
    losses = [args.loss] if args.loss else cfg["losses"]
    ms = [int(v) for v in args.m.split(",")] if args.m else cfg["m"]
    out_dir = Path(args.results_root) / cfg["experiment"]
    device = torch.device(args.device)

    planned = skipped = 0
    for manifold in manifolds:
        for loss_name in losses:
            path = out_dir / f"{manifold}_{loss_name}.csv"
            done = existing_keys(path)
            trace_path = out_dir / f"{manifold}_{loss_name}_trace.csv"
            for m in ms:
                for n in cfg["n"]:
                    for seed in seed_list(cfg["seeds"]):
                        key = (manifold, m, n, seed, loss_name, cfg["k"], cfg["optim"]["epochs"],
                               cfg["noise_sigma"], cfg["ambient_dim"], cfg["embed_seed"])
                        if tuple(str(v) for v in key) in done:
                            skipped += 1
                            continue
                        planned += 1
                        if args.dry_run:
                            print("would run:", dict(zip(KEY_COLUMNS, key)))
                            continue
                        trace = [] if cfg.get("trace_every", 0) else None
                        row = run_one(cfg, manifold, m, n, seed, loss_name, device, trace)
                        append_row(path, row)
                        for tr in trace or []:
                            append_row(trace_path, {**dict(zip(KEY_COLUMNS, key)), **tr})
                        print(f"{manifold} m={m} n={n} seed={seed} {loss_name}: gap_loss={row['gen_gap_loss']:.4f} "
                              f"test_acc={row['test_acc']:.1f} plateau={row['plateau_rel_change']:+.3f} "
                              f"deg={row['mean_degree_train']:.1f} {row['runtime_s']:.1f}s", flush=True)
    print(f"{'planned' if args.dry_run else 'ran'} {planned} runs, skipped {skipped} already in {out_dir}")


if __name__ == "__main__":
    main()
