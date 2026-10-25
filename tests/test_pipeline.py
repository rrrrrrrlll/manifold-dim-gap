"""Fast checks of the generator, graph builder, model wiring and runner (CPU, < 1 min)."""
import numpy as np
import pytest
import torch

from mdgap.graphs import knn_graph
from mdgap.manifolds import ambient_map, intrinsic_ambient_width, sample_manifold
from mdgap.models import GNNModel
from mdgap.train import make_graph, run_one

CFG = {"ambient_dim": 128, "embed_seed": 0, "noise_sigma": 0.0, "k": 10, "test_graphs": 2,
       "model": {"hidden_size": 32, "layers": 2, "dropout": 0.5, "gnn_conv": "GCNConv"},
       "optim": {"lr": 0.01, "epochs": 20}, "trace_every": 0}


@pytest.mark.parametrize("manifold", ["torus", "sphere"])
@pytest.mark.parametrize("m", [2, 7, 24])
def test_points_lie_on_unit_manifold_and_map_is_isometric(manifold, m):
    q = ambient_map(manifold, m, 128)
    assert q.shape == (128, intrinsic_ambient_width(manifold, m))
    assert np.allclose(q.T @ q, np.eye(q.shape[1]), atol=1e-10)
    s = sample_manifold(manifold, m, 500, seed=3)
    assert s.x.shape == (500, 128) and s.latent.shape[1] == (m if manifold == "torus" else m + 1)
    assert np.allclose(np.linalg.norm(s.x, axis=1), 1.0, atol=1e-5)   # unit norm in both cases


@pytest.mark.parametrize("manifold", ["torus", "sphere"])
def test_labels_balanced(manifold):
    for m in (2, 24):
        y = sample_manifold(manifold, m, 40000, seed=0).y
        frac = np.bincount(y, minlength=4) / len(y)
        assert np.all(np.abs(frac - 0.25) < 0.01), (m, frac)


def test_torus_label_rule_uses_first_two_angles():
    s = sample_manifold("torus", 5, 1000, seed=1)
    expected = 2 * (s.latent[:, 0] >= np.pi) + (s.latent[:, 1] >= np.pi)
    assert np.array_equal(s.y, expected)


def test_streams_and_seeds_are_independent_and_reproducible():
    a = sample_manifold("torus", 4, 100, seed=0, stream=0).x
    assert np.array_equal(a, sample_manifold("torus", 4, 100, seed=0, stream=0).x)
    assert not np.allclose(a, sample_manifold("torus", 4, 100, seed=0, stream=1).x)
    assert not np.allclose(a, sample_manifold("torus", 4, 100, seed=1, stream=0).x)


def test_knn_graph_symmetric_weighted_no_self_loops():
    x = sample_manifold("sphere", 3, 800, seed=2).x
    ei, ew, stats = knn_graph(x, k=10)
    assert ei.shape[1] == ew.shape[0]
    assert not torch.any(ei[0] == ei[1])
    pairs = set(map(tuple, ei.t().tolist()))
    assert all((j, i) in pairs for i, j in pairs)
    assert torch.all((ew > 0) & (ew <= 1))
    assert stats["mean_degree"] >= 10 and stats["isolated_frac"] == 0


def test_model_full_batch_forward():
    data, _ = make_graph(sample_manifold("torus", 2, 300, seed=0), 10, "cpu")
    model = GNNModel(in_features=128, hidden_size=32, out_size=4, gnn_conv="GCNConv",
                     gnn_conv_args={}, layers=2, dropout=0.5)
    assert model(data).shape == (300, 4)


def test_run_one_reproducible_on_cpu():
    r1 = run_one(CFG, "torus", 3, 300, 0, "mse", torch.device("cpu"))
    r2 = run_one(CFG, "torus", 3, 300, 0, "mse", torch.device("cpu"))
    for key in ("train_loss", "test_loss", "gen_gap_loss", "test_acc"):
        assert np.isfinite(r1[key]) and r1[key] == r2[key]


def test_id_estimators_recover_torus_dimension():
    from mdgap.id_estimation import run_id_scale_sweep
    x = sample_manifold("torus", 4, 5000, seed=0).x.astype(np.float64)
    recs = run_id_scale_sweep(x, size_fractions=(1.0,), methods=("mle", "twonn"), random_state=0)
    for r in recs:
        assert 3.0 < r["estimated_id"] < 5.0, r


def test_trace_records_fresh_graph_gap():
    cfg = {**CFG, "trace_every": 10}
    trace = []
    row = run_one(cfg, "torus", 3, 300, 0, "mse", torch.device("cpu"), trace)
    assert [t["epoch"] for t in trace] == [10, 20]
    assert all(abs(t["gap_loss_eval"] - (t["test_loss_eval"] - t["train_loss_eval"])) < 1e-9 for t in trace)
    assert abs(trace[-1]["train_loss_eval"] - row["train_loss"]) < 1e-6   # same model state at the end


def test_pilot_summary_helpers():
    from mdgap.summarize_pilot import gap_settling_epoch, levelling_off_epoch
    e = np.array([50, 100, 150, 200, 250])
    assert levelling_off_epoch(e, [1.0, 0.5, 0.40, 0.395, 0.394], tol=0.02) == 150
    assert gap_settling_epoch(e, [0.1, 0.5, 0.9, 1.02, 1.0], rel_tol=0.1) == 150


def _simulated_grid(beta_fn, seeds, sd=0.2, rng_seed=0):
    import pandas as pd
    rng = np.random.default_rng(rng_seed)
    rows = [{"m": m, "n": n, "seed": s, "gen_gap_loss": float(np.exp(-beta_fn(m) * np.log(n) + rng.normal(0, sd)))}
            for m in (2, 4, 6, 8, 12, 16, 24) for n in (250, 500, 1000, 2000, 4000, 8000, 16000) for s in range(seeds)]
    return pd.DataFrame(rows)


def test_decision_rule_uses_t_and_corrected_bootstrap_se():
    from scipy import stats
    from mdgap.analyze_decay_exponent import analyze
    summ, per_m, _ = analyze(_simulated_grid(lambda m: 1 / (m + 4), seeds=6), n_boot=300)
    assert summ["decision"].startswith("support")
    assert summ["seeds_min"] == 6 and summ["t_df"] == 5
    assert np.isclose(summ["t_crit"], stats.t.ppf(0.95, 5))
    assert np.isclose(summ["b1_se"], summ["b1_se_boot"] * np.sqrt(6 / 5))
    assert np.isclose(summ["b1_upper95_one_sided"], summ["b1_hat"] + summ["t_crit"] * summ["b1_se"])
    span = 24 - 2
    assert np.isclose(summ["total_change_hi90"], summ["b1_upper95_one_sided"] * span)
    # constant exponent with little noise: the change over m is pinned near 0 -> "no dependence"
    summ0, _, _ = analyze(_simulated_grid(lambda m: 0.1, seeds=6, sd=0.05), n_boot=300)
    assert summ0["decision"].startswith("evidence of no dependence")


def test_decision_rule_not_estimable_with_one_seed():
    from mdgap.analyze_decay_exponent import analyze
    summ, _, _ = analyze(_simulated_grid(lambda m: 0.1, seeds=1), n_boot=50)
    assert summ["decision"].startswith("not estimable")
