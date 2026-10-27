"""Checks of the real-embedding pool, runner and analysis on a small fake data folder (CPU)."""
import numpy as np
import pandas as pd
import pytest
import torch

from mdgap.real_data import EmbeddingPool

CFG = {"k": 10, "test_graphs": 2, "model": {"hidden_size": 32, "layers": 2, "dropout": 0.5, "gnn_conv": "GCNConv"},
       "optim": {"lr": 0.01, "epochs": 20}}


@pytest.fixture
def data_dir(tmp_path):
    rng = np.random.default_rng(0)
    n = 900
    x = rng.normal(size=(n, 12)).astype(np.float32) * np.arange(1, 13, dtype=np.float32)
    np.save(tmp_path / "mnist_x.npy", x)
    np.save(tmp_path / "mnist_pca128.npy", x[:, ::-1].copy())      # stand-in for PC scores
    np.save(tmp_path / "mnist_y.npy", rng.integers(0, 10, n).astype(np.int64))
    np.save(tmp_path / "celeba_x.npy", x)
    np.save(tmp_path / "celeba_gb_y.npy", rng.integers(0, 2, n).astype(np.int64))
    np.save(tmp_path / "celeba_sb_y.npy", rng.integers(0, 2, n).astype(np.int64))
    return tmp_path


def test_train_and_test_samples_are_disjoint_and_reproducible(data_dir):
    pool = EmbeddingPool("mnist", "full", None, data_dir)
    tr = pool.indices(400, seed=3, stream=0)
    for t in (1, 2):
        te = pool.indices(400, seed=3, stream=t)
        assert len(te) == 400 and len(set(te)) == 400 and not set(tr) & set(te)
    assert np.array_equal(tr, pool.indices(400, seed=3, stream=0))
    assert not np.array_equal(tr, pool.indices(400, seed=4, stream=0))
    with pytest.raises(ValueError):
        pool.indices(451, seed=0)          # needs a pool of 2N


def test_same_images_across_widths_and_celeba_tasks(data_dir):
    a = EmbeddingPool("mnist", "pca128", 4, data_dir)
    b = EmbeddingPool("mnist", "pca128", 8, data_dir)
    assert a.dim == 4 and b.dim == 8
    assert np.array_equal(a.indices(200, 1), b.indices(200, 1))
    assert np.array_equal(a.sample(200, 1).x, b.sample(200, 1).x[:, :4])
    gb, sb = EmbeddingPool("celeba_gb", "full", None, data_dir), EmbeddingPool("celeba_sb", "full", None, data_dir)
    assert np.array_equal(gb.indices(300, 2), sb.indices(300, 2)) and gb.num_classes == 2


def test_standardization_uses_pool_statistics(data_dir):
    pool = EmbeddingPool("mnist", "full", None, data_dir)
    z = pool.standardize(pool.x)
    assert np.allclose(z.mean(0), 0, atol=1e-4) and np.allclose(z.std(0), 1, atol=1e-3)


def test_run_one_real(data_dir):
    from mdgap.train_real import run_one
    pool = EmbeddingPool("mnist", "pca128", 8, data_dir)
    row = run_one(CFG, pool, 300, 0, "mse", torch.device("cpu"))
    assert row["width"] == 8 and row["dim"] == 8 and row["num_classes"] == 10
    assert abs(row["gen_gap_loss"] - (row["test_loss"] - row["train_loss"])) < 1e-9
    row2 = run_one(CFG, pool, 300, 0, "mse", torch.device("cpu"))
    assert abs(row["train_loss"] - row2["train_loss"]) < 1e-6


def test_runner_main_writes_and_skips(data_dir, tmp_path):
    import yaml
    from mdgap.train_real import main
    cfg = {**CFG, "experiment": "t", "datasets": ["mnist"], "features": "pca128", "widths": [4, 8], "n": [100, 200],
           "seeds": 1, "losses": ["mse"]}
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    args = ["--config", str(p), "--data-dir", str(data_dir), "--results-root", str(tmp_path / "res"), "--device", "cpu"]
    main(args)
    d = pd.read_csv(tmp_path / "res" / "t" / "mnist_mse.csv")
    assert len(d) == 4 and set(d.width) == {4, 8}
    main(args)
    assert len(pd.read_csv(tmp_path / "res" / "t" / "mnist_mse.csv")) == 4


def test_spearman_exact_and_beta_table():
    from mdgap.analyze_real import beta_table, spearman_exact
    rho, p, n = spearman_exact([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])
    assert rho == pytest.approx(-1) and p == pytest.approx(2 / 120) and n == 5
    rng = np.random.default_rng(0)
    rows = [{"g": g, "n": n, "seed": s, "gen_gap_loss": float(np.exp(-b * np.log(n) + rng.normal(0, 0.05)))}
            for g, b in (("a", 0.9), ("b", 0.3)) for n in (250, 1000, 4000, 16000) for s in range(5)]
    t, _ = beta_table(pd.DataFrame(rows), "g", n_boot=100)
    assert t.set_index("g").beta_hat["a"] == pytest.approx(0.9, abs=0.05)
    assert t.set_index("g").beta_hat["b"] == pytest.approx(0.3, abs=0.05)
