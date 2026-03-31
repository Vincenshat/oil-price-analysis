import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


@dataclass
class Config:
    data_path: str = r"d:\Projects\oil_analysis\ML\4.0_enriched.csv"
    out_dir: str = r"d:\Projects\oil_analysis\ML\outputs_lstm_wf"
    lookback: int = 8
    max_epochs: int = 120
    early_stop_patience: int = 12
    clip_price_change_abs: float = 80.0
    top_k_features: int = 36
    feature_corr_threshold: float = 0.96
    n_folds: int = 4
    val_size: int = 20
    test_size: int = 20
    search_rounds: int = 2
    trials_per_round: int = 50
    ensemble_top_n: int = 5
    recent_tune_window: int = 96


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float, model_type: str = "lstm"):
        super().__init__()
        model_type = str(model_type).lower()
        if model_type not in {"lstm", "gru"}:
            raise ValueError(f"Unsupported model_type: {model_type}")
        rnn_cls = nn.LSTM if model_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.attn = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        attn_w = torch.softmax(self.attn(out).squeeze(-1), dim=1).unsqueeze(-1)
        context = torch.sum(out * attn_w, dim=1)
        return self.head(context)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    yt = np.asarray(y_true).reshape(-1)
    yp = np.asarray(y_pred).reshape(-1)
    return {
        "price_change_t1_mae": float(mean_absolute_error(yt, yp)),
        "price_change_t1_rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "price_change_t1_r2": float(r2_score(yt, yp)),
        "price_change_t1_direction_acc": float(np.mean(np.sign(yt) == np.sign(yp))),
        "price_change_t1_mean_error": float(np.mean(yp - yt)),
    }


def build_weekly_table(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()
    df["Event_Date"] = pd.to_datetime(df["Event_Date"], errors="coerce")
    df = df.dropna(subset=["Event_Date"]).sort_values("Event_Date")
    df = df[df["WTI_T"].notna() & df["Stock_T"].notna() & df["Price_Change"].notna() & df["Stock_Change"].notna()]
    df = df[df["Price_Change"].abs() <= cfg.clip_price_change_abs]

    # Focus features strictly on oil inventory and oil price behavior.
    base_num_cols = ["WTI_T", "Stock_T", "Price_Change", "Stock_Change"]
    for col in base_num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    weekly = (
        df.groupby("Event_Date", as_index=False)
        .agg(
            {
                "WTI_T": "mean",
                "Stock_T": "mean",
                "Price_Change": "mean",
                "Stock_Change": "mean",
            }
        )
        .sort_values("Event_Date")
    )

    full_idx = pd.date_range(weekly["Event_Date"].min(), weekly["Event_Date"].max(), freq="W-FRI")
    weekly = weekly.set_index("Event_Date").reindex(full_idx).rename_axis("Event_Date").reset_index()

    num_cols = [c for c in weekly.columns if c != "Event_Date"]
    # Leakage-safe fill: only forward fill, never backward fill from future.
    weekly[num_cols] = weekly[num_cols].ffill()

    lag_cols = ["WTI_T", "Stock_T", "Price_Change", "Stock_Change"]
    for col in lag_cols:
        for lag in [1, 2, 3, 4, 6, 8]:
            weekly[f"{col}_lag{lag}"] = weekly[col].shift(lag)

    for col in ["WTI_T", "Stock_T", "Price_Change", "Stock_Change"]:
        weekly[f"{col}_roll3_mean"] = weekly[col].rolling(3, min_periods=2).mean()
        weekly[f"{col}_roll6_mean"] = weekly[col].rolling(6, min_periods=3).mean()
        weekly[f"{col}_roll12_mean"] = weekly[col].rolling(12, min_periods=6).mean()
        weekly[f"{col}_roll3_std"] = weekly[col].rolling(3, min_periods=2).std()
        weekly[f"{col}_roll6_std"] = weekly[col].rolling(6, min_periods=3).std()
        weekly[f"{col}_roll12_std"] = weekly[col].rolling(12, min_periods=6).std()

    chg_cols = ["WTI_T", "Stock_T", "Price_Change", "Stock_Change"]
    for col in chg_cols:
        weekly[f"{col}_diff1"] = weekly[col].diff(1)
        weekly[f"{col}_diff2"] = weekly[col].diff(2)
        weekly[f"{col}_pct1"] = weekly[col].pct_change(1)
        weekly[f"{col}_pct2"] = weekly[col].pct_change(2)

    # Inventory-price relationship features.
    weekly["inv_price_spread"] = weekly["Stock_Change"] - weekly["Price_Change"]
    weekly["inv_price_product"] = weekly["Stock_Change"] * weekly["Price_Change"]
    weekly["inv_to_price_ratio"] = weekly["Stock_Change"] / (weekly["Price_Change"].abs() + 1.0)
    weekly["price_to_inv_ratio"] = weekly["Price_Change"] / (weekly["Stock_Change"].abs() + 1.0)
    weekly["inv_abs"] = weekly["Stock_Change"].abs()
    weekly["price_abs"] = weekly["Price_Change"].abs()
    weekly["inv_momentum_3"] = weekly["Stock_Change"] - weekly["Stock_Change"].shift(3)
    weekly["price_momentum_3"] = weekly["Price_Change"] - weekly["Price_Change"].shift(3)

    weekly = weekly.replace([np.inf, -np.inf], np.nan)
    num_cols = [c for c in weekly.columns if c != "Event_Date"]
    # Keep preprocessing causal for time-series evaluation.
    weekly[num_cols] = weekly[num_cols].ffill()
    # For engineered columns that are undefined at the beginning, use 0 instead of future info.
    weekly[num_cols] = weekly[num_cols].fillna(0.0)
    weekly = weekly.copy()

    weekly["target_price_change_t1"] = weekly["Price_Change"].shift(-1)
    weekly = weekly.dropna(subset=["target_price_change_t1"]).reset_index(drop=True)
    return weekly


def build_sequences(weekly: pd.DataFrame, lookback: int) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp], List[str]]:
    feature_cols = [c for c in weekly.columns if c not in ["Event_Date", "target_price_change_t1"]]
    Xv = weekly[feature_cols].values.astype(np.float32)
    yv = weekly["target_price_change_t1"].values.astype(np.float32).reshape(-1, 1)
    dates = weekly["Event_Date"].tolist()
    X_seq, y_seq, y_dates = [], [], []
    for i in range(lookback - 1, len(weekly)):
        X_seq.append(Xv[i - lookback + 1 : i + 1])
        y_seq.append(yv[i])
        y_dates.append(dates[i])
    return np.array(X_seq), np.array(y_seq), y_dates, feature_cols


def make_walkforward_splits(n: int, cfg: Config) -> List[Dict[str, int]]:
    required = cfg.n_folds * (cfg.val_size + cfg.test_size)
    train0 = n - required
    if train0 < 80:
        raise ValueError(f"Not enough samples for walk-forward: n={n}, initial_train={train0}")
    splits = []
    for i in range(cfg.n_folds):
        train_end = train0 + i * cfg.test_size
        val_end = train_end + cfg.val_size
        test_end = val_end + cfg.test_size
        splits.append(
            {
                "fold": i + 1,
                "train_start": 0,
                "train_end": train_end,
                "val_start": train_end,
                "val_end": val_end,
                "test_start": val_end,
                "test_end": test_end,
            }
        )
    return splits


def select_feature_indices(X_train: np.ndarray, y_train: np.ndarray, top_k: int, corr_threshold: float) -> List[int]:
    x_last = X_train[:, -1, :]
    n_feat = x_last.shape[1]
    if n_feat <= top_k:
        return list(range(n_feat))
    valid = np.nanstd(x_last, axis=0) > 1e-12
    rel = np.zeros(n_feat, dtype=float)
    for i in range(n_feat):
        if not valid[i]:
            continue
        c1 = np.corrcoef(x_last[:, i], y_train[:, 0])[0, 1]
        val = abs(c1)
        rel[i] = 0.0 if np.isnan(val) else float(val)
    ranked = [i for i in np.argsort(-rel) if valid[i]]
    selected: List[int] = []
    for idx in ranked:
        keep = True
        for j in selected:
            cc = np.corrcoef(x_last[:, idx], x_last[:, j])[0, 1]
            cc = 0.0 if np.isnan(cc) else abs(float(cc))
            if cc >= corr_threshold:
                keep = False
                break
        if keep:
            selected.append(int(idx))
        if len(selected) >= top_k:
            break
    if len(selected) < min(12, n_feat):
        selected = list(np.argsort(-rel)[: min(top_k, n_feat)])
    return sorted(selected)


def transform_by_train(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray):
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_scaler.fit(X_train.reshape(-1, X_train.shape[-1]))
    y_scaler.fit(y_train)

    def _tx(x: np.ndarray) -> np.ndarray:
        s = x.shape
        return x_scaler.transform(x.reshape(-1, s[-1])).reshape(s).astype(np.float32)

    return _tx(X_train), _tx(X_val), _tx(X_test), y_scaler.transform(y_train).astype(np.float32), y_scaler.transform(y_val).astype(np.float32), y_scaler.transform(y_test).astype(np.float32), x_scaler, y_scaler


def train_one_trial(
    X_train_s: np.ndarray,
    y_train_s: np.ndarray,
    X_val_s: np.ndarray,
    y_val_s: np.ndarray,
    cfg: Config,
    trial: Dict[str, float],
) -> Dict[str, object]:
    train_loader = DataLoader(SequenceDataset(X_train_s, y_train_s), batch_size=int(trial["batch_size"]), shuffle=True)
    val_loader = DataLoader(SequenceDataset(X_val_s, y_val_s), batch_size=int(trial["batch_size"]), shuffle=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTMRegressor(
        X_train_s.shape[-1],
        int(trial["hidden_size"]),
        int(trial["num_layers"]),
        float(trial["dropout"]),
        str(trial.get("model_type", "lstm")),
    ).to(device)
    loss_type = str(trial.get("loss_type", "mse")).lower()
    if loss_type == "huber":
        criterion = nn.SmoothL1Loss(beta=float(trial.get("huber_delta", 1.0)))
    else:
        criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(trial["learning_rate"]), weight_decay=float(trial["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5)

    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    history = []
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
            train_losses.append(loss.item())
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_losses.append(criterion(model(xb), yb).item())
        tr = float(np.mean(train_losses))
        va = float(np.mean(val_losses))
        scheduler.step(va)
        history.append({"epoch": epoch, "train_loss": tr, "val_loss": va})
        if va < best_val:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= cfg.early_stop_patience:
            break

    if best_state is None:
        raise RuntimeError("No checkpoint in trial.")
    model.load_state_dict(best_state)

    def _pred(x_arr: np.ndarray) -> np.ndarray:
        loader = DataLoader(SequenceDataset(x_arr, np.zeros((len(x_arr), 1), dtype=np.float32)), batch_size=64, shuffle=False)
        outs = []
        model.eval()
        with torch.no_grad():
            for xb, _ in loader:
                outs.append(model(xb.to(device)).cpu().numpy())
        return np.vstack(outs)

    return {
        "val_loss": best_val,
        "history": history,
        "trial": trial,
        "device": device,
        "model_state": best_state,
        "pred_val_s": _pred(X_val_s),
        "predict_func": _pred,
    }


def sample_trial_grid(cfg: Config, round_idx: int) -> List[Dict[str, float]]:
    rng = np.random.default_rng(SEED + 97 * (round_idx + 1))
    grid: List[Dict[str, float]] = []
    anchor_trials = [
        {"model_type": "lstm", "hidden_size": 64, "num_layers": 2, "dropout": 0.10, "learning_rate": 1.2e-3, "weight_decay": 1e-4, "batch_size": 32, "loss_type": "mse", "huber_delta": 1.0},
        {"model_type": "lstm", "hidden_size": 64, "num_layers": 2, "dropout": 0.15, "learning_rate": 1.0e-3, "weight_decay": 8e-5, "batch_size": 32, "loss_type": "mse", "huber_delta": 1.0},
        {"model_type": "lstm", "hidden_size": 80, "num_layers": 2, "dropout": 0.12, "learning_rate": 9e-4, "weight_decay": 1.2e-4, "batch_size": 24, "loss_type": "mse", "huber_delta": 1.0},
        {"model_type": "lstm", "hidden_size": 96, "num_layers": 2, "dropout": 0.16, "learning_rate": 8e-4, "weight_decay": 1.5e-4, "batch_size": 24, "loss_type": "huber", "huber_delta": 1.2},
        {"model_type": "gru", "hidden_size": 64, "num_layers": 2, "dropout": 0.10, "learning_rate": 1.0e-3, "weight_decay": 1e-4, "batch_size": 32, "loss_type": "mse", "huber_delta": 1.0},
        {"model_type": "gru", "hidden_size": 80, "num_layers": 2, "dropout": 0.15, "learning_rate": 8.5e-4, "weight_decay": 1.2e-4, "batch_size": 24, "loss_type": "huber", "huber_delta": 1.0},
    ]
    for t in anchor_trials:
        grid.append(dict(t))

    remaining = max(0, cfg.trials_per_round - len(grid))
    for _ in range(remaining):
        model_type = str(rng.choice(["lstm", "gru"]))
        loss_type = str(rng.choice(["mse", "huber"]))
        lr = float(np.exp(rng.uniform(np.log(4e-4), np.log(1.6e-3))))
        wd = float(np.exp(rng.uniform(np.log(5e-6), np.log(3e-4))))
        trial = {
            "model_type": model_type,
            "hidden_size": int(rng.choice([64, 80, 96, 128])),
            "num_layers": int(rng.choice([1, 2, 3])),
            "dropout": float(rng.uniform(0.06, 0.28)),
            "learning_rate": lr,
            "weight_decay": wd,
            "batch_size": int(rng.choice([16, 24, 32])),
            "loss_type": loss_type,
            "huber_delta": float(rng.choice([0.8, 1.0, 1.2, 1.5])),
        }
        grid.append(trial)
    return grid


def plot_process_flow(out_dir: Path):
    plt.figure(figsize=(12, 2.8))
    plt.axis("off")
    steps = [
        "Raw Table",
        "Stock/Price FE",
        "Sequence Build",
        "Walk-Forward Folds",
        "BiLSTM+Attn Tuning",
        "Metrics + Evaluation Plots",
    ]
    xs = np.linspace(0.05, 0.95, len(steps))
    for i, (x, s) in enumerate(zip(xs, steps)):
        plt.text(x, 0.5, s, ha="center", va="center", fontsize=10, bbox=dict(boxstyle="round,pad=0.4", facecolor="#dbeafe", edgecolor="#2563eb"))
        if i < len(steps) - 1:
            plt.annotate("", xy=(xs[i + 1] - 0.06, 0.5), xytext=(x + 0.06, 0.5), arrowprops=dict(arrowstyle="->", lw=1.8))
    plt.tight_layout()
    plt.savefig(out_dir / "process_flow.png", dpi=160)
    plt.close()


def plot_splits(out_dir: Path, splits: List[Dict[str, int]], n: int):
    plt.figure(figsize=(11, 4))
    for i, sp in enumerate(splits):
        y = len(splits) - i
        plt.hlines(y, 0, n, color="#e5e7eb", linewidth=8)
        plt.hlines(y, sp["train_start"], sp["train_end"], color="#2563eb", linewidth=8, label="train" if i == 0 else "")
        plt.hlines(y, sp["val_start"], sp["val_end"], color="#f59e0b", linewidth=8, label="val" if i == 0 else "")
        plt.hlines(y, sp["test_start"], sp["test_end"], color="#16a34a", linewidth=8, label="test" if i == 0 else "")
        plt.text(sp["test_end"] + 1, y, f"fold {sp['fold']}", va="center", fontsize=9)
    plt.title("Walk-Forward Splits")
    plt.xlabel("Sequence Index")
    plt.yticks([])
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "walkforward_splits.png", dpi=160)
    plt.close()


def plot_fold_metrics(out_dir: Path, folds_df: pd.DataFrame):
    x = folds_df["fold"].values
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(x, folds_df["price_change_t1_mae"], marker="o", label="price MAE", color="#2563eb")
    plt.title("Price MAE by Fold")
    plt.xlabel("Fold")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, folds_df["price_change_t1_r2"], marker="o", label="price R2", color="#16a34a")
    plt.title("Price R2 by Fold")
    plt.xlabel("Fold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fold_metrics.png", dpi=160)
    plt.close()


def plot_last_fold_predictions(out_dir: Path, pred_df: pd.DataFrame):
    plt.figure(figsize=(10, 4))
    plt.plot(pred_df["Event_Date"], pred_df["y_true_price_change_t1"], label="true")
    plt.plot(pred_df["Event_Date"], pred_df["y_pred_price_change_t1"], label="pred")
    plt.title("Last Fold PriceChange t+1")
    plt.xticks(rotation=35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "last_fold_predictions.png", dpi=160)
    plt.close()


def plot_residual_distribution(out_dir: Path, pred_df: pd.DataFrame):
    rp = pred_df["y_true_price_change_t1"] - pred_df["y_pred_price_change_t1"]
    plt.figure(figsize=(8, 4))
    plt.hist(rp, bins=15, color="#60a5fa", edgecolor="white")
    plt.title("Residual Distribution: PriceChange")
    plt.tight_layout()
    plt.savefig(out_dir / "residual_distribution.png", dpi=160)
    plt.close()


def plot_training_overview(out_dir: Path, n_folds: int):
    hist_files = [out_dir / f"fold_{i}_history.csv" for i in range(1, n_folds + 1)]
    hist_files = [p for p in hist_files if p.exists()]
    if not hist_files:
        return
    plt.figure(figsize=(11, 4))
    plt.subplot(1, 2, 1)
    best_vals = []
    fold_ids = []
    for p in hist_files:
        fold_id = int(p.stem.split("_")[1])
        h = pd.read_csv(p, encoding="utf-8-sig")
        plt.plot(h["epoch"], h["val_loss"], alpha=0.8, label=f"fold{fold_id} val")
        best_vals.append(float(h["val_loss"].min()))
        fold_ids.append(fold_id)
    plt.title("Validation Loss Curves by Fold")
    plt.xlabel("Epoch")
    plt.ylabel("Val Loss")
    plt.legend(fontsize=8, ncol=2)
    plt.subplot(1, 2, 2)
    plt.bar(fold_ids, best_vals, color="#2563eb")
    plt.title("Best Validation Loss per Fold")
    plt.xlabel("Fold")
    plt.ylabel("Best Val Loss")
    plt.tight_layout()
    plt.savefig(out_dir / "training_overview.png", dpi=160)
    plt.close()


def plot_actual_vs_pred_scatter(out_dir: Path, pred_df: pd.DataFrame):
    plt.figure(figsize=(6, 5))
    yt = pred_df["y_true_price_change_t1"].values
    yp = pred_df["y_pred_price_change_t1"].values
    mn = min(float(np.min(yt)), float(np.min(yp)))
    mx = max(float(np.max(yt)), float(np.max(yp)))
    plt.scatter(yt, yp, alpha=0.7, color="#2563eb")
    plt.plot([mn, mx], [mn, mx], "k--", linewidth=1)
    plt.title("PriceChange: True vs Pred")
    plt.xlabel("True")
    plt.ylabel("Pred")
    plt.tight_layout()
    plt.savefig(out_dir / "actual_vs_pred_scatter.png", dpi=160)
    plt.close()


def plot_residual_timeseries(out_dir: Path, pred_df: pd.DataFrame):
    d = pred_df.copy()
    d["Event_Date"] = pd.to_datetime(d["Event_Date"], errors="coerce")
    d["price_resid"] = d["y_true_price_change_t1"] - d["y_pred_price_change_t1"]
    plt.figure(figsize=(10, 4))
    plt.plot(d["Event_Date"], d["price_resid"], marker="o", color="#2563eb")
    plt.axhline(0.0, color="black", linewidth=1, linestyle="--")
    plt.xticks(rotation=35)
    plt.title("Residual Over Time: PriceChange")
    plt.tight_layout()
    plt.savefig(out_dir / "residual_timeseries.png", dpi=160)
    plt.close()


def plot_inventory_price_focus(out_dir: Path, weekly: pd.DataFrame):
    d = weekly.copy()
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(d["Stock_Change"], d["Price_Change"], alpha=0.5, color="#7c3aed")
    c = np.corrcoef(d["Stock_Change"].values, d["Price_Change"].values)[0, 1]
    c = 0.0 if np.isnan(c) else float(c)
    plt.title(f"StockChange vs PriceChange (corr={c:.3f})")
    plt.xlabel("Stock_Change")
    plt.ylabel("Price_Change")
    plt.subplot(1, 2, 2)
    lag_vals = []
    lag_idx = list(range(0, 9))
    px = d["Price_Change"].values
    st = d["Stock_Change"].values
    for lag in lag_idx:
        if lag == 0:
            a, b = st, px
        else:
            a, b = st[:-lag], px[lag:]
        if len(a) < 5:
            lag_vals.append(0.0)
        else:
            cc = np.corrcoef(a, b)[0, 1]
            lag_vals.append(0.0 if np.isnan(cc) else float(cc))
    plt.plot(lag_idx, lag_vals, marker="o", color="#7c3aed")
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.title("Lag Correlation: Stock(t) -> Price(t+lag)")
    plt.xlabel("Lag (weeks)")
    plt.ylabel("Correlation")
    plt.tight_layout()
    plt.savefig(out_dir / "inventory_price_focus.png", dpi=160)
    plt.close()


def export_final_package(
    out_dir: Path,
    cfg: Config,
    X: np.ndarray,
    y: np.ndarray,
    y_dates: List[pd.Timestamp],
    feature_cols: List[str],
    best_params_list: List[Dict[str, float]],
):
    pkg_dir = out_dir / "final_package"
    pkg_dir.mkdir(parents=True, exist_ok=True)

    n = len(X)
    holdout_n = cfg.test_size
    train_all_end = n - holdout_n
    val_n = min(cfg.val_size, max(12, int(train_all_end * 0.15)))
    fit_end = train_all_end - val_n
    if fit_end <= 40:
        raise ValueError("Not enough samples for final package export.")

    X_fit = X[:fit_end]
    y_fit = y[:fit_end]
    X_val = X[fit_end:train_all_end]
    y_val = y[fit_end:train_all_end]
    X_hold = X[train_all_end:]
    y_hold = y[train_all_end:]
    d_hold = y_dates[train_all_end:]

    selected_names = feature_cols

    X_fit_s, X_val_s, X_hold_s, y_fit_s, y_val_s, y_hold_s, x_scaler, y_scaler = transform_by_train(
        X_fit, X_val, X_hold, y_fit, y_val, y_hold
    )

    if not best_params_list:
        raise ValueError("best_params_list is empty.")

    model_payloads = []
    hold_preds = []
    val_preds = []
    inv_losses = []
    for i, params in enumerate(best_params_list, start=1):
        res = train_one_trial(X_fit_s, y_fit_s, X_val_s, y_val_s, cfg, params)
        val_pred_s_i = res["predict_func"](X_val_s)
        val_pred_i = y_scaler.inverse_transform(val_pred_s_i)
        hold_pred_s_i = res["predict_func"](X_hold_s)
        hold_pred_i = y_scaler.inverse_transform(hold_pred_s_i)
        val_preds.append(val_pred_i)
        hold_preds.append(hold_pred_i)
        val_true_i = y_scaler.inverse_transform(y_val_s)
        mae_full = float(mean_absolute_error(val_true_i.reshape(-1), val_pred_i.reshape(-1)))
        tail_n = min(8, len(val_true_i))
        mae_tail = float(
            mean_absolute_error(val_true_i.reshape(-1)[-tail_n:], val_pred_i.reshape(-1)[-tail_n:])
        )
        dyn_loss = 0.7 * mae_full + 0.3 * mae_tail
        inv_losses.append(1.0 / (dyn_loss + 1e-8))
        model_payloads.append(
            {
                "idx": i,
                "params": params,
                "val_loss": float(res["val_loss"]),
                "val_mae_full": mae_full,
                "val_mae_tail": mae_tail,
                "history": res["history"],
                "model_state": res["model_state"],
            }
        )

    weights = np.array(inv_losses, dtype=float)
    weights = weights / np.sum(weights)
    val_pred_ens = np.zeros_like(val_preds[0])
    hold_pred_ens = np.zeros_like(hold_preds[0])
    for w, vp in zip(weights, val_preds):
        val_pred_ens += float(w) * vp
    for w, hp in zip(weights, hold_preds):
        hold_pred_ens += float(w) * hp
    hold_true = y_scaler.inverse_transform(y_hold_s)
    val_true = y_scaler.inverse_transform(y_val_s)

    # Train a recent-window specialist and blend with ensemble by validation MAE.
    recent_n = min(cfg.recent_tune_window, len(X_fit_s))
    recent_start = max(0, len(X_fit_s) - recent_n)
    X_recent = X_fit_s[recent_start:]
    y_recent = y_fit_s[recent_start:]
    recent_params = model_payloads[0]["params"]
    recent_res = train_one_trial(X_recent, y_recent, X_val_s, y_val_s, cfg, recent_params)
    val_pred_recent = y_scaler.inverse_transform(recent_res["predict_func"](X_val_s))
    hold_pred_recent = y_scaler.inverse_transform(recent_res["predict_func"](X_hold_s))

    best_alpha = 1.0
    best_val_mae = float("inf")
    val_blend_best = val_pred_ens
    hold_blend_best = hold_pred_ens
    for a in np.linspace(0.0, 1.0, 21):
        vb = a * val_pred_ens + (1.0 - a) * val_pred_recent
        mae = float(mean_absolute_error(val_true.reshape(-1), vb.reshape(-1)))
        if mae < best_val_mae:
            best_val_mae = mae
            best_alpha = float(a)
            val_blend_best = vb
            hold_blend_best = a * hold_pred_ens + (1.0 - a) * hold_pred_recent

    core_idx = {n: feature_cols.index(n) for n in ["WTI_T", "Stock_T", "Price_Change", "Stock_Change"]}
    Xv_last = X_val[:, -1, :]
    Xh_last = X_hold[:, -1, :]
    corr_X_val = np.column_stack(
        [
            val_blend_best.reshape(-1),
            Xv_last[:, core_idx["WTI_T"]],
            Xv_last[:, core_idx["Stock_T"]],
            Xv_last[:, core_idx["Price_Change"]],
            Xv_last[:, core_idx["Stock_Change"]],
        ]
    )
    corr_X_hold = np.column_stack(
        [
            hold_blend_best.reshape(-1),
            Xh_last[:, core_idx["WTI_T"]],
            Xh_last[:, core_idx["Stock_T"]],
            Xh_last[:, core_idx["Price_Change"]],
            Xh_last[:, core_idx["Stock_Change"]],
        ]
    )
    corr_model = Ridge(alpha=1.0, random_state=SEED)
    corr_model.fit(corr_X_val, val_true.reshape(-1))
    hold_pred_corr = corr_model.predict(corr_X_hold).reshape(-1, 1)
    hold_metrics = evaluate_predictions(hold_true, hold_pred_corr)

    pred_df = pd.DataFrame(
        {
            "Event_Date": d_hold,
            "y_true_price_change_t1": hold_true.reshape(-1),
            "y_pred_price_change_t1": hold_pred_corr.reshape(-1),
        }
    )
    pred_df.to_csv(pkg_dir / "final_holdout_predictions.csv", index=False, encoding="utf-8-sig")
    # Keep compatibility by saving the first model as final_lstm_model.pt.
    pd.DataFrame(model_payloads[0]["history"]).to_csv(pkg_dir / "final_train_history.csv", index=False, encoding="utf-8-sig")
    torch.save(model_payloads[0]["model_state"], pkg_dir / "final_lstm_model.pt")
    for mp in model_payloads:
        torch.save(mp["model_state"], pkg_dir / f"final_lstm_model_{mp['idx']}.pt")
    torch.save(recent_res["model_state"], pkg_dir / "final_recent_specialist_model.pt")
    joblib.dump(corr_model, pkg_dir / "final_error_corrector.pkl")
    joblib.dump(x_scaler, pkg_dir / "final_x_scaler.pkl")
    joblib.dump(y_scaler, pkg_dir / "final_y_scaler.pkl")
    with open(pkg_dir / "final_feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(selected_names, f, ensure_ascii=False, indent=2)

    ensemble_models = []
    for w, mp in zip(weights.tolist(), model_payloads):
        ensemble_models.append(
            {
                "idx": mp["idx"],
                "weight": float(w),
                "val_loss": mp["val_loss"],
                "val_mae_full": mp["val_mae_full"],
                "val_mae_tail": mp["val_mae_tail"],
                "params": mp["params"],
                "model_file": f"final_lstm_model_{mp['idx']}.pt",
            }
        )

    final_info = {
        "best_params": model_payloads[0]["params"],
        "ensemble_size": len(model_payloads),
        "ensemble_models": ensemble_models,
        "error_corrector": {
            "type": "ridge",
            "alpha": 1.0,
            "features": ["pred_raw", "WTI_T", "Stock_T", "Price_Change", "Stock_Change"],
            "model_file": "final_error_corrector.pkl",
        },
        "recent_specialist": {
            "enabled": True,
            "recent_window": int(recent_n),
            "blend_alpha_ensemble": float(best_alpha),
            "blend_alpha_recent": float(1.0 - best_alpha),
            "val_mae_after_blend": float(best_val_mae),
            "params": recent_params,
            "model_file": "final_recent_specialist_model.pt",
        },
        "fit_samples": int(len(X_fit)),
        "val_samples": int(len(X_val)),
        "holdout_samples": int(len(X_hold)),
        "holdout_metrics": hold_metrics,
    }
    with open(pkg_dir / "final_package_summary.json", "w", encoding="utf-8") as f:
        json.dump(final_info, f, ensure_ascii=False, indent=2)
    return final_info


def main():
    cfg = Config()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(cfg.data_path, encoding="utf-8-sig")
    weekly = build_weekly_table(raw, cfg)
    weekly.to_csv(out_dir / "weekly_preprocessed.csv", index=False, encoding="utf-8-sig")
    X, y, y_dates, feature_cols = build_sequences(weekly, cfg.lookback)
    n = len(X)
    splits = make_walkforward_splits(n, cfg)
    plot_process_flow(out_dir)
    plot_splits(out_dir, splits, n)

    round_trial_grids = [sample_trial_grid(cfg, r) for r in range(cfg.search_rounds)]
    trial_grid = [t for g in round_trial_grids for t in g]

    fold_rows = []
    all_trial_logs = []
    last_pred_df = None
    last_best_state = None
    last_scalers = None
    last_best_features = None

    for sp in splits:
        fold = sp["fold"]
        print(f"\n=== Fold {fold}/{cfg.n_folds} ===")
        X_train = X[sp["train_start"] : sp["train_end"]]
        y_train = y[sp["train_start"] : sp["train_end"]]
        X_val = X[sp["val_start"] : sp["val_end"]]
        y_val = y[sp["val_start"] : sp["val_end"]]
        X_test = X[sp["test_start"] : sp["test_end"]]
        y_test = y[sp["test_start"] : sp["test_end"]]
        d_test = y_dates[sp["test_start"] : sp["test_end"]]

        selected_names = feature_cols
        X_train_s, X_val_s, X_test_s, y_train_s, y_val_s, y_test_s, x_scaler, y_scaler = transform_by_train(
            X_train, X_val, X_test, y_train, y_val, y_test
        )

        best = None
        trial_logs = []
        trial_pred_pool = []
        for tid, trial in enumerate(trial_grid, start=1):
            print(f"Fold {fold} Trial {tid}/{len(trial_grid)} {trial}")
            res = train_one_trial(X_train_s, y_train_s, X_val_s, y_val_s, cfg, trial)
            val_pred = y_scaler.inverse_transform(res["pred_val_s"])
            val_true = y_scaler.inverse_transform(y_val_s)
            val_metrics = evaluate_predictions(val_true, val_pred)
            test_pred_tmp = y_scaler.inverse_transform(res["predict_func"](X_test_s))
            test_true_tmp = y_scaler.inverse_transform(y_test_s)
            test_metrics_tmp = evaluate_predictions(test_true_tmp, test_pred_tmp)
            score = val_metrics["price_change_t1_mae"]
            log = {
                "fold": fold,
                "trial_id": tid,
                "params": trial,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics_tmp,
                "selection_score": float(score),
            }
            trial_logs.append(log)
            all_trial_logs.append(log)
            trial_pred_pool.append(
                {
                    "trial_id": tid,
                    "score": float(score),
                    "test_pred": test_pred_tmp.reshape(-1),
                }
            )
            if best is None or score < best["selection_score"]:
                best = {"trial_id": tid, "result": res, "selection_score": float(score), "val_metrics": val_metrics}

        if best is None:
            raise RuntimeError("No best trial found.")

        top_preds = sorted(trial_pred_pool, key=lambda z: z["score"])[: max(1, min(cfg.ensemble_top_n, len(trial_pred_pool)))]
        ens_w = np.array([1.0 / (x["score"] + 1e-8) for x in top_preds], dtype=float)
        ens_w = ens_w / np.sum(ens_w)
        pred_test = np.zeros_like(top_preds[0]["test_pred"], dtype=float)
        for w, item in zip(ens_w, top_preds):
            pred_test += float(w) * item["test_pred"]
        pred_test = pred_test.reshape(-1, 1)
        y_test_real = y_scaler.inverse_transform(y_test_s)
        test_metrics = evaluate_predictions(y_test_real, pred_test)

        fold_rows.append(
            {
                "fold": fold,
                "train_n": len(X_train),
                "val_n": len(X_val),
                "test_n": len(X_test),
                "best_trial_id": best["trial_id"],
                "feature_count": len(selected_names),
                **test_metrics,
            }
        )

        pred_df = pd.DataFrame(
            {
                "Event_Date": d_test,
                "y_true_price_change_t1": y_test_real.reshape(-1),
                "y_pred_price_change_t1": pred_test.reshape(-1),
            }
        )
        pred_df.to_csv(out_dir / f"fold_{fold}_predictions.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(best["result"]["history"]).to_csv(out_dir / f"fold_{fold}_history.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(trial_logs).to_json(out_dir / f"fold_{fold}_trials.json", orient="records", indent=2, force_ascii=False)

        if fold == cfg.n_folds:
            last_pred_df = pred_df
            last_best_state = best["result"]["model_state"]
            last_scalers = (x_scaler, y_scaler)
            last_best_features = selected_names

    folds_df = pd.DataFrame(fold_rows)
    folds_df.to_csv(out_dir / "walkforward_fold_metrics.csv", index=False, encoding="utf-8-sig")
    plot_fold_metrics(out_dir, folds_df)
    plot_training_overview(out_dir, cfg.n_folds)
    plot_inventory_price_focus(out_dir, weekly)
    if last_pred_df is not None:
        plot_last_fold_predictions(out_dir, last_pred_df)
        plot_actual_vs_pred_scatter(out_dir, last_pred_df)
        plot_residual_distribution(out_dir, last_pred_df)
        plot_residual_timeseries(out_dir, last_pred_df)

    trial_rank = (
        pd.DataFrame(
            [
                {
                    "fold": x["fold"],
                    "trial_id": x["trial_id"],
                    "params": json.dumps(x["params"], ensure_ascii=False, sort_keys=True),
                    "selection_score": x["selection_score"],
                    "val_price_mae": x["val_metrics"]["price_change_t1_mae"],
                    "test_price_mae": x["test_metrics"]["price_change_t1_mae"],
                }
                for x in all_trial_logs
            ]
        )
        .groupby(["trial_id", "params"], as_index=False)
        .agg(
            mean_selection_score=("selection_score", "mean"),
            std_selection_score=("selection_score", "std"),
            mean_val_price_mae=("val_price_mae", "mean"),
            mean_test_price_mae=("test_price_mae", "mean"),
        )
        .sort_values("mean_selection_score")
        .reset_index(drop=True)
    )
    trial_rank.to_csv(out_dir / "global_trial_ranking.csv", index=False, encoding="utf-8-sig")
    best_global = trial_rank.iloc[0].to_dict()
    topn = max(1, min(cfg.ensemble_top_n, len(trial_rank)))
    top_global = trial_rank.head(topn).to_dict(orient="records")
    best_global_params_list = [json.loads(x["params"]) for x in top_global]

    final_pkg_info = export_final_package(
        out_dir=out_dir,
        cfg=cfg,
        X=X,
        y=y,
        y_dates=y_dates,
        feature_cols=feature_cols,
        best_params_list=best_global_params_list,
    )

    summary = {
        "n_sequences": int(n),
        "n_folds": cfg.n_folds,
        "val_size": cfg.val_size,
        "test_size": cfg.test_size,
        "fold_metrics_mean": folds_df[
            [
                "price_change_t1_mae",
                "price_change_t1_rmse",
                "price_change_t1_r2",
                "price_change_t1_direction_acc",
            ]
        ].mean().to_dict(),
        "fold_metrics_std": folds_df[
            [
                "price_change_t1_mae",
                "price_change_t1_rmse",
                "price_change_t1_r2",
                "price_change_t1_direction_acc",
            ]
        ].std().to_dict(),
        "search_rounds": cfg.search_rounds,
        "trials_per_round": cfg.trials_per_round,
        "trial_count": len(trial_grid),
        "ensemble_top_n": cfg.ensemble_top_n,
        "global_best_trial": best_global,
        "global_top_trials": top_global,
        "final_package": final_pkg_info,
        "process_plots": [
            "process_flow.png",
            "walkforward_splits.png",
            "fold_metrics.png",
            "training_overview.png",
            "inventory_price_focus.png",
            "last_fold_predictions.png",
            "actual_vs_pred_scatter.png",
            "residual_distribution.png",
            "residual_timeseries.png",
        ],
    }
    with open(out_dir / "walkforward_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if last_best_state is not None and last_scalers is not None and last_best_features is not None:
        torch.save(last_best_state, out_dir / "last_fold_lstm_model.pt")
        joblib.dump(last_scalers[0], out_dir / "last_fold_x_scaler.pkl")
        joblib.dump(last_scalers[1], out_dir / "last_fold_y_scaler.pkl")
        with open(out_dir / "last_fold_feature_columns.json", "w", encoding="utf-8") as f:
            json.dump(last_best_features, f, ensure_ascii=False, indent=2)
    print("\nWalk-forward complete.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nArtifacts saved to: {out_dir}")


if __name__ == "__main__":
    main()
