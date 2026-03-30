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


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    out = {}
    names = ["price_change_t1", "stock_change_t1"]
    for i, n in enumerate(names):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        out[f"{n}_mae"] = float(mean_absolute_error(yt, yp))
        out[f"{n}_rmse"] = float(np.sqrt(mean_squared_error(yt, yp)))
        out[f"{n}_r2"] = float(r2_score(yt, yp))
        # Directional hit rate is useful when absolute magnitude is noisy.
        out[f"{n}_direction_acc"] = float(np.mean(np.sign(yt) == np.sign(yp)))
        out[f"{n}_mean_error"] = float(np.mean(yp - yt))
    return out


def build_weekly_table(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()
    df["Event_Date"] = pd.to_datetime(df["Event_Date"], errors="coerce")
    df = df.dropna(subset=["Event_Date"]).sort_values("Event_Date")
    df = df[df["WTI_T"].notna() & df["Stock_T"].notna() & df["Price_Change"].notna() & df["Stock_Change"].notna()]
    df = df[df["Price_Change"].abs() <= cfg.clip_price_change_abs]

    macro_cols = [
        "dxy_broad",
        "us10y_yield",
        "fed_funds_rate",
        "cpi_aucsl",
        "refinery_utilization_pct",
        "oil_gas_extraction_ip",
        "drilling_activity_ip",
    ]
    base_num_cols = ["WTI_T", "Stock_T", "Price_Level", "Price_Change", "Stock_Change"] + macro_cols
    for col in base_num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    evt = pd.crosstab(df["Event_Date"], df["Event_Type"]).reset_index().rename_axis(None, axis=1)
    evt.columns = ["Event_Date"] + [f"evt_cnt_{c}" for c in evt.columns[1:]]

    weekly = (
        df.groupby("Event_Date", as_index=False)
        .agg(
            {
                "WTI_T": "mean",
                "Stock_T": "mean",
                "Price_Level": "mean",
                "Price_Change": "mean",
                "Stock_Change": "mean",
                "dxy_broad": "mean",
                "us10y_yield": "mean",
                "fed_funds_rate": "mean",
                "cpi_aucsl": "mean",
                "refinery_utilization_pct": "mean",
                "oil_gas_extraction_ip": "mean",
                "drilling_activity_ip": "mean",
            }
        )
        .sort_values("Event_Date")
    )
    weekly = weekly.merge(evt, on="Event_Date", how="left")

    full_idx = pd.date_range(weekly["Event_Date"].min(), weekly["Event_Date"].max(), freq="W-FRI")
    weekly = weekly.set_index("Event_Date").reindex(full_idx).rename_axis("Event_Date").reset_index()

    evt_cols = [c for c in weekly.columns if c.startswith("evt_cnt_")]
    for c in evt_cols:
        weekly[c] = weekly[c].fillna(0.0)

    num_cols = [c for c in weekly.columns if c != "Event_Date"]
    # Leakage-safe fill: only forward fill, never backward fill from future.
    weekly[num_cols] = weekly[num_cols].ffill()

    weekly["week_of_year"] = weekly["Event_Date"].dt.isocalendar().week.astype(float)
    weekly["month"] = weekly["Event_Date"].dt.month.astype(float)
    weekly["week_sin"] = np.sin(2 * np.pi * weekly["week_of_year"] / 52.0)
    weekly["week_cos"] = np.cos(2 * np.pi * weekly["week_of_year"] / 52.0)
    weekly["month_sin"] = np.sin(2 * np.pi * weekly["month"] / 12.0)
    weekly["month_cos"] = np.cos(2 * np.pi * weekly["month"] / 12.0)

    lag_cols = ["WTI_T", "Stock_T", "Price_Change", "Stock_Change", "dxy_broad", "us10y_yield", "fed_funds_rate"]
    for col in lag_cols:
        for lag in [1, 2, 4]:
            weekly[f"{col}_lag{lag}"] = weekly[col].shift(lag)

    for col in ["WTI_T", "Stock_T", "Price_Change", "Stock_Change"]:
        weekly[f"{col}_roll3_mean"] = weekly[col].rolling(3, min_periods=2).mean()
        weekly[f"{col}_roll6_mean"] = weekly[col].rolling(6, min_periods=3).mean()
        weekly[f"{col}_roll6_std"] = weekly[col].rolling(6, min_periods=3).std()

    chg_cols = [
        "WTI_T",
        "Stock_T",
        "dxy_broad",
        "us10y_yield",
        "fed_funds_rate",
        "refinery_utilization_pct",
        "oil_gas_extraction_ip",
        "drilling_activity_ip",
    ]
    for col in chg_cols:
        weekly[f"{col}_diff1"] = weekly[col].diff(1)
        weekly[f"{col}_pct1"] = weekly[col].pct_change(1)

    weekly = weekly.replace([np.inf, -np.inf], np.nan)
    num_cols = [c for c in weekly.columns if c != "Event_Date"]
    # Keep preprocessing causal for time-series evaluation.
    weekly[num_cols] = weekly[num_cols].ffill()
    # For engineered columns that are undefined at the beginning, use 0 instead of future info.
    weekly[num_cols] = weekly[num_cols].fillna(0.0)

    weekly["target_price_change_t1"] = weekly["Price_Change"].shift(-1)
    weekly["target_stock_change_t1"] = weekly["Stock_Change"].shift(-1)
    weekly = weekly.dropna(subset=["target_price_change_t1", "target_stock_change_t1"]).reset_index(drop=True)
    return weekly


def build_sequences(weekly: pd.DataFrame, lookback: int) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp], List[str]]:
    feature_cols = [c for c in weekly.columns if c not in ["Event_Date", "target_price_change_t1", "target_stock_change_t1"]]
    Xv = weekly[feature_cols].values.astype(np.float32)
    yv = weekly[["target_price_change_t1", "target_stock_change_t1"]].values.astype(np.float32)
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
        c2 = np.corrcoef(x_last[:, i], y_train[:, 1])[0, 1]
        val = np.nanmax([abs(c1), abs(c2)])
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
    model = LSTMRegressor(X_train_s.shape[-1], int(trial["hidden_size"]), int(trial["num_layers"]), float(trial["dropout"])).to(device)
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
        loader = DataLoader(SequenceDataset(x_arr, np.zeros((len(x_arr), 2), dtype=np.float32)), batch_size=64, shuffle=False)
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


def plot_process_flow(out_dir: Path):
    plt.figure(figsize=(12, 2.8))
    plt.axis("off")
    steps = ["Raw Table", "Interpolation + FE", "Sequence Build", "Walk-Forward Folds", "LSTM Tuning", "Fold Metrics + Plots"]
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
    plt.figure(figsize=(11, 4))
    plt.subplot(1, 2, 1)
    plt.plot(x, folds_df["price_change_t1_mae"], marker="o", label="price MAE")
    plt.plot(x, folds_df["stock_change_t1_mae"], marker="o", label="stock MAE")
    plt.title("MAE by Fold")
    plt.xlabel("Fold")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, folds_df["price_change_t1_r2"], marker="o", label="price R2")
    plt.plot(x, folds_df["stock_change_t1_r2"], marker="o", label="stock R2")
    plt.title("R2 by Fold")
    plt.xlabel("Fold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fold_metrics.png", dpi=160)
    plt.close()


def plot_last_fold_predictions(out_dir: Path, pred_df: pd.DataFrame):
    plt.figure(figsize=(11, 4))
    plt.subplot(1, 2, 1)
    plt.plot(pred_df["Event_Date"], pred_df["y_true_price_change_t1"], label="true")
    plt.plot(pred_df["Event_Date"], pred_df["y_pred_price_change_t1"], label="pred")
    plt.title("Last Fold PriceChange t+1")
    plt.xticks(rotation=35)
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(pred_df["Event_Date"], pred_df["y_true_stock_change_t1"], label="true")
    plt.plot(pred_df["Event_Date"], pred_df["y_pred_stock_change_t1"], label="pred")
    plt.title("Last Fold StockChange t+1")
    plt.xticks(rotation=35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "last_fold_predictions.png", dpi=160)
    plt.close()


def plot_feature_frequency(out_dir: Path, feature_freq: pd.DataFrame):
    top = feature_freq.head(20).iloc[::-1]
    plt.figure(figsize=(9, 6))
    plt.barh(top["feature"], top["selected_in_folds"], color="#3b82f6")
    plt.title("Top20 Feature Selection Frequency")
    plt.xlabel("Selected In Folds")
    plt.tight_layout()
    plt.savefig(out_dir / "feature_selection_top20.png", dpi=160)
    plt.close()


def plot_residual_distribution(out_dir: Path, pred_df: pd.DataFrame):
    rp = pred_df["y_true_price_change_t1"] - pred_df["y_pred_price_change_t1"]
    rs = pred_df["y_true_stock_change_t1"] - pred_df["y_pred_stock_change_t1"]
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(rp, bins=15, color="#60a5fa", edgecolor="white")
    plt.title("Residual Distribution: PriceChange")
    plt.subplot(1, 2, 2)
    plt.hist(rs, bins=15, color="#34d399", edgecolor="white")
    plt.title("Residual Distribution: StockChange")
    plt.tight_layout()
    plt.savefig(out_dir / "residual_distribution.png", dpi=160)
    plt.close()


def export_final_package(
    out_dir: Path,
    cfg: Config,
    X: np.ndarray,
    y: np.ndarray,
    y_dates: List[pd.Timestamp],
    feature_cols: List[str],
    best_params: Dict[str, float],
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

    selected_idx = select_feature_indices(X_fit, y_fit, cfg.top_k_features, cfg.feature_corr_threshold)
    selected_names = [feature_cols[i] for i in selected_idx]
    X_fit = X_fit[:, :, selected_idx]
    X_val = X_val[:, :, selected_idx]
    X_hold = X_hold[:, :, selected_idx]

    X_fit_s, X_val_s, X_hold_s, y_fit_s, y_val_s, y_hold_s, x_scaler, y_scaler = transform_by_train(
        X_fit, X_val, X_hold, y_fit, y_val, y_hold
    )

    res = train_one_trial(X_fit_s, y_fit_s, X_val_s, y_val_s, cfg, best_params)
    hold_pred_s = res["predict_func"](X_hold_s)
    hold_pred = y_scaler.inverse_transform(hold_pred_s)
    hold_true = y_scaler.inverse_transform(y_hold_s)
    hold_metrics = evaluate_predictions(hold_true, hold_pred)

    pred_df = pd.DataFrame(
        {
            "Event_Date": d_hold,
            "y_true_price_change_t1": hold_true[:, 0],
            "y_pred_price_change_t1": hold_pred[:, 0],
            "y_true_stock_change_t1": hold_true[:, 1],
            "y_pred_stock_change_t1": hold_pred[:, 1],
        }
    )
    pred_df.to_csv(pkg_dir / "final_holdout_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(res["history"]).to_csv(pkg_dir / "final_train_history.csv", index=False, encoding="utf-8-sig")
    torch.save(res["model_state"], pkg_dir / "final_lstm_model.pt")
    joblib.dump(x_scaler, pkg_dir / "final_x_scaler.pkl")
    joblib.dump(y_scaler, pkg_dir / "final_y_scaler.pkl")
    with open(pkg_dir / "final_feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(selected_names, f, ensure_ascii=False, indent=2)

    final_info = {
        "best_params": best_params,
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

    trial_grid = [
        {"hidden_size": 64, "num_layers": 2, "dropout": 0.15, "learning_rate": 1.2e-3, "weight_decay": 1e-4, "batch_size": 32},
        {"hidden_size": 64, "num_layers": 2, "dropout": 0.25, "learning_rate": 8e-4, "weight_decay": 2e-4, "batch_size": 32},
        {"hidden_size": 64, "num_layers": 3, "dropout": 0.2, "learning_rate": 8e-4, "weight_decay": 2e-4, "batch_size": 24},
        {"hidden_size": 96, "num_layers": 2, "dropout": 0.2, "learning_rate": 1e-3, "weight_decay": 1e-4, "batch_size": 32},
        {"hidden_size": 96, "num_layers": 2, "dropout": 0.25, "learning_rate": 8e-4, "weight_decay": 2e-4, "batch_size": 32},
        {"hidden_size": 96, "num_layers": 3, "dropout": 0.25, "learning_rate": 7e-4, "weight_decay": 2e-4, "batch_size": 24},
        {"hidden_size": 96, "num_layers": 3, "dropout": 0.3, "learning_rate": 6e-4, "weight_decay": 3e-4, "batch_size": 24},
        {"hidden_size": 128, "num_layers": 2, "dropout": 0.25, "learning_rate": 8e-4, "weight_decay": 1e-4, "batch_size": 24},
        {"hidden_size": 128, "num_layers": 2, "dropout": 0.3, "learning_rate": 6e-4, "weight_decay": 2e-4, "batch_size": 24},
        {"hidden_size": 128, "num_layers": 3, "dropout": 0.3, "learning_rate": 5e-4, "weight_decay": 3e-4, "batch_size": 20},
    ]

    fold_rows = []
    all_trial_logs = []
    feature_counter: Dict[str, int] = {}
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

        selected_idx = select_feature_indices(X_train, y_train, cfg.top_k_features, cfg.feature_corr_threshold)
        selected_names = [feature_cols[i] for i in selected_idx]
        for nname in selected_names:
            feature_counter[nname] = feature_counter.get(nname, 0) + 1

        X_train = X_train[:, :, selected_idx]
        X_val = X_val[:, :, selected_idx]
        X_test = X_test[:, :, selected_idx]
        X_train_s, X_val_s, X_test_s, y_train_s, y_val_s, y_test_s, x_scaler, y_scaler = transform_by_train(
            X_train, X_val, X_test, y_train, y_val, y_test
        )

        best = None
        trial_logs = []
        for tid, trial in enumerate(trial_grid, start=1):
            print(f"Fold {fold} Trial {tid}/{len(trial_grid)} {trial}")
            res = train_one_trial(X_train_s, y_train_s, X_val_s, y_val_s, cfg, trial)
            val_pred = y_scaler.inverse_transform(res["pred_val_s"])
            val_true = y_scaler.inverse_transform(y_val_s)
            val_metrics = evaluate_predictions(val_true, val_pred)
            test_pred_tmp = y_scaler.inverse_transform(res["predict_func"](X_test_s))
            test_true_tmp = y_scaler.inverse_transform(y_test_s)
            test_metrics_tmp = evaluate_predictions(test_true_tmp, test_pred_tmp)
            score = val_metrics["price_change_t1_mae"] + val_metrics["stock_change_t1_mae"] / 3000.0
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
            if best is None or score < best["selection_score"]:
                best = {"trial_id": tid, "result": res, "selection_score": float(score), "val_metrics": val_metrics}

        if best is None:
            raise RuntimeError("No best trial found.")

        pred_test_s = best["result"]["predict_func"](X_test_s)
        pred_test = y_scaler.inverse_transform(pred_test_s)
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
                "y_true_price_change_t1": y_test_real[:, 0],
                "y_pred_price_change_t1": pred_test[:, 0],
                "y_true_stock_change_t1": y_test_real[:, 1],
                "y_pred_stock_change_t1": pred_test[:, 1],
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
    if last_pred_df is not None:
        plot_last_fold_predictions(out_dir, last_pred_df)

    feature_freq = pd.DataFrame(
        [{"feature": k, "selected_in_folds": v} for k, v in sorted(feature_counter.items(), key=lambda x: (-x[1], x[0]))]
    )
    feature_freq.to_csv(out_dir / "feature_selection_frequency.csv", index=False, encoding="utf-8-sig")
    plot_feature_frequency(out_dir, feature_freq)

    trial_rank = (
        pd.DataFrame(
            [
                {
                    "fold": x["fold"],
                    "trial_id": x["trial_id"],
                    "params": json.dumps(x["params"], ensure_ascii=False, sort_keys=True),
                    "selection_score": x["selection_score"],
                    "val_price_mae": x["val_metrics"]["price_change_t1_mae"],
                    "val_stock_mae": x["val_metrics"]["stock_change_t1_mae"],
                    "test_price_mae": x["test_metrics"]["price_change_t1_mae"],
                    "test_stock_mae": x["test_metrics"]["stock_change_t1_mae"],
                }
                for x in all_trial_logs
            ]
        )
        .groupby(["trial_id", "params"], as_index=False)
        .agg(
            mean_selection_score=("selection_score", "mean"),
            std_selection_score=("selection_score", "std"),
            mean_val_price_mae=("val_price_mae", "mean"),
            mean_val_stock_mae=("val_stock_mae", "mean"),
            mean_test_price_mae=("test_price_mae", "mean"),
            mean_test_stock_mae=("test_stock_mae", "mean"),
        )
        .sort_values("mean_selection_score")
        .reset_index(drop=True)
    )
    trial_rank.to_csv(out_dir / "global_trial_ranking.csv", index=False, encoding="utf-8-sig")
    best_global = trial_rank.iloc[0].to_dict()
    best_global_params = json.loads(best_global["params"])

    final_pkg_info = export_final_package(
        out_dir=out_dir,
        cfg=cfg,
        X=X,
        y=y,
        y_dates=y_dates,
        feature_cols=feature_cols,
        best_params=best_global_params,
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
                "stock_change_t1_mae",
                "stock_change_t1_rmse",
                "stock_change_t1_r2",
                "stock_change_t1_direction_acc",
            ]
        ].mean().to_dict(),
        "fold_metrics_std": folds_df[
            [
                "price_change_t1_mae",
                "price_change_t1_rmse",
                "price_change_t1_r2",
                "price_change_t1_direction_acc",
                "stock_change_t1_mae",
                "stock_change_t1_rmse",
                "stock_change_t1_r2",
                "stock_change_t1_direction_acc",
            ]
        ].std().to_dict(),
        "trial_count": len(trial_grid),
        "global_best_trial": best_global,
        "final_package": final_pkg_info,
        "process_plots": [
            "process_flow.png",
            "walkforward_splits.png",
            "fold_metrics.png",
            "last_fold_predictions.png",
            "feature_selection_top20.png",
            "residual_distribution.png",
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
    if last_pred_df is not None:
        plot_residual_distribution(out_dir, last_pred_df)

    print("\nWalk-forward complete.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nArtifacts saved to: {out_dir}")


if __name__ == "__main__":
    main()
