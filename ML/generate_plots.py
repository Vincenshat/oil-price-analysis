from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_training_curve(output_dir: Path):
    hist_files = sorted(output_dir.glob("fold_*_history.csv"))
    if not hist_files:
        return
    plt.figure(figsize=(10, 5))
    for f in hist_files:
        fold = f.stem.split("_")[1]
        df = pd.read_csv(f, encoding="utf-8-sig")
        plt.plot(df["epoch"], df["train_loss"], alpha=0.4, label=f"fold{fold} train")
        plt.plot(df["epoch"], df["val_loss"], alpha=0.8, linestyle="--", label=f"fold{fold} val")
    plt.title("Training Process: Loss Curves by Fold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "training_process_curve.png", dpi=170)
    plt.close()


def plot_final_result(output_dir: Path):
    p = output_dir / "final_package" / "final_holdout_predictions.csv"
    if not p.exists():
        return
    df = pd.read_csv(p, encoding="utf-8-sig")
    df["Event_Date"] = pd.to_datetime(df["Event_Date"], errors="coerce")
    plt.figure(figsize=(11, 4))
    plt.subplot(1, 2, 1)
    plt.plot(df["Event_Date"], df["y_true_price_change_t1"], label="true")
    plt.plot(df["Event_Date"], df["y_pred_price_change_t1"], label="pred")
    plt.title("Final Holdout: Price Change")
    plt.xticks(rotation=30)
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(df["Event_Date"], df["y_true_stock_change_t1"], label="true")
    plt.plot(df["Event_Date"], df["y_pred_stock_change_t1"], label="pred")
    plt.title("Final Holdout: Stock Change")
    plt.xticks(rotation=30)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "final_result_holdout.png", dpi=170)
    plt.close()


def plot_descriptive(data_path: Path, output_dir: Path):
    df = pd.read_csv(data_path, encoding="utf-8-sig")
    df["Event_Date"] = pd.to_datetime(df["Event_Date"], errors="coerce")
    num_cols = [
        "WTI_T",
        "Stock_T",
        "Price_Change",
        "Stock_Change",
        "dxy_broad",
        "us10y_yield",
        "fed_funds_rate",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    d = df.dropna(subset=["Event_Date"]).sort_values("Event_Date")

    # A compact descriptive dashboard
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    s = d.groupby("Event_Date", as_index=False)["Price_Change"].mean()
    plt.plot(s["Event_Date"], s["Price_Change"])
    plt.title("Descriptive: Price Change Over Time")
    plt.xticks(rotation=30)

    plt.subplot(2, 2, 2)
    s2 = d.groupby("Event_Date", as_index=False)["Stock_Change"].mean()
    plt.plot(s2["Event_Date"], s2["Stock_Change"])
    plt.title("Descriptive: Stock Change Over Time")
    plt.xticks(rotation=30)

    plt.subplot(2, 2, 3)
    plt.hist(d["Price_Change"].dropna(), bins=25, alpha=0.8)
    plt.title("Descriptive: Price Change Distribution")

    plt.subplot(2, 2, 4)
    corr_cols = [c for c in ["WTI_T", "Stock_T", "Price_Change", "Stock_Change", "dxy_broad", "us10y_yield", "fed_funds_rate"] if c in d.columns]
    corr = d[corr_cols].corr().fillna(0)
    im = plt.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    plt.xticks(range(len(corr_cols)), corr_cols, rotation=45, ha="right", fontsize=8)
    plt.yticks(range(len(corr_cols)), corr_cols, fontsize=8)
    plt.title("Descriptive: Correlation Heatmap")
    plt.colorbar(im, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_dir / "descriptive_dashboard.png", dpi=170)
    plt.close()


def main():
    base_dir = Path(r"d:\Projects\oil_analysis\ML")
    data_path = base_dir / "4.0_enriched.csv"
    output_dir = base_dir / "outputs_lstm_wf"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_training_curve(output_dir)
    plot_final_result(output_dir)
    plot_descriptive(data_path, output_dir)
    print("Generated:")
    print(output_dir / "training_process_curve.png")
    print(output_dir / "final_result_holdout.png")
    print(output_dir / "descriptive_dashboard.png")


if __name__ == "__main__":
    main()
