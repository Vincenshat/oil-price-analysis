import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from train_lstm_pipeline import Config, LSTMRegressor, build_sequences, build_weekly_table, main as train_pipeline_main
from generate_plots import main as generate_plots_main


def run_train():
    train_pipeline_main()


def run_predict(data_path: str, package_dir: str):
    pkg = Path(package_dir)
    if not pkg.exists():
        raise FileNotFoundError(f"Package directory not found: {pkg}")

    summary_path = pkg / "final_package_summary.json"
    features_path = pkg / "final_feature_columns.json"
    model_path = pkg / "final_lstm_model.pt"
    x_scaler_path = pkg / "final_x_scaler.pkl"
    y_scaler_path = pkg / "final_y_scaler.pkl"
    err_corr_path = pkg / "final_error_corrector.pkl"

    for p in [summary_path, features_path, model_path, x_scaler_path, y_scaler_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required file missing: {p}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    best_params = summary["best_params"]
    ensemble_models = summary.get("ensemble_models", [])
    recent_specialist = summary.get("recent_specialist", {})
    selected_features = json.loads(features_path.read_text(encoding="utf-8"))

    cfg = Config(data_path=data_path)
    raw = pd.read_csv(data_path, encoding="utf-8-sig")
    weekly = build_weekly_table(raw, cfg)
    X, _, y_dates, feature_cols = build_sequences(weekly, cfg.lookback)
    if len(X) == 0:
        raise ValueError("No sequence sample available for prediction.")

    missing = [c for c in selected_features if c not in feature_cols]
    if missing:
        raise ValueError(f"Selected features not found in current data: {missing}")

    idx = [feature_cols.index(c) for c in selected_features]
    x_last = X[-1:, :, idx]

    x_scaler = joblib.load(x_scaler_path)
    y_scaler = joblib.load(y_scaler_path)
    shp = x_last.shape
    x_last_s = x_scaler.transform(x_last.reshape(-1, shp[-1])).reshape(shp).astype(np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if ensemble_models:
        pred_components = []
        weight_sum = 0.0
        for em in ensemble_models:
            w = float(em.get("weight", 0.0))
            if w <= 0:
                continue
            params = em["params"]
            mfile = em.get("model_file", "")
            mpath = pkg / mfile if mfile else model_path
            model = LSTMRegressor(
                input_size=len(selected_features),
                hidden_size=int(params["hidden_size"]),
                num_layers=int(params["num_layers"]),
                dropout=float(params["dropout"]),
                model_type=str(params.get("model_type", "lstm")),
            ).to(device)
            state = torch.load(mpath, map_location=device)
            model.load_state_dict(state)
            model.eval()
            with torch.no_grad():
                pred_s_i = model(torch.tensor(x_last_s, dtype=torch.float32, device=device)).cpu().numpy()
            pred_components.append((w, y_scaler.inverse_transform(pred_s_i).reshape(-1)[0]))
            weight_sum += w
        if not pred_components:
            raise ValueError("No valid ensemble models found in package summary.")
        pred = float(sum(w * p for w, p in pred_components) / max(weight_sum, 1e-12))
    else:
        model = LSTMRegressor(
            input_size=len(selected_features),
            hidden_size=int(best_params["hidden_size"]),
            num_layers=int(best_params["num_layers"]),
            dropout=float(best_params["dropout"]),
            model_type=str(best_params.get("model_type", "lstm")),
        ).to(device)
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            pred_s = model(torch.tensor(x_last_s, dtype=torch.float32, device=device)).cpu().numpy()
        pred = y_scaler.inverse_transform(pred_s).reshape(-1)[0]

    latest_row = weekly.iloc[-1]
    latest_wti = float(latest_row["WTI_T"])
    latest_stock = float(latest_row["Stock_T"])
    latest_stock_change = float(latest_row["Stock_Change"])
    latest_price_change = float(latest_row["Price_Change"])

    if recent_specialist.get("enabled", False):
        rsp = recent_specialist.get("params", {})
        recent_model_file = recent_specialist.get("model_file", "final_recent_specialist_model.pt")
        recent_path = pkg / recent_model_file
        if recent_path.exists() and rsp:
            recent_model = LSTMRegressor(
                input_size=len(selected_features),
                hidden_size=int(rsp["hidden_size"]),
                num_layers=int(rsp["num_layers"]),
                dropout=float(rsp["dropout"]),
                model_type=str(rsp.get("model_type", "lstm")),
            ).to(device)
            recent_state = torch.load(recent_path, map_location=device)
            recent_model.load_state_dict(recent_state)
            recent_model.eval()
            with torch.no_grad():
                pred_recent_s = recent_model(torch.tensor(x_last_s, dtype=torch.float32, device=device)).cpu().numpy()
            pred_recent = float(y_scaler.inverse_transform(pred_recent_s).reshape(-1)[0])
            a_ens = float(recent_specialist.get("blend_alpha_ensemble", 1.0))
            a_rec = float(recent_specialist.get("blend_alpha_recent", 0.0))
            pred = a_ens * pred + a_rec * pred_recent

    if err_corr_path.exists():
        corr_model = joblib.load(err_corr_path)
        corr_input = np.array([[pred, latest_wti, latest_stock, latest_price_change, latest_stock_change]], dtype=np.float64)
        pred = float(corr_model.predict(corr_input).reshape(-1)[0])

    out = {
        "latest_sequence_date": str(y_dates[-1]),
        "latest_wti_price": latest_wti,
        "latest_stock_barrels": latest_stock,
        "latest_stock_change": latest_stock_change,
        "latest_price_change": latest_price_change,
        "pred_price_change_t1": float(pred),
        "pred_wti_price_t1": float(latest_wti + pred),
        "model_package": str(pkg),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))

    pred_path = pkg / "predict_next_output.json"
    pred_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved prediction: {pred_path}")


def run_plots():
    generate_plots_main()


def main():
    parser = argparse.ArgumentParser(description="Oil Analysis ML/DL pipeline entrypoint.")
    sub = parser.add_subparsers(dest="command", required=True)

    train_cmd = sub.add_parser("train", help="Run walk-forward training/tuning pipeline.")
    train_cmd.set_defaults(func=lambda args: run_train())

    pred_cmd = sub.add_parser("predict", help="Run one-step prediction with exported final package.")
    pred_cmd.add_argument("--data", default=r"d:\Projects\oil_analysis\ML\4.0_enriched.csv", help="Input enriched CSV path.")
    pred_cmd.add_argument(
        "--package-dir",
        default=r"d:\Projects\oil_analysis\ML\outputs_lstm_wf\final_package",
        help="Final package directory path.",
    )
    pred_cmd.set_defaults(func=lambda args: run_predict(args.data, args.package_dir))

    plot_cmd = sub.add_parser("plots", help="Generate training/final/descriptive plots.")
    plot_cmd.set_defaults(func=lambda args: run_plots())

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
