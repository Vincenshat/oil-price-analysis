from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
ML_DIR = BASE_DIR / "ML"
OUT_DIR = ML_DIR / "outputs_lstm_wf"
PKG_DIR = OUT_DIR / "final_package"


def tr(lang: str, zh: str, en: str) -> str:
    return zh if lang == "中文" else en


@st.cache_data(show_spinner=False)
def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def run_cmd(args: list[str]) -> tuple[bool, str]:
    res = subprocess.run(args, capture_output=True, text=True)
    if res.returncode != 0:
        return False, (res.stderr or res.stdout).strip()
    return True, "ok"


def run_predict() -> tuple[bool, str]:
    return run_cmd([sys.executable, str(ML_DIR / "main.py"), "predict"])


def run_plots() -> tuple[bool, str]:
    return run_cmd([sys.executable, str(ML_DIR / "main.py"), "plots"])


def run_train() -> tuple[bool, str]:
    return run_cmd([sys.executable, str(ML_DIR / "main.py"), "train"])


def show_top_metrics(lang: str, summary: dict, pred: dict) -> None:
    fold_mean = summary.get("fold_metrics_mean", {})
    hold = summary.get("final_package", {}).get("holdout_metrics", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(tr(lang, "回测MAE(均值)", "Walk-forward MAE"), f"{fold_mean.get('price_change_t1_mae', np.nan):.3f}")
    c2.metric(tr(lang, "最终Holdout MAE", "Final Holdout MAE"), f"{hold.get('price_change_t1_mae', np.nan):.3f}")
    c3.metric(tr(lang, "方向准确率", "Direction Accuracy"), f"{hold.get('price_change_t1_direction_acc', np.nan) * 100:.1f}%")
    c4.metric(tr(lang, "下一期价格变化预测", "Predicted Price Change"), f"{pred.get('pred_price_change_t1', np.nan):.3f}")


def show_prediction_card(lang: str, pred: dict) -> None:
    if not pred:
        st.warning(tr(lang, "暂无预测结果。", "No prediction output found."))
        return
    st.markdown(f"### {tr(lang, '当前输入与预测输出', 'Current Inputs and Forecast')}")
    st.write(
        tr(
            lang,
            (
                f"- 当前油价 `WTI_T`: **{pred.get('latest_wti_price', np.nan):.3f}**\n"
                f"- 当前库存 `Stock_T`: **{pred.get('latest_stock_barrels', np.nan):,.0f}**\n"
                f"- 当前库存变化 `Stock_Change`: **{pred.get('latest_stock_change', np.nan):,.0f}**\n"
                f"- 当前价格变化 `Price_Change`: **{pred.get('latest_price_change', np.nan):.3f}**\n"
                f"- 预测下一期价格变化: **{pred.get('pred_price_change_t1', np.nan):.3f}**\n"
                f"- 推导下一期油价: **{pred.get('pred_wti_price_t1', np.nan):.3f}**"
            ),
            (
                f"- Current WTI: **{pred.get('latest_wti_price', np.nan):.3f}**\n"
                f"- Current Stock: **{pred.get('latest_stock_barrels', np.nan):,.0f}**\n"
                f"- Current Stock Change: **{pred.get('latest_stock_change', np.nan):,.0f}**\n"
                f"- Current Price Change: **{pred.get('latest_price_change', np.nan):.3f}**\n"
                f"- Predicted Price Change t+1: **{pred.get('pred_price_change_t1', np.nan):.3f}**\n"
                f"- Implied WTI t+1: **{pred.get('pred_wti_price_t1', np.nan):.3f}**"
            ),
        )
    )


def show_holdout_charts(lang: str, holdout_df: pd.DataFrame) -> None:
    if holdout_df.empty:
        st.warning(tr(lang, "未找到 holdout 预测文件。", "Holdout prediction file is missing."))
        return
    df = holdout_df.copy()
    df["Event_Date"] = pd.to_datetime(df["Event_Date"], errors="coerce")
    fig1 = px.line(
        df.melt(
            id_vars=["Event_Date"],
            value_vars=["y_true_price_change_t1", "y_pred_price_change_t1"],
            var_name="series",
            value_name="value",
        ),
        x="Event_Date",
        y="value",
        color="series",
        markers=True,
        title=tr(lang, "Holdout: 真实值 vs 预测值", "Holdout: Actual vs Predicted"),
    )
    st.plotly_chart(fig1, use_container_width=True)

    df["residual"] = df["y_true_price_change_t1"] - df["y_pred_price_change_t1"]
    fig2 = px.histogram(df, x="residual", nbins=18, title=tr(lang, "残差分布", "Residual Distribution"))
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.scatter(
        df,
        x="y_true_price_change_t1",
        y="y_pred_price_change_t1",
        title=tr(lang, "校准图：真实 vs 预测", "Calibration: True vs Pred"),
    )
    mn = min(df["y_true_price_change_t1"].min(), df["y_pred_price_change_t1"].min())
    mx = max(df["y_true_price_change_t1"].max(), df["y_pred_price_change_t1"].max())
    fig3.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines", line={"dash": "dash"}, name="y=x"))
    st.plotly_chart(fig3, use_container_width=True)


def show_fold_table(lang: str, fold_df: pd.DataFrame) -> None:
    if fold_df.empty:
        return
    st.markdown(f"### {tr(lang, 'Walk-forward 各折结果', 'Walk-forward Fold Results')}")
    st.dataframe(fold_df, use_container_width=True)
    fig = px.bar(fold_df, x="fold", y="price_change_t1_mae", title=tr(lang, "各折 MAE", "Fold MAE"))
    st.plotly_chart(fig, use_container_width=True)


def show_artifact_images(lang: str) -> None:
    st.markdown(f"### {tr(lang, '过程图与描述图', 'Process and Descriptive Figures')}")
    img_names = [
        "process_flow.png",
        "walkforward_splits.png",
        "fold_metrics.png",
        "training_overview.png",
        "inventory_price_focus.png",
        "last_fold_predictions.png",
        "actual_vs_pred_scatter.png",
        "residual_distribution.png",
        "residual_timeseries.png",
        "training_process_curve.png",
        "final_result_holdout.png",
        "descriptive_dashboard.png",
        "holdout_diagnostics_dashboard.png",
        "weekly_core_panel.png",
    ]
    shown = 0
    for name in img_names:
        p = OUT_DIR / name
        if p.exists():
            st.image(str(p), caption=name, use_container_width=True)
            shown += 1
    if shown == 0:
        st.info(tr(lang, "尚未生成图像，请先运行 plots 或 train。", "No figure found. Run plots/train first."))


def show_downloads(lang: str) -> None:
    st.markdown(f"### {tr(lang, '结果文件下载', 'Downloads')}")
    files = [
        PKG_DIR / "predict_next_output.json",
        PKG_DIR / "final_package_summary.json",
        PKG_DIR / "final_holdout_predictions.csv",
        OUT_DIR / "walkforward_summary.json",
        OUT_DIR / "walkforward_fold_metrics.csv",
        OUT_DIR / "global_trial_ranking.csv",
    ]
    for p in files:
        if p.exists():
            st.download_button(
                label=f"{tr(lang, '下载', 'Download')} {p.name}",
                data=p.read_bytes(),
                file_name=p.name,
                mime="application/octet-stream",
                use_container_width=True,
            )


def main() -> None:
    st.set_page_config(page_title="Oil Analysis Forecast Console", page_icon="🛢️", layout="wide")
    lang = st.sidebar.selectbox("Language / 语言", ["中文", "English"], index=0)

    st.title(tr(lang, "石油价格变化预测控制台", "Oil Price-Change Forecast Console"))
    st.caption(
        tr(
            lang,
            "当前版本仅使用 WTI_T / Stock_T / Price_Change / Stock_Change 及其时序衍生特征，目标为 price_change_t1。",
            "Current version uses only WTI_T / Stock_T / Price_Change / Stock_Change and their time-series derivatives to predict price_change_t1.",
        )
    )

    with st.sidebar:
        st.markdown("## Actions")
        if st.button(tr(lang, "刷新预测", "Refresh Prediction"), use_container_width=True):
            ok, msg = run_predict()
            if ok:
                load_json.clear()
                st.success(tr(lang, "预测已刷新。", "Prediction refreshed."))
            else:
                st.error(msg)
        if st.button(tr(lang, "重生成图表", "Regenerate Plots"), use_container_width=True):
            ok, msg = run_plots()
            if ok:
                st.success(tr(lang, "图表已生成。", "Plots regenerated."))
            else:
                st.error(msg)
        if st.button(tr(lang, "完整重训（耗时长）", "Full Retrain (long)"), use_container_width=True):
            ok, msg = run_train()
            if ok:
                load_json.clear()
                load_csv.clear()
                st.success(tr(lang, "训练完成，请刷新页面查看。", "Training finished. Refresh to view."))
            else:
                st.error(msg)

    summary = load_json(OUT_DIR / "walkforward_summary.json")
    package_summary = load_json(PKG_DIR / "final_package_summary.json")
    pred = load_json(PKG_DIR / "predict_next_output.json")
    holdout_df = load_csv(PKG_DIR / "final_holdout_predictions.csv")
    fold_df = load_csv(OUT_DIR / "walkforward_fold_metrics.csv")

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            tr(lang, "总览", "Overview"),
            tr(lang, "模型评估", "Evaluation"),
            tr(lang, "图表看板", "Figure Board"),
            tr(lang, "导出与文件", "Exports"),
        ]
    )

    with tab1:
        show_top_metrics(lang, summary, pred)
        show_prediction_card(lang, pred)
        if package_summary:
            st.json(package_summary)

    with tab2:
        show_holdout_charts(lang, holdout_df)
        show_fold_table(lang, fold_df)

    with tab3:
        show_artifact_images(lang)

    with tab4:
        show_downloads(lang)
        st.markdown(f"### {tr(lang, '关键文件清单', 'Key Files')}")
        key_paths = [
            OUT_DIR / "walkforward_summary.json",
            PKG_DIR / "final_package_summary.json",
            PKG_DIR / "predict_next_output.json",
            OUT_DIR / "weekly_preprocessed.csv",
            OUT_DIR / "global_trial_ranking.csv",
        ]
        for p in key_paths:
            st.write(f"- `{p}`")


if __name__ == "__main__":
    main()
