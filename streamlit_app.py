from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA


BASE_DIR = Path(__file__).resolve().parent
ML_DIR = BASE_DIR / "ML"
NLP_DIR = BASE_DIR / "NLP"


def tr(lang: str, zh: str, en: str) -> str:
    return zh if lang == "中文" else en


def inject_style() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(circle at 12% 12%, #1f3b79 0%, #0d1117 35%, #06080f 100%);
            color: #e8f0ff;
        }
        .hero {
            border: 1px solid rgba(124, 221, 255, 0.25);
            border-radius: 16px;
            padding: 18px 20px;
            background: linear-gradient(135deg, rgba(17,35,65,0.72), rgba(16,19,35,0.72));
            box-shadow: 0 0 20px rgba(54, 197, 240, 0.12);
            margin-bottom: 12px;
        }
        .hero h1 { font-size: 1.8rem; margin: 0 0 6px 0; color: #a9ecff; }
        .hero p { margin: 0; color: #d7e9ff; }
        .guide-card {
            border: 1px solid rgba(124, 221, 255, 0.18);
            border-radius: 12px;
            padding: 12px 14px;
            background: rgba(9, 17, 30, 0.65);
            margin: 8px 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_chart_with_note(fig: go.Figure, note_md: str) -> None:
    c1, c2 = st.columns([2.4, 1.2])
    with c1:
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown(note_md)


def build_truth_pred_line(
    df: pd.DataFrame,
    x_col: str,
    y_true_col: str,
    y_pred_col: str,
    title: str,
    lang: str,
) -> go.Figure:
    true_name = tr(lang, "真实值", "Actual")
    pred_name = tr(lang, "预测值", "Prediction")
    line_df = df[[x_col, y_true_col, y_pred_col]].rename(
        columns={y_true_col: true_name, y_pred_col: pred_name}
    )
    line_df = line_df.melt(id_vars=[x_col], var_name="series", value_name="value")
    fig = px.line(
        line_df,
        x=x_col,
        y="value",
        color="series",
        markers=True,
        color_discrete_map={
            true_name: "#1f77b4",  # blue
            pred_name: "#ff7f0e",  # orange
        },
    )
    fig.update_layout(title=title, legend_title_text=tr(lang, "序列", "Series"))
    return fig


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


def run_ml_predict(data_path: str | None = None) -> tuple[bool, str, dict]:
    use_data = data_path or str(ML_DIR / "4.0_enriched.csv")
    cmd = [
        sys.executable,
        str(ML_DIR / "main.py"),
        "predict",
        "--data",
        use_data,
        "--package-dir",
        str(ML_DIR / "outputs_lstm_wf" / "final_package"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return False, (result.stderr or result.stdout).strip(), {}
    pred_path = ML_DIR / "outputs_lstm_wf" / "final_package" / "predict_next_output.json"
    pred = load_json(pred_path)
    return True, tr("中文", "预测刷新完成。", "Prediction refreshed."), pred


def run_ml_train() -> tuple[bool, str]:
    cmd = [sys.executable, str(ML_DIR / "main.py"), "train"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return False, (result.stderr or result.stdout).strip()
    return True, "训练完成。"


def build_prediction_insights(pred: dict, summary: dict, lang: str) -> list[str]:
    if not pred:
        return [tr(lang, "暂无可用预测结果。", "No prediction available.")]
    fold_mean = summary.get("fold_metrics_mean", {})
    p = float(pred.get("pred_price_change_t1", 0.0))
    s = float(pred.get("pred_stock_change_t1", 0.0))
    price_mae = float(fold_mean.get("price_change_t1_mae", 0.0))
    stock_mae = float(fold_mean.get("stock_change_t1_mae", 0.0))
    out = [
        tr(lang, "价格方向：上涨" if p > 0 else "价格方向：下跌", "Price direction: Up" if p > 0 else "Price direction: Down"),
        tr(lang, "库存方向：增加" if s > 0 else "库存方向：下降", "Inventory direction: Up" if s > 0 else "Inventory direction: Down"),
    ]
    out.append(
        tr(lang, "价格预测接近噪声区间，建议保守解读。", "Price move is close to noise band; interpret conservatively.")
        if abs(p) < max(0.8, 0.35 * max(price_mae, 1e-6))
        else tr(lang, "价格信号超过常规误差阈值。", "Price signal exceeds typical error threshold.")
    )
    out.append(
        tr(lang, "库存波动偏小，可能处于震荡。", "Inventory move is small; likely range-bound.")
        if abs(s) < 0.35 * max(stock_mae, 1e-6)
        else tr(lang, "库存波动较大，关注供应侧变化。", "Inventory move is large; watch supply-side shifts.")
    )
    return out


def nlp_text_predict(text: str, keywords: list[str]) -> tuple[float, str, pd.DataFrame]:
    if not text.strip():
        return 0.0, "empty", pd.DataFrame(columns=["keyword", "count"])
    hits = {kw: text.count(kw) for kw in keywords if text.count(kw) > 0}
    if not hits:
        return 0.0, "neutral", pd.DataFrame(columns=["keyword", "count"])
    bullish = {"减产", "战争", "制裁", "需求"}
    bearish = {"库存", "供应"}
    score = 0.0
    for k, c in hits.items():
        if k in bullish:
            score += 1.0 * c
        elif k in bearish:
            score -= 1.0 * c
        else:
            score += 0.2 * c
    label = "up" if score > 0 else "down" if score < 0 else "neutral"
    hit_df = pd.DataFrame({"keyword": list(hits.keys()), "count": list(hits.values())})
    return score, label, hit_df.sort_values("count", ascending=False)


def show_overview(
    summary: dict,
    pred: dict,
    holdout_df: pd.DataFrame,
    fold_df: pd.DataFrame,
    vectors_df: pd.DataFrame,
    weekly_df: pd.DataFrame,
    lang: str,
) -> None:
    st.subheader(tr(lang, "研究总览", "Research Overview"))
    fold_mean = summary.get("fold_metrics_mean", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(tr(lang, "价格 MAE(均值)", "Price MAE(mean)"), f"{fold_mean.get('price_change_t1_mae', np.nan):.3f}")
    c2.metric(tr(lang, "库存 MAE(均值)", "Stock MAE(mean)"), f"{fold_mean.get('stock_change_t1_mae', np.nan):.1f}")
    c3.metric(tr(lang, "下一期价格预测", "Next price prediction"), f"{pred.get('pred_price_change_t1', np.nan):.3f}")
    c4.metric(tr(lang, "下一期库存预测", "Next stock prediction"), f"{pred.get('pred_stock_change_t1', np.nan):.1f}")
    st.caption(
        tr(
            lang,
            f"样本量：时序序列 {summary.get('n_sequences', '-')}, NLP 向量 {len(vectors_df):,} 行。",
            f"Samples: sequences {summary.get('n_sequences', '-')}, NLP vectors {len(vectors_df):,} rows.",
        )
    )

    st.markdown(f"### {tr(lang, '预测结论', 'Prediction Findings')}")
    for line in build_prediction_insights(pred, summary, lang):
        st.write(f"- {line}")

    st.markdown(f"### {tr(lang, '调查结果摘要', 'Investigation Summary')}")
    audit_findings = compute_audit_findings(summary, weekly_df)
    if not audit_findings:
        st.success(tr(lang, "当前未发现显著数据与评估异常。", "No major data/evaluation issues detected."))
    else:
        for item in audit_findings:
            st.write(f"- {tr(lang, item['zh'], item['en'])}")

    if not fold_df.empty:
        fig2 = px.bar(fold_df, x="fold", y=["price_change_t1_mae", "stock_change_t1_mae"], barmode="group")
        fig2.update_layout(title=tr(lang, "调查结果：各折误差对比", "Investigation: Error by fold"))
        render_chart_with_note(
            fig2,
            tr(
                lang,
                "### 图解\n- 对比每折 price/stock MAE。\n- 若某折显著偏高，说明该时段分布不同。\n- 可据此决定是否分场景建模。",
                "### Reading guide\n- Compare fold-level MAE for price/stock.\n- One high fold implies regime shift.\n- Consider regime-based modeling.",
            ),
        )


def show_ml_analysis(holdout_df: pd.DataFrame, fold_df: pd.DataFrame, train_hist_df: pd.DataFrame, lang: str) -> None:
    st.subheader(tr(lang, "ML 预测与误差分析", "ML Prediction & Error Analysis"))
    if holdout_df.empty:
        st.warning(tr(lang, "未找到 holdout 预测文件。", "Holdout prediction file not found."))
        return
    h = holdout_df.copy()
    h["Event_Date"] = pd.to_datetime(h["Event_Date"])
    target = st.radio(tr(lang, "选择目标序列", "Target series"), ["price_change_t1", "stock_change_t1"], horizontal=True)
    y_true, y_pred = f"y_true_{target}", f"y_pred_{target}"
    fig1 = build_truth_pred_line(
        h,
        "Event_Date",
        y_true,
        y_pred,
        f"{target}: {tr(lang, '真实值 vs 预测值', 'Truth vs Prediction')}",
        lang,
    )
    render_chart_with_note(
        fig1,
        tr(
            lang,
            "### 图解\n- 时序对齐展示模型跟踪能力。\n- 峰值段若滞后，说明模型对突发冲击反应不足。",
            "### Reading guide\n- Time alignment shows tracking quality.\n- Lag at peaks implies weak shock response.",
        ),
    )
    h["residual"] = h[y_true] - h[y_pred]
    fig2 = px.histogram(h, x="residual", nbins=18, title=tr(lang, "残差分布", "Residual Distribution"))
    render_chart_with_note(
        fig2,
        tr(
            lang,
            "### 图解\n- 残差集中在 0 附近更理想。\n- 长尾说明存在少量高风险样本。",
            "### Reading guide\n- Centered around zero is desirable.\n- Heavy tails indicate risky cases.",
        ),
    )
    if not fold_df.empty:
        metric = st.selectbox(
            tr(lang, "查看各折指标", "Fold metric"),
            ["price_change_t1_mae", "price_change_t1_rmse", "stock_change_t1_mae", "stock_change_t1_rmse"],
        )
        fig3 = px.bar(fold_df, x="fold", y=metric, color=metric, color_continuous_scale="Blues")
        fig3.update_layout(title=tr(lang, "Walk-forward 各折表现", "Walk-forward fold performance"))
        render_chart_with_note(
            fig3,
            tr(
                lang,
                "### 图解\n- 横轴是时间折，纵轴是选定误差。\n- 曲线/柱子越平稳，泛化越稳定。",
                "### Reading guide\n- X: temporal folds, Y: selected error.\n- Flatter profile means stable generalization.",
            ),
        )
    if not train_hist_df.empty:
        fig4 = px.line(train_hist_df, x="epoch", y=["train_loss", "val_loss"], title=tr(lang, "训练损失曲线", "Training Loss Curves"))
        render_chart_with_note(
            fig4,
            tr(
                lang,
                "### 图解\n- train_loss 持续下降是学习进展。\n- val_loss 回升可能过拟合。",
                "### Reading guide\n- Falling train_loss shows learning.\n- Rising val_loss hints overfitting.",
            ),
        )


def show_nlp_analysis(vectors_df: pd.DataFrame, weekly_df: pd.DataFrame, lang: str) -> None:
    st.subheader(tr(lang, "NLP 向量洞察", "NLP Vector Insights"))
    if vectors_df.empty:
        st.warning(tr(lang, "未找到 NLP 向量文件。", "NLP vector file not found."))
        return
    n_rows = len(vectors_df)
    end_idx = st.slider(tr(lang, "样本窗口", "Sample window"), 50, min(n_rows, 400), min(220, min(n_rows, 400)))
    sample = vectors_df.iloc[:end_idx].copy()
    pca = PCA(n_components=2, random_state=42)
    points = pca.fit_transform(sample.values)
    pca_df = pd.DataFrame({"PC1": points[:, 0], "PC2": points[:, 1], "idx": np.arange(len(sample))})
    fig1 = px.scatter(pca_df, x="PC1", y="PC2", color="idx", color_continuous_scale="Turbo")
    fig1.update_layout(title=tr(lang, f"PCA 投影（前 {end_idx} 条）", f"PCA projection (first {end_idx})"))
    render_chart_with_note(
        fig1,
        tr(
            lang,
            f"### 图解\n- PCA 将高维向量压到二维。\n- 若出现明显簇，说明文本语义有分层。\n- 方差解释率：PC1={pca.explained_variance_ratio_[0]:.3f}, PC2={pca.explained_variance_ratio_[1]:.3f}",
            f"### Reading guide\n- PCA compresses high-dimensional vectors to 2D.\n- Clusters indicate semantic grouping.\n- Explained variance: PC1={pca.explained_variance_ratio_[0]:.3f}, PC2={pca.explained_variance_ratio_[1]:.3f}",
        ),
    )
    col = st.selectbox(tr(lang, "查看单维分布", "Single dimension"), list(vectors_df.columns)[:30])
    fig2 = px.histogram(vectors_df, x=col, nbins=30, title=f"{col} distribution")
    render_chart_with_note(
        fig2,
        tr(
            lang,
            "### 图解\n- 观察该维度是否偏态/长尾。\n- 极端值多时，模型可能受少量样本影响。",
            "### Reading guide\n- Check skew/heavy tails.\n- Many extremes may dominate model behavior.",
        ),
    )
    if not weekly_df.empty and "Price_Change" in weekly_df.columns:
        align_n = min(len(vectors_df), len(weekly_df))
        corr = vectors_df.iloc[:align_n].corrwith(weekly_df["Price_Change"].iloc[:align_n]).sort_values(key=np.abs, ascending=False).head(12)
        corr_df = corr.reset_index()
        corr_df.columns = ["vector_dim", "corr"]
        fig3 = px.bar(corr_df, x="vector_dim", y="corr", color="corr", color_continuous_scale="RdBu")
        fig3.update_layout(title=tr(lang, "向量维度与价格变化相关性 Top12", "Top12 correlation with price change"))
        render_chart_with_note(
            fig3,
            tr(
                lang,
                "### 图解\n- 绝对值越大，线性关联越强。\n- 正相关维度可解释上涨语义，负相关对应下行语义。",
                "### Reading guide\n- Higher abs(corr) means stronger linear relation.\n- Positive dims may map to bullish semantics, negative to bearish.",
            ),
        )


def show_prediction_workspace(summary: dict, pred: dict, weekly_df: pd.DataFrame, nlp_cfg: dict, lang: str) -> None:
    st.subheader(tr(lang, "预测模块（ML 与 NLP 分离）", "Prediction Module (ML and NLP separated)"))
    tab_ml, tab_nlp = st.tabs(
        [
            tr(lang, "ML 数值输入预测", "ML Numeric Prediction"),
            tr(lang, "NLP 文本输入预测", "NLP Text Prediction"),
        ]
    )

    with tab_ml:
        st.markdown(f"### {tr(lang, '1) 上传数值 CSV 做真实模型预测', '1) Upload numeric CSV for model inference')}")
        up = st.file_uploader(tr(lang, "上传数值 CSV", "Upload numeric CSV"), type=["csv"], accept_multiple_files=False, key="ml_upload")
        if up is not None:
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                tmp.write(up.getvalue())
                tmp_path = tmp.name
            if st.button(tr(lang, "执行 ML 预测", "Run ML prediction"), use_container_width=True, key="btn_ml_upload"):
                ok, msg, p = run_ml_predict(tmp_path)
                if ok:
                    st.success(tr(lang, "预测完成。", "Prediction done."))
                    st.json(p)
                    for line in build_prediction_insights(p, summary, lang):
                        st.write(f"- {line}")
                else:
                    st.error(msg)

        st.markdown("---")
        st.markdown(f"### {tr(lang, '2) 手动数值场景预测（近邻法）', '2) Manual numeric scenario (nearest neighbors)')}")
        needed = [
            "Price_Change",
            "Stock_Change",
            "WTI_T",
            "dxy_broad",
            "us10y_yield",
            "fed_funds_rate",
            "target_price_change_t1",
            "target_stock_change_t1",
        ]
        if weekly_df.empty or any(c not in weekly_df.columns for c in needed):
            st.warning(tr(lang, "历史字段不足，无法执行场景预测。", "Missing fields for scenario prediction."))
        else:
            core = weekly_df[needed].dropna().copy()
            inputs = needed[:6]
            vals = {}
            for c in inputs:
                q05, q95, dv = float(core[c].quantile(0.05)), float(core[c].quantile(0.95)), float(core[c].iloc[-1])
                vals[c] = st.slider(c, float(min(q05, dv)), float(max(q95, dv)), float(dv), key=f"ml_{c}")
            k = st.slider("K", 5, 30, 12, 1, key="ml_k")
            if st.button(tr(lang, "生成场景预测", "Generate scenario forecast"), use_container_width=True, key="btn_scn"):
                X = core[inputs]
                x = pd.Series(vals)[inputs]
                sigma = X.std().replace(0, 1.0)
                dist = np.sqrt((((X - x) / sigma) ** 2).sum(axis=1))
                idx = dist.nsmallest(k).index
                p1 = core.loc[idx, "target_price_change_t1"].mean()
                p2 = core.loc[idx, "target_stock_change_t1"].mean()
                st.metric("price_change_t1", f"{p1:.3f}")
                st.metric("stock_change_t1", f"{p2:.1f}")
                near = weekly_df.loc[idx, ["Event_Date", "Price_Change", "Stock_Change", "target_price_change_t1"]].copy()
                near["Event_Date"] = pd.to_datetime(near["Event_Date"])
                fig = px.scatter(
                    near.sort_values("Event_Date"),
                    x="Event_Date",
                    y="target_price_change_t1",
                    size=np.abs(near["Stock_Change"]) + 1,
                    color="Price_Change",
                )
                fig.update_layout(title=tr(lang, "近邻样本分布", "Nearest samples distribution"))
                render_chart_with_note(
                    fig,
                    tr(
                        lang,
                        "### 解释\n- 该方法不替代主模型，只用于快速场景推演。\n- 近邻越集中，场景预测通常越稳定。",
                        "### Explanation\n- This is a fast what-if tool, not a replacement for the main model.\n- More concentrated neighbors usually imply more stable forecast.",
                    ),
                )

    with tab_nlp:
        st.markdown(f"### {tr(lang, '文本输入预测（关键词语义信号）', 'Text-input prediction (keyword semantic signal)')}")
        keywords = nlp_cfg.get("visualization", {}).get("keywords", [])
        text = st.text_area(tr(lang, "输入新闻/事件文本", "Input event/news text"), "", key="nlp_text")
        if st.button(tr(lang, "执行 NLP 预测", "Run NLP prediction"), use_container_width=True, key="btn_nlp_pred"):
            score, label, hit_df = nlp_text_predict(text, keywords)
            if label == "empty":
                st.warning(tr(lang, "请输入文本后再预测。", "Please input text first."))
            else:
                label_map = {
                    "up": tr(lang, "偏上涨", "Bullish"),
                    "down": tr(lang, "偏下跌", "Bearish"),
                    "neutral": tr(lang, "中性", "Neutral"),
                }
                st.metric(tr(lang, "文本信号方向", "Text signal direction"), label_map[label])
                st.metric(tr(lang, "文本信号分数", "Text signal score"), f"{score:.2f}")
                if not hit_df.empty:
                    fig = px.bar(hit_df, x="keyword", y="count", color="count", color_continuous_scale="Bluered")
                    fig.update_layout(title=tr(lang, "关键词命中强度", "Keyword hit strength"))
                    render_chart_with_note(
                        fig,
                        tr(
                            lang,
                            "### 解释\n- 这是文本语义信号，不是价格点位回归结果。\n- 建议和 ML 数值预测联用。",
                            "### Explanation\n- This is a semantic signal, not point forecast.\n- Use together with ML numeric forecast.",
                        ),
                    )


def show_data_explorer(weekly_df: pd.DataFrame, vectors_df: pd.DataFrame, lang: str) -> None:
    st.subheader(tr(lang, "数据视图", "Data View"))
    if weekly_df.empty:
        st.warning(tr(lang, "未找到周度特征数据。", "Weekly feature data not found."))
        return
    df = weekly_df.copy()
    df["Event_Date"] = pd.to_datetime(df["Event_Date"])
    st.caption(tr(lang, f"周度样本：{len(df)}，字段：{df.shape[1]}", f"Weekly rows: {len(df)}, columns: {df.shape[1]}"))
    c1, c2, c3 = st.columns(3)
    with c1:
        sdt = st.date_input(tr(lang, "起始日期", "Start date"), value=df["Event_Date"].min().date())
    with c2:
        edt = st.date_input(tr(lang, "结束日期", "End date"), value=df["Event_Date"].max().date())
    with c3:
        value_col = st.selectbox(tr(lang, "主观察字段", "Primary field"), [c for c in ["WTI_T", "Price_Change", "Stock_Change", "dxy_broad", "us10y_yield"] if c in df.columns])
    view = df[(df["Event_Date"].dt.date >= sdt) & (df["Event_Date"].dt.date <= edt)].copy()
    fig = px.line(view, x="Event_Date", y=value_col, markers=True, title=f"{value_col}")
    render_chart_with_note(
        fig,
        tr(
            lang,
            "### 图解\n- 用于观察宏观变量的阶段性趋势。\n- 可配合下方滚动均值查看平滑走势。",
            "### Reading guide\n- Observe regime-wise trend of macro variable.\n- Use rolling mean below for smoother signal.",
        ),
    )
    roll_n = st.slider(tr(lang, "滚动窗口（周）", "Rolling window (weeks)"), 3, 24, 8)
    view[f"{value_col}_roll"] = view[value_col].rolling(roll_n).mean()
    fig2 = px.line(view, x="Event_Date", y=[value_col, f"{value_col}_roll"])
    fig2.update_layout(title=tr(lang, "原序列 vs 滚动均值", "Raw vs rolling mean"))
    render_chart_with_note(
        fig2,
        tr(
            lang,
            "### 图解\n- 滚动均值过滤短期噪声。\n- 原序列与均值偏离大时，代表短期冲击更强。",
            "### Reading guide\n- Rolling mean filters short-term noise.\n- Large deviation indicates stronger short-term shocks.",
        ),
    )
    st.metric(tr(lang, "NLP 向量行数", "NLP vector rows"), f"{len(vectors_df):,}")
    with st.expander(tr(lang, "查看筛选后的数据", "View filtered data"), expanded=False):
        st.dataframe(view.tail(120), use_container_width=True)


def show_experiment_tracker(summary: dict, trial_df: pd.DataFrame, fold_df: pd.DataFrame, lang: str) -> None:
    st.subheader(tr(lang, "实验追踪中心", "Experiment Tracker"))
    a, b, c = st.columns(3)
    a.metric(tr(lang, "试验数", "Trials"), summary.get("trial_count", "-"))
    b.metric(tr(lang, "折数", "Folds"), summary.get("n_folds", "-"))
    c.metric(tr(lang, "序列样本", "Sequences"), summary.get("n_sequences", "-"))
    if not trial_df.empty:
        n = st.slider(tr(lang, "展示前 N 个 trial", "Top N trials"), 3, min(20, len(trial_df)), min(10, len(trial_df)))
        sub = trial_df.head(n)
        fig1 = px.bar(sub, x="trial_id", y="mean_selection_score", color="mean_selection_score")
        fig1.update_layout(title=tr(lang, "Trial 排名（越低越优）", "Trial ranking (lower is better)"))
        render_chart_with_note(
            fig1,
            tr(
                lang,
                "### 图解\n- 评分综合了价格与库存误差。\n- 可快速定位候选超参组合。",
                "### Reading guide\n- Score combines price/stock errors.\n- Quickly locate candidate hyper-parameter sets.",
            ),
        )
    if not fold_df.empty:
        m = st.selectbox(tr(lang, "折线跟踪指标", "Fold metric"), ["price_change_t1_mae", "stock_change_t1_mae", "price_change_t1_rmse", "stock_change_t1_rmse"])
        fig2 = px.line(fold_df, x="fold", y=m, markers=True)
        fig2.update_layout(title=tr(lang, "跨折稳定性", "Cross-fold stability"))
        render_chart_with_note(
            fig2,
            tr(
                lang,
                "### 图解\n- 趋势平稳说明模型更稳健。\n- 突刺代表某些时间段难度更高。",
                "### Reading guide\n- Flat trend indicates robustness.\n- Spikes imply difficult periods.",
            ),
        )


def show_feature_lab(feature_df: pd.DataFrame, weekly_df: pd.DataFrame, lang: str) -> None:
    st.subheader(tr(lang, "特征实验室", "Feature Lab"))
    if feature_df.empty or weekly_df.empty:
        st.warning(tr(lang, "缺少特征或周度数据。", "Missing feature or weekly data."))
        return
    top_n = st.slider("Top N", 10, min(40, len(feature_df)), min(20, len(feature_df)))
    top = feature_df.sort_values("selected_in_folds", ascending=False).head(top_n)
    fig1 = px.bar(top, x="feature", y="selected_in_folds", color="selected_in_folds")
    fig1.update_layout(title=tr(lang, "特征入选频次", "Feature selection frequency"))
    render_chart_with_note(
        fig1,
        tr(
            lang,
            "### 图解\n- 频次高代表跨折稳定贡献。\n- 可据此做特征压缩和解释。",
            "### Reading guide\n- High frequency means stable contribution.\n- Useful for feature pruning and interpretation.",
        ),
    )
    target = st.radio(tr(lang, "相关性目标", "Correlation target"), ["Price_Change", "Stock_Change"], horizontal=True)
    cand = [f for f in top["feature"] if f in weekly_df.columns]
    if cand:
        corr = weekly_df[cand + [target]].corr()[target].drop(target).sort_values(key=np.abs, ascending=False).head(15)
        cdf = corr.reset_index()
        cdf.columns = ["feature", "corr"]
        fig2 = px.bar(cdf, x="feature", y="corr", color="corr", color_continuous_scale="RdBu")
        fig2.update_layout(title=tr(lang, "相关性 Top15", "Top15 correlations"))
        render_chart_with_note(
            fig2,
            tr(
                lang,
                "### 图解\n- 看绝对值强度与正负方向。\n- 仅表示线性相关，不等于因果。",
                "### Reading guide\n- Check magnitude and sign.\n- Correlation is not causality.",
            ),
        )


def show_risk_monitor(holdout_df: pd.DataFrame, summary: dict, lang: str) -> None:
    st.subheader(tr(lang, "风险监控", "Risk Monitor"))
    if holdout_df.empty:
        st.warning(tr(lang, "缺少 holdout 数据。", "Missing holdout data."))
        return
    t = st.radio(tr(lang, "监控目标", "Target"), ["price_change_t1", "stock_change_t1"], horizontal=True)
    y_true, y_pred = f"y_true_{t}", f"y_pred_{t}"
    resid = holdout_df[y_true] - holdout_df[y_pred]
    sigma = resid.std() if resid.std() > 0 else 1.0
    z = np.abs((resid - resid.mean()) / sigma)
    risk = holdout_df.copy()
    risk["anomaly_score"] = z
    risk["is_anomaly"] = risk["anomaly_score"] >= 2.0
    ratio = float(risk["is_anomaly"].mean())
    fig1 = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=ratio * 100,
            title={"text": tr(lang, "异常样本占比(%)", "Anomaly ratio (%)")},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#47d7ff"}},
        )
    )
    render_chart_with_note(
        fig1,
        tr(
            lang,
            "### 图解\n- 异常占比持续上升时，建议重训。\n- 建议与数据漂移指标联动监控。",
            "### Reading guide\n- Rising anomaly ratio suggests retraining.\n- Monitor together with data drift.",
        ),
    )
    fig2 = px.scatter(risk.reset_index(), x="index", y="anomaly_score", color="is_anomaly")
    fig2.update_layout(title=tr(lang, "异常得分轨迹", "Anomaly score trajectory"))
    render_chart_with_note(
        fig2,
        tr(
            lang,
            "### 图解\n- 阈值通常设为 2 或 3。\n- 连续异常点说明模型可能失配。",
            "### Reading guide\n- Typical threshold is 2 or 3.\n- Consecutive anomalies imply model mismatch.",
        ),
    )
    m = summary.get("fold_metrics_mean", {})
    st.write(tr(lang, f"- 当前 MAE: price={m.get('price_change_t1_mae', np.nan):.3f}, stock={m.get('stock_change_t1_mae', np.nan):.1f}", f"- Current MAE: price={m.get('price_change_t1_mae', np.nan):.3f}, stock={m.get('stock_change_t1_mae', np.nan):.1f}"))


def compute_audit_findings(summary: dict, weekly_df: pd.DataFrame) -> list[dict]:
    findings: list[dict] = []
    if weekly_df.empty:
        findings.append({"zh": "周度数据为空。", "en": "Weekly data is empty."})
        return findings
    w = weekly_df.copy()
    w["Event_Date"] = pd.to_datetime(w["Event_Date"], errors="coerce")
    if w["Event_Date"].isna().any():
        findings.append({"zh": "存在无法解析的日期。", "en": "Unparseable dates found."})
    if w["Event_Date"].duplicated().any():
        findings.append({"zh": "存在重复日期记录。", "en": "Duplicated dates found."})
    if not w["Event_Date"].is_monotonic_increasing:
        findings.append({"zh": "日期并非严格递增。", "en": "Dates are not strictly increasing."})

    miss_ratio = w.isna().mean().mean()
    if miss_ratio > 0.01:
        findings.append({"zh": f"缺失率偏高：{miss_ratio:.2%}。", "en": f"Missing ratio is high: {miss_ratio:.2%}."})

    fold_mean = summary.get("fold_metrics_mean", {})
    p_r2 = float(fold_mean.get("price_change_t1_r2", 0.0))
    s_r2 = float(fold_mean.get("stock_change_t1_r2", 0.0))
    if p_r2 < 0:
        findings.append({"zh": "价格任务 R2 < 0，低于简单基线。", "en": "Price task R2 < 0, under simple baseline."})
    if s_r2 < 0:
        findings.append({"zh": "库存任务 R2 < 0，低于简单基线。", "en": "Stock task R2 < 0, under simple baseline."})
    return findings


def show_data_audit(summary: dict, weekly_df: pd.DataFrame, holdout_df: pd.DataFrame, lang: str) -> None:
    st.subheader(tr(lang, "数据合理性审计与流程修正", "Data Audit and Process Corrections"))
    issues = [tr(lang, x["zh"], x["en"]) for x in compute_audit_findings(summary, weekly_df)]

    train_code = (ML_DIR / "train_lstm_pipeline.py").read_text(encoding="utf-8")
    if "interpolate(method=\"linear\", limit_direction=\"both\")" in train_code:
        issues.append(tr(lang, "检测到双向插值，可能引入未来信息泄漏。", "Bidirectional interpolation detected; potential future leakage."))

    if not issues:
        st.success(tr(lang, "未发现显著数据或流程问题。", "No major data/process issues detected."))
    else:
        st.error(tr(lang, "发现以下问题：", "Detected issues:"))
        for i in issues:
            st.write(f"- {i}")

    st.markdown(f"### {tr(lang, '修正建议', 'Correction recommendations')}")
    st.markdown(
        tr(
            lang,
            "- 使用**仅前向填充**避免未来信息泄漏。\n- 增加方向准确率指标并与 MAE/RMSE 联合解释。\n- 对异常值占比高的字段先做稳健裁剪或分段建模。\n- 若 R2 持续为负，优先比较简单基线并缩减模型复杂度。",
            "- Use **forward-only fill** to avoid future leakage.\n- Add direction accuracy with MAE/RMSE for interpretation.\n- For high-outlier fields, apply robust clipping or regime modeling.\n- If R2 stays negative, compare against naive baseline and reduce model complexity.",
        )
    )
    if st.button(tr(lang, "执行完整重训（耗时较长）", "Run full retraining (long)"), use_container_width=True):
        ok, msg = run_ml_train()
        if ok:
            st.success(tr(lang, "重训完成，请刷新页面查看新结果。", "Retraining completed. Refresh to load new results."))
        else:
            st.error(msg)


def show_model_card_and_download(summary: dict, package_summary: dict, lang: str) -> None:
    st.subheader(tr(lang, "模型卡与下载中心", "Model Card & Download Center"))
    best = package_summary.get("best_params", {})
    holdout = package_summary.get("holdout_metrics", {})
    st.markdown(f"### {tr(lang, '模型卡', 'Model Card')}")
    st.write(tr(lang, "- 模型：LSTMRegressor（双输出）", "- Model: LSTMRegressor (dual output)"))
    st.write(tr(lang, "- 输入：周度时序特征 + 工程特征", "- Input: weekly sequence + engineered features"))
    st.write(tr(lang, "- 输出：target_price_change_t1 / target_stock_change_t1", "- Output: target_price_change_t1 / target_stock_change_t1"))
    st.write(f"- best_params: `{json.dumps(best, ensure_ascii=False)}`")
    st.write(f"- holdout_metrics: `{json.dumps(holdout, ensure_ascii=False)}`")
    st.write(f"- trial_count: `{summary.get('trial_count', '-')}`")
    st.markdown(f"### {tr(lang, '文件下载', 'Downloads')}")
    files = [
        ML_DIR / "outputs_lstm_wf" / "final_package" / "predict_next_output.json",
        ML_DIR / "outputs_lstm_wf" / "final_package" / "final_holdout_predictions.csv",
        ML_DIR / "outputs_lstm_wf" / "walkforward_fold_metrics.csv",
        ML_DIR / "outputs_lstm_wf" / "global_trial_ranking.csv",
        NLP_DIR / "outputs" / "text_vectors.csv",
    ]
    for p in files:
        if p.exists():
            st.download_button(label=f"{tr(lang, '下载', 'Download')} {p.name}", data=p.read_bytes(), file_name=p.name, mime="application/octet-stream", use_container_width=True)


def show_algorithm_principles(lang: str) -> None:
    st.subheader(tr(lang, "算法原理与流程", "Algorithm Principles & Process"))
    st.markdown(
        tr(
            lang,
            """
### 1) ML 主流程（LSTM）
1. 周度对齐与特征工程：滞后、滚动统计、周期项。
2. Walk-forward 切分：保证时间因果，避免泄漏。
3. 多组超参 trial：比较验证/测试误差，选择最优参数。
4. 最终模型重训并导出：产出 `final_package` 与预测 JSON。

### 2) NLP 主流程（Word2Vec）
1. 中文分词（jieba），构建词序列。
2. 训练 Word2Vec 得到词向量。
3. 句向量=词向量均值，形成事件语义特征。
4. PCA/相关性分析解释文本信号与价格变化关系。

### 3) 关键方法说明
- **LSTM**：通过门控机制记忆长期依赖，适合时序预测。  
- **Walk-forward**：每折只用过去预测未来，更符合真实交易场景。  
- **近邻场景预测**：标准化距离找到历史相似状态，取其未来均值。  
- **异常监控**：以残差 z-score 识别模型失配样本。
            """,
            """
### 1) ML pipeline (LSTM)
1. Weekly alignment + feature engineering: lags, rolling stats, seasonality.
2. Walk-forward split to preserve temporal causality.
3. Hyper-parameter trials and model selection.
4. Final retrain and export `final_package` + prediction JSON.

### 2) NLP pipeline (Word2Vec)
1. Chinese tokenization (jieba) to token sequences.
2. Train Word2Vec embeddings.
3. Sentence vector = average token vectors.
4. PCA/correlation for interpretability with price changes.

### 3) Method highlights
- **LSTM** captures long dependencies via gating mechanisms.  
- **Walk-forward** mimics real forecasting and avoids leakage.  
- **Nearest-neighbor scenario** averages future outcomes of similar states.  
- **Anomaly monitor** uses residual z-score for mismatch detection.
            """,
        )
    )


def show_guide(lang: str) -> None:
    st.subheader(tr(lang, "使用引导", "Usage Guide"))
    st.markdown(
        f"""
        <div class="guide-card"><b>1.</b> {tr(lang, "先看总览，理解误差与方向。", "Start from Overview for error and direction.")}</div>
        <div class="guide-card"><b>2.</b> {tr(lang, "进入 ML 分析，检查残差和跨折稳定性。", "Go to ML Analysis for residuals and fold stability.")}</div>
        <div class="guide-card"><b>3.</b> {tr(lang, "查看 NLP 洞察，理解文本语义结构。", "Use NLP Insights to inspect semantic structures.")}</div>
        <div class="guide-card"><b>4.</b> {tr(lang, "在用户预测页做上传预测和场景推演。", "Use User Prediction page for upload/scenario inference.")}</div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Oil Analysis Research Platform", page_icon="🛢️", layout="wide", initial_sidebar_state="expanded")
    inject_style()
    top_l, top_r = st.columns([7, 2])
    with top_r:
        lang = st.selectbox("Language / 语言", ["中文", "English"], index=0, label_visibility="visible")

    st.markdown(
        f"""
        <div class="hero">
            <h1>{tr(lang, "Oil Analysis 学术分析平台", "Oil Analysis Academic Platform")}</h1>
            <p>{tr(lang, "用于油价相关数值与文本建模研究：结构清晰、可解释、可审计。", "For numeric and text modeling research on oil markets: structured, explainable, and auditable.")}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    summary = load_json(ML_DIR / "outputs_lstm_wf" / "walkforward_summary.json")
    pred = load_json(ML_DIR / "outputs_lstm_wf" / "final_package" / "predict_next_output.json")
    holdout_df = load_csv(ML_DIR / "outputs_lstm_wf" / "final_package" / "final_holdout_predictions.csv")
    fold_df = load_csv(ML_DIR / "outputs_lstm_wf" / "walkforward_fold_metrics.csv")
    train_hist_df = load_csv(ML_DIR / "outputs_lstm_wf" / "final_package" / "final_train_history.csv")
    vectors_df = load_csv(NLP_DIR / "outputs" / "text_vectors.csv")
    weekly_df = load_csv(ML_DIR / "outputs_lstm_wf" / "weekly_preprocessed.csv")
    nlp_cfg = load_json(NLP_DIR / "config.json")

    page_options = {
        tr(lang, "研究总览", "Overview"): "overview",
        tr(lang, "预测模块", "Prediction Module"): "pred",
        tr(lang, "ML 分析", "ML Analysis"): "ml",
        tr(lang, "NLP 洞察", "NLP Insights"): "nlp",
        tr(lang, "数据审计", "Data Audit"): "audit",
        tr(lang, "算法原理", "Algorithm Principles"): "algo",
    }

    with st.sidebar:
        st.header(tr(lang, "参数面板", "Parameter Panel"))
        if st.button(tr(lang, "刷新预测结果", "Refresh prediction"), use_container_width=True):
            ok, msg, new_pred = run_ml_predict()
            if ok:
                load_json.clear()
                pred = new_pred
                st.success(tr(lang, "刷新完成。", "Refreshed."))
            else:
                st.error(msg)
        page_label = st.radio(tr(lang, "导航", "Navigation"), list(page_options.keys()), index=0)
        st.caption(tr(lang, "图表支持缩放、拖拽、框选与导出。", "Charts support zoom, pan, box-select and export."))

    page = page_options[page_label]
    if page == "overview":
        show_overview(summary, pred, holdout_df, fold_df, vectors_df, weekly_df, lang)
    elif page == "pred":
        show_prediction_workspace(summary, pred, weekly_df, nlp_cfg, lang)
    elif page == "ml":
        show_ml_analysis(holdout_df, fold_df, train_hist_df, lang)
    elif page == "nlp":
        show_nlp_analysis(vectors_df, weekly_df, lang)
    elif page == "audit":
        show_data_audit(summary, weekly_df, holdout_df, lang)
    else:
        show_algorithm_principles(lang)


if __name__ == "__main__":
    main()

