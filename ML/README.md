# Oil Analysis ML 强化版说明

## 1. 目标与约束

本项目当前只做一件事：

- **预测目标**：`price_change_t1`（下一期油价变化）
- **输入来源**（严格限定）：
  - `WTI_T`（当前油价）
  - `Stock_T`（当前库存）
  - `Price_Change`（当前价格变化）
  - `Stock_Change`（当前库存变化）
  - 以上四项的时序衍生特征（lag / rolling / diff / pct / 交互项）

不再使用美元指数、利率等宏观列作为模型输入。

---

## 2. 当前模型流程（已落地）

训练主脚本：`train_lstm_pipeline.py`

流程步骤：

1. 读取 `4.0_enriched.csv` 并按周对齐
2. 构建四大核心序列及其时序特征
3. 构建 `lookback` 序列样本，目标为 `target_price_change_t1`
4. Walk-forward 多折验证（时间因果）
5. 多轮随机搜索（当前为 `2 x 50 = 100` 组）
6. Top-N 模型加权集成（`ensemble_top_n=5`）
7. 最近窗口专门模型（recent specialist）
8. 验证集自动学习集成/近期模型混合比例
9. 最终误差校正（Ridge）
10. 导出最终预测包和评估图

---

## 3. 强化策略说明

当前强化版包含三层组合：

- **层1：Top-N 模型集成**
  - 从全局 trial 排名选前 N 个参数组合
  - 每个子模型单独训练并生成预测
  - 按验证表现动态赋权

- **层2：近期窗口专模**
  - 使用最近窗口数据（`recent_tune_window=96`）训练 specialist
  - 用验证集自动确定与层1的混合比例

- **层3：误差校正**
  - 使用 Ridge 对组合结果做二次校正
  - 校正输入：`pred_raw`, `WTI_T`, `Stock_T`, `Price_Change`, `Stock_Change`

---

## 4. 主要文件结构

### 训练与入口

- `main.py`：统一入口（train / predict / plots）
- `train_lstm_pipeline.py`：训练、搜索、导出
- `generate_plots.py`：补充生成描述图与诊断图

### 结果目录 `outputs_lstm_wf/`

- `walkforward_summary.json`：全局回测总结
- `walkforward_fold_metrics.csv`：各折指标
- `global_trial_ranking.csv`：trial 排名
- `weekly_preprocessed.csv`：周度特征数据
- `fold_*_history.csv / predictions.csv / trials.json`：折内明细
- 图像文件（流程图、训练图、评估图、诊断图）

### 最终模型包 `outputs_lstm_wf/final_package/`

- `final_package_summary.json`：最终包指标与参数
- `final_feature_columns.json`：最终特征列
- `final_holdout_predictions.csv`：holdout 预测明细
- `final_train_history.csv`：主模型训练历史
- `final_x_scaler.pkl`, `final_y_scaler.pkl`
- `final_lstm_model.pt`（兼容）
- `final_lstm_model_1.pt` ... `final_lstm_model_5.pt`（集成子模型）
- `final_recent_specialist_model.pt`（近期专模）
- `final_error_corrector.pkl`（误差校正器）
- `predict_next_output.json`（最新预测输出）

---

## 5. 已生成图表（核心）

流程/训练：

- `process_flow.png`
- `walkforward_splits.png`
- `training_overview.png`
- `training_process_curve.png`

评估/诊断：

- `fold_metrics.png`
- `last_fold_predictions.png`
- `actual_vs_pred_scatter.png`
- `residual_distribution.png`
- `residual_timeseries.png`
- `holdout_diagnostics_dashboard.png`

描述性：

- `inventory_price_focus.png`
- `weekly_core_panel.png`
- `descriptive_dashboard.png`
- `final_result_holdout.png`

---

## 6. 如何运行

安装依赖：

```bash
pip install -r requirements.txt
```

训练（完整流程）：

```bash
python main.py train
```

预测（使用最终模型包）：

```bash
python main.py predict
```

补生成图表：

```bash
python main.py plots
```

---

## 7. Streamlit 前端

父目录下前端：`../streamlit_app.py`

现已改为仅服务当前强化版 ML 流程：

- 总览：核心指标 + 最新预测
- 模型评估：holdout 曲线、残差、校准图、fold 表
- 图表看板：展示所有 PNG 产物
- 导出区：一键下载关键 JSON/CSV 结果

---

## 8. 当前结果读取建议

先看：

1. `final_package_summary.json` 的 `holdout_metrics`
2. `walkforward_summary.json` 的 `fold_metrics_mean`
3. `predict_next_output.json` 的当前输入与预测输出

---

## 9. 注意事项

- 这是时序预测，不能承诺“完美预测”
- 指标改善要重点看：`MAE/RMSE/R2/方向准确率`
- 若数据分布变化明显，建议定期重训并关注近期窗口表现

---

## 10. 结果解读模板（可直接用于汇报）

你可以按下面模板直接填数，形成固定周报：

### A) 本期结论（一句话）

- 示例：本期模型判断 `price_change_t1` 为正，预测值为 `+1.10`，方向偏上行。

### B) 指标摘要（核心四项）

- Holdout MAE：`__`
- Holdout RMSE：`__`
- Holdout R2：`__`
- 方向准确率：`__%`

### C) 与上期对比（环比）

- MAE 变化：`__ -> __`（`下降/上升 __%`）
- 方向准确率变化：`__% -> __%`
- 结论：`稳定提升 / 小幅回撤 / 明显退化`

### D) 风险与解释

- 是否出现连续残差偏同一方向：`是/否`
- 是否出现异常点增多（残差长尾变重）：`是/否`
- 解释要点（建议 1-2 条）：
  - `__`
  - `__`

### E) 动作建议（下周）

- `继续使用当前模型 / 触发重训 / 调整窗口参数 / 增加最近窗口权重`

---

## 11. 每张图怎么读（速查）

### 流程类

- `process_flow.png`：确认端到端步骤是否完整（数据 -> 训练 -> 集成 -> 校正 -> 导出）。
- `walkforward_splits.png`：检查切分是否严格时间因果，避免未来泄漏。

### 训练类

- `training_overview.png`、`training_process_curve.png`：
  - `train_loss` 下降且 `val_loss` 不反弹过大，说明训练稳定；
  - 若 `val_loss` 明显上拐，通常表示过拟合风险增加。

### 评估类

- `fold_metrics.png`：看各折 MAE/RMSE 是否平稳，尖峰折代表该时段 regime 更难。
- `last_fold_predictions.png`、`final_result_holdout.png`：
  - 真实值与预测值走势是否同向；
  - 峰值附近是否严重滞后。
- `actual_vs_pred_scatter.png`：
  - 点越贴近对角线越好；
  - 若斜率偏小，常见为振幅压缩（大涨大跌预测不足）。
- `residual_distribution.png`：
  - 分布以 0 为中心越好；
  - 长尾越重，极端周误差越大。
- `residual_timeseries.png`：
  - 关注是否连续同号偏差；
  - 连续偏正/偏负通常意味着结构性偏差。
- `holdout_diagnostics_dashboard.png`：
  - 一图看校准、残差时间线、绝对误差时间线、分位数对齐；
  - 用于判断“整体准不准 + 哪段最不稳”。

### 描述类

- `inventory_price_focus.png`、`weekly_core_panel.png`：观察核心四变量是否出现新分布形态。
- `descriptive_dashboard.png`：查看分布偏态、相关性是否明显漂移。

---

## 12. 阈值分级与决策规则（可执行）

以下阈值用于快速分级，避免“凭感觉”决策。建议以最近 3-5 次重训窗口滚动评估。

### A) 单次评估分级

- **A级（可直接使用）**
  - `方向准确率 >= 70%`
  - 且 `R2 > 0`
  - 且 MAE 未显著高于近三期中位数
- **B级（可用但需观察）**
  - `方向准确率 60%~70%`
  - 或 `R2` 约等于 0（`-0.05` 到 `0.05`）
  - 且无明显连续偏差
- **C级（需干预）**
  - `方向准确率 < 60%`
  - 或 `R2 < -0.05`
  - 或残差出现连续同向偏差 >= 3 个点

### B) 触发动作

- 满足任一条件即触发重训/调参：
  1. MAE 连续 2 期恶化（每期恶化幅度 > 8%）
  2. 方向准确率跌破 60%
  3. `residual_timeseries` 出现明显连续同向偏差
  4. `holdout_diagnostics_dashboard` 中绝对误差在近期段持续抬升

### C) 动作优先级（从轻到重）

1. 先执行 `python main.py plots`，确认是否只是解读偏差；
2. 再执行 `python main.py predict`，确认在线输出是否一致；
3. 执行 `python main.py train` 完整重训；
4. 若仍不达标，再提高最近窗口权重或缩短近期窗口长度重试。
