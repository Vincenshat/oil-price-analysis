# 石油价格与库存变化预测：主要发现（扩展解释版）

## 0. 阅读说明

本文件重点回答三件事：  
1) 数据本身“长什么样”，它透露了哪些结构信息；  
2) 这些结构信息如何影响模型效果；  
3) 在当前数据约束下，结果应该如何正确解读。  

每个章节都给出对应图，方便你在图文报告/PPT中一一对应。

---

## 1. 数据结构与样本画像（Data Portrait）

### 1.1 数据覆盖范围与密度

- 数据源文件：`4.0_enriched.csv`
- 样本行数：`199`
- 日期范围：`2021-01-29` 到 `2026-03-05`
- 唯一日期数：`133`
- 平均“每个日期的事件行数”：约 `1.5` 行

这意味着数据是“**周度市场数据 + 事件增强**”的混合结构，不是纯单变量时间序列。  
同一天可能出现多条事件记录，因此建模前必须做同日聚合和对齐，否则会把“事件条数差异”误当成“时间频率变化”。

**对应图：**
- `outputs_lstm_wf/process_flow.png`
- `outputs_lstm_wf/walkforward_splits.png`

### 1.2 事件类型构成（语义层）

事件类型分布呈明显“非均衡”：
- 能源库存类事件最多（主导样本）
- 宏观/政策/地缘类事件次之
- 说明数据生成机制天然偏向“供给与库存冲击”叙事

这会带来一个直接影响：  
模型更容易学到“库存链路”的局部规律，而对宏观冲击和制度性变化的泛化能力更依赖外部变量质量。

**对应图：**
- `outputs_lstm_wf/descriptive_dashboard.png`（可结合时间线和相关性面板理解）

### 1.3 缺失与可用性

清洗后核心建模列几乎完整：
- `WTI_T`、`Stock_T`、`Price_Change`、`Stock_Change` 缺失率均接近 `0`
- 宏观增强列（如 `dxy_broad`、`us10y_yield`）仍有轻微缺失（约 1%~3%），已在管线中插值处理

这说明当前问题的核心不再是“数据缺失”，而是“**信号强度与噪声比**”。

**对应图：**
- `outputs_lstm_wf/descriptive_dashboard.png`

---

## 2. 描述性统计解释（Descriptive Interpretation）

### 2.1 目标变量的分布特性

从分布看：
- `Price_Change` 中位数为正（约 `2.39`），但均值接近 `0`（约 `-0.065`）
- `Stock_Change` 中位数接近 `0`（约 `-60`），但标准差很高（约 `8876`）

解释：
- 油价变化在样本中呈“**小幅波动 + 偶发大冲击**”特征；
- 库存变化更像“**高噪声、厚尾分布**”，存在明显极端周；
- 这会导致模型优化时被少数大波动样本牵引，形成“平均误差可接受但R²偏弱”的典型表现。

**对应图：**
- `outputs_lstm_wf/descriptive_dashboard.png`（分布子图）

### 2.2 年度层面的波动差异（Regime Clue）

按年聚合可见：
- `2022` 年油价波动标准差明显高于相对平稳年份（地缘与供给冲击密集期）
- 库存变化每年波动水平都偏高，且极值跨度大

解释：
- 同一个模型在不同年份其实面对的是不同“机制区间”（regime）；
- 单模型统一拟合所有年份时，容易出现“某些年份表现好，跨区间就掉线”的现象。

**对应图：**
- `outputs_lstm_wf/descriptive_dashboard.png`
- `outputs_lstm_wf/fold_metrics.png`

### 2.3 相关性不是因果，但能揭示可学边界

周度相关性显示：
- `WTI_T` 与 `Stock_T` 呈显著负相关（`-0.721`）
- `Price_Change` 与 `Stock_Change` 轻度负相关（`-0.213`）
- 宏观三项（美元/长端利率/政策利率）彼此相关性较高，存在共线性

解释：
- 价格水平与库存水平关系清晰，但“下一周变化量”的可预测关系较弱；
- 宏观变量可能更多提供“方向约束”而非精确幅度解释；
- 这也是特征筛选和正则化必须做的原因。

**对应图：**
- `outputs_lstm_wf/descriptive_dashboard.png`（相关性热力图）
- `outputs_lstm_wf/feature_selection_top20.png`

---

## 3. 时间结构解释：为什么要 Walk-Forward

### 3.1 自相关弱，决定了不能只靠“惯性预测”

lag-1 自相关：
- `Price_Change` 约 `-0.007`（几乎无惯性）
- `Stock_Change` 约 `0.113`（弱惯性）

解释：
- 目标序列“短记忆”特征明显，仅靠前一期值难以稳定预测下一期；
- 模型必须依赖外生信息与组合特征，而非简单自回归。

### 3.2 Walk-forward 反映真实部署场景

使用 4 折 walk-forward 的意义是：
- 每一折都模拟“用历史预测未来”
- 避免随机打乱造成的信息泄露
- 把“时间漂移风险”显性化

你看到的折间波动（尤其库存）其实很有价值：  
它说明模型在不同时间区间的稳定性并不一致，这比单次 train/test 的“偶然高分”更可信。

**对应图：**
- `outputs_lstm_wf/walkforward_splits.png`
- `outputs_lstm_wf/fold_metrics.png`

---

## 4. 模型结果解释（不仅报指标，还解释含义）

### 4.1 跨折平均表现（主结论）

- 价格变化 `price_change_t1`
  - MAE `3.164`
  - RMSE `3.512`
  - R² `-0.064`
- 库存变化 `stock_change_t1`
  - MAE `6312.024`
  - RMSE `7647.964`
  - R² `-0.578`

如何解读：
- MAE/RMSE说明模型能给出一个“量级合理的近似”；
- 负 R² 说明它对方差解释度不足，无法稳定超越“简单基线”；
- 库存目标更难，体现为误差和方差解释都更差。

### 4.2 为什么会出现“MAE可看但R²偏弱”

常见于这类数据：
- 目标噪声大、峰值稀疏（极端周少但影响大）
- regime 切换（宏观状态/地缘状态变化）
- 样本总量不足以支撑高维特征稳定学习

因此，当前模型更适合“风险区间提示”和“相对变化参考”，不适合作为单点高置信交易信号。

**对应图：**
- `outputs_lstm_wf/fold_metrics.png`
- `outputs_lstm_wf/last_fold_predictions.png`
- `outputs_lstm_wf/residual_distribution.png`

---

## 5. 调参与最优参数的业务含义

### 5.1 全局最优参数（10组参数 × 4折）

最优 trial（`trial_id=7`）：
- `hidden_size=96`
- `num_layers=3`
- `dropout=0.3`
- `learning_rate=0.0006`
- `weight_decay=0.0003`
- `batch_size=24`

平均表现：
- `mean_test_price_mae = 3.1248`
- `mean_test_stock_mae = 6353.653`

### 5.2 参数背后的解释

- 更深网络（3层）说明特征关系并非线性单层可描述；
- 较高 dropout（0.3）和较强权重衰减说明过拟合风险真实存在；
- 较低学习率更稳，符合小样本+噪声场景。

这套参数组合本质上是“**稳健优先**”而不是“极致拟合优先”。

**对应图：**
- `outputs_lstm_wf/training_process_curve.png`
- `outputs_lstm_wf/fold_metrics.png`

---

## 6. 特征工程与可解释发现

### 6.1 哪类特征真正有贡献

从跨折入选频率看，稳定特征主要来自：
- 目标与核心变量的历史滞后项
- 滚动均值/滚动波动项
- 利率/美元等宏观约束项

这说明：
- “历史行为轨迹”是主信号；
- “宏观变量”是边际修正信号；
- “单次事件文本”在当前建模形态下对连续数值回归贡献有限。

### 6.2 为什么描述性图重要

`descriptive_dashboard` 不是装饰图，它直接告诉我们：
- 数据分布是否偏态/厚尾
- 是否存在结构变化年份
- 哪些变量共线显著，需在筛选中控制

**对应图：**
- `outputs_lstm_wf/feature_selection_top20.png`
- `outputs_lstm_wf/descriptive_dashboard.png`

---

## 7. 最终模型包与部署可用性

最终可复现包位置：`outputs_lstm_wf/final_package`

包含：
- `final_lstm_model.pt`
- `final_x_scaler.pkl`
- `final_y_scaler.pkl`
- `final_feature_columns.json`
- `final_holdout_predictions.csv`
- `final_package_summary.json`

保留集表现：
- `price_change_t1`：MAE `3.096`，RMSE `3.633`，R² `-1.895`
- `stock_change_t1`：MAE `8987.453`，RMSE `10462.089`，R² `-0.059`

解读要点：
- 保留集样本仅 20 周，统计方差很高；
- 极端周对 R² 影响非常大；
- 当前结果更适合“滚动监控 + 辅助决策”，而非一次性定型。

**对应图：**
- `outputs_lstm_wf/final_result_holdout.png`
- `outputs_lstm_wf/residual_distribution.png`

---

## 8. 结论分层（给汇报可直接引用）

### 8.1 可以明确说“做到了”的

- 数据清洗、对齐、增强、建模、回测、调参、导出全流程已闭环；
- 训练与推理可复现（有 `main.py` 与完整模型包）；
- 过程图、结果图、描述性图已完整生成。

### 8.2 需要谨慎解读的

- 负 R² 表明在幅度精确回归上仍不稳定；
- 库存变化目标对极端样本高度敏感；
- 现有数据规模与噪声结构限制了上限。

### 8.3 下一步最值得做的三件事

- 增加更长历史和更高频外生特征（供需拆分、期限结构、更细粒度宏观发布节奏）；
- 做两阶段策略（先方向分类，再幅度回归）；
- 加入模型融合（LSTM + 树模型）并继续 walk-forward 验证。

**对应图（结论页建议组合）：**
- 方法可信性：`outputs_lstm_wf/process_flow.png` + `outputs_lstm_wf/walkforward_splits.png`
- 模型稳定性：`outputs_lstm_wf/fold_metrics.png` + `outputs_lstm_wf/training_process_curve.png`
- 数据解释性：`outputs_lstm_wf/descriptive_dashboard.png` + `outputs_lstm_wf/feature_selection_top20.png`
- 最终落地效果：`outputs_lstm_wf/final_result_holdout.png` + `outputs_lstm_wf/residual_distribution.png`

