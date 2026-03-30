# Oil Analysis 项目总览

这是一个融合 `ML` 与 `NLP` 的油价分析项目，并提供统一的 Streamlit 学术分析前端（支持中英双语切换）：

- `ML/`：时序特征工程 + LSTM 预测（下一期价格变化、库存变化）
- `NLP/`：中文事件文本分词 + Word2Vec 向量建模
- `streamlit_app.py`：统一学术分析平台（中英双语、图文解释、可审计流程）

---

## 1. 目录说明

- `ML/main.py`：ML 入口（`train` / `predict` / `plots`）
- `ML/outputs_lstm_wf/`：训练指标、预测结果、最终模型包
- `NLP/oil_price_nlp.py`：NLP 训练与向量导出
- `NLP/outputs/text_vectors.csv`：NLP 语义向量输出
- `streamlit_app.py`：前端应用
- `requirements.txt`：统一依赖清单

---

## 2. 环境安装

```bash
pip install -r requirements.txt
```

---

## 3. 启动方式

### 启动交互前端（推荐）

```bash
streamlit run streamlit_app.py
```

启动后可在右上角 `Language / 语言` 切换 `中文` / `English`。

### 单独运行 ML 推理

```bash
python ML/main.py predict
```

### 单独运行 NLP 向量处理

```bash
python NLP/oil_price_nlp.py
```

---

## 4. 前端功能说明

### 研究总览
- 展示核心 KPI：价格/库存 MAE 与最新预测值
- 展示真实值 vs 预测值
- 展示各折误差对比

### ML 分析
- 价格与库存目标切换
- 残差分布可视化
- 各折指标对比
- 训练损失曲线展示

### NLP 洞察
- 文本向量 PCA 投影
- 单维向量分布分析
- NLP 维度与价格变化相关性 Top 分析

### 预测模块（ML/NLP 分离）
- **ML 数值输入预测**：上传数值 CSV 调用 `ML/main.py predict`
- **ML 场景预测**：关键变量滑杆输入 + 近邻历史估计
- **NLP 文本输入预测**：基于关键词语义信号输出方向与分数

### 数据审计
- 数据完整性、日期顺序、异常值比例检查
- 评估合理性检查（R2 与基线比较）
- 可直接触发完整重训

### 算法原理
- ML（LSTM + Walk-forward）流程说明
- NLP（Word2Vec + PCA）流程说明
- 近邻场景预测、异常监控方法说明

---

## 5. 界面与交互

- 中英双语切换（页面标题、导航、说明文本）
- 图表与解释并排展示（每个关键图旁均有 Markdown 解读）
- 图表支持缩放、框选、拖拽、导出

---

## 6. 数据与结果文件

- `ML/outputs_lstm_wf/final_package/predict_next_output.json`：最新预测结果
- `ML/outputs_lstm_wf/final_package/final_holdout_predictions.csv`：holdout 结果
- `ML/outputs_lstm_wf/walkforward_fold_metrics.csv`：分折指标
- `NLP/outputs/text_vectors.csv`：文本向量矩阵

---

## 7. 备注

- 上传 CSV 预测依赖与训练数据兼容的字段结构。
- 手动场景预测属于快速推演，不能替代正式模型输出。
- 文本 Insight 用于辅助判断，建议与量化预测联合解读。

