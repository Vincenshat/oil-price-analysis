# Oil Analysis ML/DL Pipeline  
# 石油分析机器学习/深度学习管线

## 1) Project Overview | 项目概览

**EN**  
This project builds a machine learning / deep learning pipeline for forecasting next-week:
- `Price_Change` (oil price change)
- `Stock_Change` (inventory/bbl change)

The pipeline includes:
- data cleaning and alignment
- feature engineering (lag/rolling/momentum/seasonality)
- walk-forward validation
- LSTM hyperparameter tuning
- model/package export for inference

**中文**  
本项目用于预测下一周：
- `Price_Change`（油价变化）
- `Stock_Change`（库存/桶数变化）

管线包含：
- 数据清洗与对齐
- 特征工程（滞后/滚动/动量/周期）
- walk-forward 时序回测
- LSTM 调参与筛选
- 模型与推理包导出

---

## 2) Main Files | 主要文件

- `main.py`: unified entrypoint / 统一入口（训练 + 预测）
- `train_lstm_pipeline.py`: full walk-forward training pipeline / 完整训练脚本
- `4.0_enriched.csv`: cleaned + enriched dataset / 清洗增强后的数据
- `outputs_lstm_wf/`: training artifacts and plots / 训练产物与过程图
- `outputs_lstm_wf/final_package/`: exported deploy package / 最终可部署模型包

---

## 3) Environment Setup | 环境准备

```bash
pip install -r requirements.txt
```

---

## 4) Run Training | 训练模型

```bash
python main.py train
```

**EN**  
This will run walk-forward folds + multiple hyperparameter trials and generate:
- metrics CSV/JSON
- fold predictions
- process plots
- final package

**中文**  
该命令会执行多折 walk-forward + 多组超参数测试，输出：
- 指标文件（CSV/JSON）
- 各折预测结果
- 过程图文件
- 最终模型包

---

## 5) Run Prediction | 执行预测

Default:
```bash
python main.py predict
```

Custom path:
```bash
python main.py predict --data "d:\Projects\oil_analysis\ML\4.0_enriched.csv" --package-dir "d:\Projects\oil_analysis\ML\outputs_lstm_wf\final_package"
```

Output file:
- `outputs_lstm_wf/final_package/predict_next_output.json`

---

## 6) Generated Process Plots | 已生成过程图

Inside `outputs_lstm_wf/`:
- `process_flow.png`
- `walkforward_splits.png`
- `fold_metrics.png`
- `last_fold_predictions.png`
- `feature_selection_top20.png`
- `residual_distribution.png`

---

## 7) Notes | 说明

**EN**
- Current pipeline is engineering-complete and reproducible.
- Model performance may still be constrained by sample size and signal strength.

**中文**
- 目前流程已经完整、可复现、可部署。
- 指标上限仍受样本规模和可预测信号强度限制。

