# Ass1：数据抓取 + 分析可视化 + 组合优化（含 LightGBM）

本目录是 Ass1 的完整实现：从 AkShare 拉取 12 个标的（日度收盘价），生成统计与图表，并用（Naive/LightGBM）预测收益向量驱动 Monte Carlo 组合优化，最后用 Streamlit 提供交互式网页。

## 功能概览

- **数据抓取与清洗**：AkShare 日度 close，去重；在收益率上做 Z-score 异常点剔除。
- **训练/测试切分**：train=2024-11-01..2025-11-30；test=2025-12-01..2026-01-31（写入 data.json 的 meta）。
- **分析与图表导出**：年化指标、相关性热力图、收益分布 KDE、滚动波动率（HTML + CSV）、夏普比率图。
- **预测与优化**：
  - Naive：近 30 日均值预测。
  - LightGBM：滞后项 + 滚动统计特征，固定测试集评估。
  - 组合优化：Monte Carlo（最大夏普 / 最小波动），并给出“低/中/高”风险偏好对应权重。
- **交互网页 (Streamlit)**：
  - **动态投资指南**：基于选择的资产/风险偏好，自动生成双栏布局的深度分析文档（左栏科普，右栏技术原理）。
  - **图表展示**：相关性热力图、夏普比率前沿、收益/风险分布等。
  - **权重推荐**：根据优化结果动态生成的资金分配表。

## 目录与关键文件

- `fetch_data.py`：按标的下载数据并写入 `data.json`、生成 `read.txt`（按 universe 顺序输出月均价）。
- `analyze.py`：基于 `data.json` 导出 CSV + HTML 图表（`*_metrics.csv`、`*_corr_heatmap.html` 等）。
- `run_models.py`：基于 `data.json` 生成 `models.json`，并导出权重饼图 HTML。
- `app.py`：Streamlit 应用入口（读取 `data.json` / `models.json`），包含动态 HTML 报告生成逻辑。
- `requirements.txt`：项目依赖列表。
- `preview.md`：投资建议文案的原始 Markdown 模板（仅供参考，实际逻辑已内嵌至 `app.py`）。

## 快速开始（Windows / PowerShell）

1) 安装依赖：

```powershell
# 确保在 Ass1 目录下
python -m pip install -r requirements.txt
```

2) 下载数据并生成 `data.json` / `read.txt`（首次建议带 `--reset`）：

```powershell
python .\fetch_data.py --reset --symbol SPY
python .\fetch_data.py --symbol AU0

$symbols = "NVDA","MSFT","TSMC","GOOGL","AMZN","AAPL","ASML","META","AVGO","ORCL"
foreach ($s in $symbols) { python .\fetch_data.py --symbol $s }
```

3) 生成分析结果与模型结果：

```powershell
python .\analyze.py --dataset all
python .\run_models.py --json .\data.json
```

4) 启动网页：

```powershell
streamlit run .\app.py --server.port 8502
```

## 主要输出

- `data.json`：原始/清洗后的价格序列、月均价、异常点与日志、train/test 元信息。
- `read.txt`：按 12 标的顺序输出每月均价（用于作业提交/检查）。
- `models.json`：Naive/LightGBM 评估与预测、优化结果（权重、指标、风险偏好推荐权重等）。
- `*_prices.html` / `*_corr_heatmap.html` / `*_rolling_vol.html` / `*_returns_kde.html`：可直接打开的离线图表页面。
- `*_metrics.csv` / `*_corr.csv`：对应的指标与相关性矩阵。

## 导出与分享网页

- Streamlit 页面右上角 “⋮” → “Print/打印” → “另存为 PDF”。
- 图表页（`*.html`）本身就是离线页面，可直接发给他人打开（建议连同同目录资源一起打包）。

## 常见问题

- **HTTPS 请求报 SSL EOF**：当前环境可能存在 requests 访问 https 不稳定的情况；`fetch_data.py` 会在内部强制将 requests 的 https URL 降级为 http 以提高成功率。
- **AkShare 接口差异**：部分版本 `stock_us_daily` 参数与历史写法不同；本实现采用拉取后再在本地按日期过滤的方式处理。
