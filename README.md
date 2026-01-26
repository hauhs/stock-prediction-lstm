# 📈 基于Python和LSTM的股票预测系统

基于深度学习LSTM模型的股票价格预测与分析系统，采用Streamlit构建交互式Web界面。

## ✨ 功能特点

- **股票数据获取** - 支持美股、A股等多市场股票数据
- **LSTM预测模型** - 双层LSTM神经网络进行价格预测
- **技术指标分析** - MA、RSI、MACD等技术指标计算与可视化
- **风险评估** - 波动率、VaR、夏普比率等风险指标
- **智能投资建议** - 基于多维度分析生成投资建议
- **实验对比** - 支持不同参数配置的对比实验

## 🛠️ 技术栈

- **深度学习**: TensorFlow / Keras
- **数据处理**: Pandas, NumPy, scikit-learn
- **数据获取**: yfinance
- **可视化**: Plotly
- **Web框架**: Streamlit

## 📦 安装

```bash
# 克隆项目
git clone https://github.com/您的用户名/stock-prediction-lstm.git
cd stock-prediction-lstm

# 安装依赖
pip install -r requirements.txt
```

## 🚀 运行

```bash
streamlit run app.py
```

访问 http://localhost:8501 使用系统。

## 📁 项目结构

```
bishe/
├── app.py                 # 主应用入口
├── requirements.txt       # 项目依赖
├── data/
│   └── data_loader.py     # 数据获取与预处理
├── models/
│   └── lstm_model.py      # LSTM模型定义与训练
└── utils/
    └── visualization.py   # 可视化工具
```

## 📊 使用说明

1. 选择股票代码（如AAPL、GOOGL、000001.SS等）
2. 设置日期范围和模型参数
3. 点击"开始预测"进行分析
4. 查看预测结果、技术指标和投资建议

## ⚠️ 免责声明

本系统仅供学习研究参考，不构成投资建议。股票市场具有高风险性，请谨慎决策。

## 📄 许可证

MIT License
