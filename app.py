# -*- coding: utf-8 -*-
"""
基于Python的股票预测和结果展示系统
主应用入口 - Streamlit网页应用
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_loader import (
    get_stock_data, preprocess_data, create_sequences,
    split_data, calculate_technical_indicators, calculate_risk_metrics,
    analyze_trend, generate_investment_advice, STOCK_MAPPING
)
from models.lstm_model import (
    build_lstm_model, train_model, predict,
    predict_future, evaluate_model
)
from utils.visualization import (
    create_stock_chart, create_prediction_chart,
    create_technical_chart, create_risk_gauge, create_returns_distribution,
    create_full_prediction_chart, get_chinese_config
)

# 页面配置
st.set_page_config(
    page_title="股票预测系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e88e5;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #424242;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        color: #155724;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        border-radius: 5px;
        padding: 1rem;
        color: #856404;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """主函数"""
    
    # 标题
    st.markdown('<h1 class="main-header">📈 股票预测系统</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">基于LSTM深度学习的股票价格预测与可视化分析平台</p>', unsafe_allow_html=True)
    
    # 侧边栏配置
    with st.sidebar:
        st.header("🧭 功能导航")
        menu = st.radio(
            "选择功能板块",
            ["🔮 股价预测与实验对比", "📥 历史数据一键导出"],
            index=0
        )
        st.divider()
        
    # 初始化变量以防作用域报错
    predict_btn = False
    stock_symbol = "AAPL"
    start_date = datetime.now() - timedelta(days=365*2)
    end_date = datetime.now()
    seq_length = 60
    epochs = 50
    future_days = 30
    
    if menu == "🔮 股价预测与实验对比":
        with st.sidebar:
            st.header("⚙️ 参数设置")
            
            # 股票选择
            st.subheader("📊 股票选择")
            
            # 预设股票列表
            preset_stocks = list(STOCK_MAPPING.keys())
            preset_display = [f"{code} - {name}" for code, name in STOCK_MAPPING.items()]
            
            use_preset = st.checkbox("使用预设股票", value=True)
            
            if use_preset:
                selected_display = st.selectbox(
                    "选择股票",
                    preset_display,
                    index=0
                )
                stock_symbol = selected_display.split(" - ")[0]
            else:
                stock_symbol = st.text_input(
                    "输入股票代码",
                    value="AAPL",
                    help="美股直接输入代码如AAPL，A股需加后缀如600519.SS(上海)或000001.SZ(深圳)"
                )
            
            st.divider()
            
            # 日期范围
            st.subheader("📅 日期范围")
            st.caption("提示：股票数据可追溯到1990年，但具体时间范围取决于数据源")
            
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "开始日期",
                    value=datetime.now() - timedelta(days=365*2),
                    min_value=datetime(1990, 1, 1),
                    max_value=datetime.now()
                )
            with col2:
                end_date = st.date_input(
                    "结束日期",
                    value=datetime.now(),
                    min_value=datetime(1990, 1, 1),
                    max_value=datetime.now()
                )
            
            st.divider()
            
            # 模型参数
            st.subheader("🧠 模型参数")
            
            seq_length = st.slider(
                "序列长度（天）",
                min_value=10,
                max_value=200,
                value=60,
                step=5,
                help="用于预测的历史数据天数，建议30-120天"
            )
            
            epochs = st.slider(
                "训练轮数",
                min_value=5,
                max_value=200,
                value=50,
                step=5,
                help="模型训练的迭代次数，越多越精确但耗时越长"
            )
            
            future_days = st.slider(
                "预测天数",
                min_value=1,
                max_value=120,
                value=30,
                step=1,
                help="预测未来多少天的股价，时间越长不确定性越大"
            )
            
            st.divider()
            
            # 开始预测按钮
            predict_btn = st.button("🚀 开始预测", type="primary", use_container_width=True)

    elif menu == "📥 历史数据一键导出":
        st.subheader("📥 股票历史数据查询与导出")
        st.caption("输入股票代码和日期区间，快速预览并以 CSV / Excel 格式下载历史行情数据")
        
        # 参数配置区域
        col_select, col_range = st.columns(2)
        with col_select:
            st.markdown("### 📊 1. 选择股票")
            use_preset_export = st.checkbox("使用预设股票", value=True, key="export_use_preset")
            if use_preset_export:
                preset_display_export = [f"{code} - {name}" for code, name in STOCK_MAPPING.items()]
                selected_display_export = st.selectbox(
                    "选择股票",
                    preset_display_export,
                    index=0,
                    key="export_preset_selectbox"
                )
                stock_symbol_export = selected_display_export.split(" - ")[0]
            else:
                stock_symbol_export = st.text_input(
                    "输入股票代码",
                    value="AAPL",
                    help="美股直接输入代码如AAPL，A股需加后缀如600519.SS(上海)或000001.SZ(深圳)",
                    key="export_custom_input"
                )
        with col_range:
            st.markdown("### 📅 2. 日期区间")
            col_start, col_end = st.columns(2)
            with col_start:
                start_date_export = st.date_input(
                    "开始日期",
                    value=datetime.now() - timedelta(days=365*2),
                    min_value=datetime(1990, 1, 1),
                    max_value=datetime.now(),
                    key="export_start_date"
                )
            with col_end:
                end_date_export = st.date_input(
                    "结束日期",
                    value=datetime.now(),
                    min_value=datetime(1990, 1, 1),
                    max_value=datetime.now(),
                    key="export_end_date"
                )
            
            include_indicators = st.checkbox(
                "同时计算并包含技术指标 (MA5, MA10, MA20, MACD, RSI, 波动率)",
                value=False,
                key="export_include_indicators",
                help="如果勾选，导出的数据中将包含计算好的技术指标，方便直接用于研究分析。"
            )
            
        st.divider()
        
        # 查询按钮
        if st.button("🔍 查询历史数据", type="primary", use_container_width=True):
            if start_date_export > end_date_export:
                st.error("❌ 开始日期不能晚于结束日期，请重新选择！")
            else:
                with st.spinner("📥 正在拉取数据，请稍候..."):
                    try:
                        df_export = get_stock_data(
                            stock_symbol_export,
                            start_date_export.strftime('%Y-%m-%d'),
                            end_date_export.strftime('%Y-%m-%d')
                        )
                        
                        if include_indicators:
                            df_export = calculate_technical_indicators(df_export)
                            
                        # 计算涨跌幅 (%) 并保留 2 位小数
                        df_export['Daily_Return'] = (df_export['Close'].pct_change() * 100).round(2)
                        
                        # 格式化日期列，去除时间部分的 00:00:00，转换为字符串
                        df_export['Date'] = pd.to_datetime(df_export['Date']).dt.strftime('%Y-%m-%d')
                        
                        # 重命名六个核心表头和新添加的涨跌幅表头为中文
                        rename_dict = {
                            'Date': '日期',
                            'Open': '开盘价',
                            'High': '最高价',
                            'Low': '最低价',
                            'Close': '收盘价',
                            'Volume': '成交量',
                            'Daily_Return': '涨跌幅(%)'
                        }
                        df_export = df_export.rename(columns=rename_dict)
                        
                        # 确保核心列位于最前，把涨跌幅加在核心 6 个字段的最后，之后再追加其他技术指标
                        core_cols = ['日期', '开盘价', '最高价', '最低价', '收盘价', '成交量', '涨跌幅(%)']
                        other_cols = [col for col in df_export.columns if col not in core_cols]
                        df_export = df_export[core_cols + other_cols]
                        
                        # 按日期进行倒序排列，使最新的日期排在最前面
                        df_export = df_export.sort_values(by='日期', ascending=False)
                            
                        st.session_state['export_df'] = df_export
                        st.session_state['export_symbol'] = stock_symbol_export
                        st.session_state['export_start'] = start_date_export
                        st.session_state['export_end'] = end_date_export
                        st.success(f"✅ 成功获取 {stock_symbol_export} 的历史数据，共 {len(df_export)} 条记录！")
                    except Exception as e:
                        st.error(f"❌ 数据获取失败: {str(e)}")
                        
        # 结果展示与导出区域
        if 'export_df' in st.session_state and st.session_state.get('export_symbol') == stock_symbol_export:
            df_export = st.session_state['export_df']
            
            # 显示数据基本信息
            st.markdown("### 📊 数据概览")
            meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
            with meta_col1:
                st.metric("总交易天数", f"{len(df_export)} 天")
            with meta_col2:
                # 此时 Close 已经被改名为 收盘价，所以这里应该读取 收盘价 列
                st.metric("期间最高收盘价", f"${df_export['收盘价'].max():.2f}")
            with meta_col3:
                st.metric("期间最低收盘价", f"${df_export['收盘价'].min():.2f}")
            with meta_col4:
                st.metric("均值收盘价", f"${df_export['收盘价'].mean():.2f}")
                
            # 提供下载按钮
            st.markdown("### 💾 导出数据")
            dl_col1, dl_col2, _ = st.columns([1, 1, 2])
            
            # 导出 CSV
            csv_data = df_export.to_csv(index=False).encode('utf-8-sig') # utf-8-sig 可以防止 Excel 打开时中文乱码
            with dl_col1:
                st.download_button(
                    label="📥 下载 CSV 格式文件",
                    data=csv_data,
                    file_name=f"{stock_symbol_export}_history_{start_date_export}_{end_date_export}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
            # 导出 Excel
            try:
                import io
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df_export.to_excel(writer, index=False, sheet_name='Stock_History')
                excel_data = buffer.getvalue()
                
                with dl_col2:
                    st.download_button(
                        label="📥 下载 Excel 格式文件",
                        data=excel_data,
                        file_name=f"{stock_symbol_export}_history_{start_date_export}_{end_date_export}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
            except Exception as e:
                with dl_col2:
                    st.error(f"Excel 导出组件加载失败: {str(e)}")
                    st.info("如有需要，请先下载 CSV 格式。")
            
            # 显示数据预览
            st.markdown("### 👁️ 数据预览 (最新 100 条)")
            st.dataframe(df_export.sort_values(by='日期', ascending=False).head(100), use_container_width=True)

    # 引导与指引主页面（当在预测界面但还没点击预测按钮时）
    if menu == "🔮 股价预测与实验对比" and not predict_btn:
        st.info("👈 请在左侧侧边栏选择股票并设置参数，然后点击 '🚀 开始预测' 按钮启动预测。")
        st.markdown("""
        ### 💡 系统简介
        本系统是一个基于 LSTM 深度学习网络的股票价格预测与分析平台，支持以下功能：
        1. **多维度预测**：使用 LSTM 模型预测未来股价走势。
        2. **K线图表展示**：提供交互式 K 线图、技术指标、历史价格对比图。
        3. **技术指标分析**：自动计算 MA、RSI、MACD 等主流技术指标。
        4. **风险量化分析**：计算年化波动率、最大回撤、VaR(95%) 风险值等。
        5. **智能投资建议**：根据技术指标与预测走势，生成综合评分与建议。
        6. **实验对比分析**：支持不同序列长度 and 训练轮数的对比实验。
        """)

    # 主内容区域
    if predict_btn:
        try:
            # 进度条
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 1. 获取数据
            status_text.text("📥 正在获取股票数据...")
            progress_bar.progress(10)
            
            df = get_stock_data(
                stock_symbol,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            
            if len(df) < seq_length + 20:
                st.error(f"❌ 数据不足！获取到 {len(df)} 条数据，至少需要 {seq_length + 20} 条数据进行训练。")
                return
            
            # 计算技术指标
            df = calculate_technical_indicators(df)
            
            progress_bar.progress(20)
            
            # 2. 数据预处理
            status_text.text("🔧 正在预处理数据...")
            
            scaled_data, scaler = preprocess_data(df)
            X, y = create_sequences(scaled_data, seq_length)
            X_train, X_test, y_train, y_test = split_data(X, y, train_ratio=0.8)
            
            # 保存数据到session_state，供模型对比使用
            st.session_state['X_train'] = X_train
            st.session_state['X_test'] = X_test
            st.session_state['y_train'] = y_train
            st.session_state['y_test'] = y_test
            st.session_state['scaler'] = scaler
            st.session_state['seq_length'] = seq_length
            
            progress_bar.progress(30)
            
            # 3. 构建并训练模型
            status_text.text("🧠 正在训练LSTM模型...")
            
            model = build_lstm_model(input_shape=(seq_length, 1))
            
            # 使用回调显示训练进度
            history = train_model(
                model, X_train, y_train,
                X_val=X_test, y_val=y_test,
                epochs=epochs, batch_size=32
            )
            
            progress_bar.progress(70)
            
            # 4. 预测
            status_text.text("📊 正在生成预测结果...")
            
            # 测试集预测
            test_predictions = predict(model, X_test)
            test_predictions = scaler.inverse_transform(test_predictions)
            y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
            
            # 未来预测
            last_sequence = scaled_data[-seq_length:].flatten()
            future_predictions = predict_future(model, last_sequence, scaler, future_days)
            
            # 生成未来日期
            last_date = pd.to_datetime(df['Date'].iloc[-1])
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=future_days,
                freq='B'  # 工作日
            )
            
            progress_bar.progress(90)
            
            # 5. 评估模型
            status_text.text("📈 正在计算评估指标...")
            
            metrics = evaluate_model(y_test_actual.flatten(), test_predictions.flatten())
            risk_metrics = calculate_risk_metrics(df)
            
            progress_bar.progress(100)
            status_text.text("✅ 预测完成！")
            
            # 存储到session_state
            st.session_state['df'] = df
            st.session_state['test_predictions'] = test_predictions
            st.session_state['y_test_actual'] = y_test_actual
            st.session_state['future_predictions'] = future_predictions
            st.session_state['future_dates'] = future_dates
            st.session_state['metrics'] = metrics
            st.session_state['risk_metrics'] = risk_metrics
            st.session_state['history'] = history
            st.session_state['stock_symbol'] = stock_symbol
            st.session_state['seq_length'] = seq_length
            
        except Exception as e:
            st.error(f"❌ 发生错误: {str(e)}")
            return
    
    # 显示结果
    if 'df' in st.session_state:
        display_results()


def display_results():
    """显示预测结果"""
    
    df = st.session_state['df']
    test_predictions = st.session_state['test_predictions']
    y_test_actual = st.session_state['y_test_actual']
    future_predictions = st.session_state['future_predictions']
    future_dates = st.session_state['future_dates']
    metrics = st.session_state['metrics']
    risk_metrics = st.session_state['risk_metrics']
    history = st.session_state['history']
    stock_symbol = st.session_state['stock_symbol']
    seq_length = st.session_state['seq_length']
    
    stock_name = STOCK_MAPPING.get(stock_symbol, stock_symbol)
    
    st.divider()
    
    # 概览指标
    st.markdown(f'<h2 class="sub-header">📊 {stock_name} 预测概览</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        current_price = df['Close'].iloc[-1]
        st.metric("当前价格", f"${current_price:.2f}")
    
    with col2:
        pred_price = future_predictions[-1]
        change = (pred_price - current_price) / current_price * 100
        st.metric("预测价格", f"${pred_price:.2f}", f"{change:+.2f}%")
    
    with col3:
        st.metric("模型R²", f"{metrics['R²']:.4f}")
    
    with col4:
        st.metric("RMSE", f"{metrics['RMSE']:.4f}")
    
    with col5:
        st.metric("MAPE", metrics['MAPE'])
    
    # 选项卡
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📈 价格预测", "📊 K线图表", "📉 技术指标", "⚠️ 风险分析", "🔧 模型详情", "💡 智能分析", "🔬 实验对比"
    ])
    
    with tab1:
        st.subheader("股票价格预测结果")
        st.caption("💡 提示：使用图表上方的时间范围按钮快速切换视图，或拖动底部滑块调整显示范围")
        
        # 构建预测图表数据 - 使用完整历史数据
        train_size = len(df) - len(y_test_actual) - seq_length
        test_dates = df['Date'].iloc[train_size + seq_length:].values
        
        fig = create_full_prediction_chart(
            df=df,
            test_dates=test_dates,
            y_test_actual=y_test_actual,
            test_predictions=test_predictions,
            future_dates=future_dates,
            future_predictions=future_predictions,
            seq_length=seq_length,
            title=f"{stock_name} 股票价格预测"
        )
        st.plotly_chart(fig, use_container_width=True, config=get_chinese_config())
        
        # 未来预测表格
        st.subheader("未来价格预测详情")
        future_df = pd.DataFrame({
            '日期': future_dates.strftime('%Y-%m-%d'),
            '预测价格': [f"${p:.2f}" for p in future_predictions],
            '涨跌幅': [f"{(p/df['Close'].iloc[-1]-1)*100:+.2f}%" for p in future_predictions]
        })
        st.dataframe(future_df, use_container_width=True, hide_index=True)
    
    with tab2:
        st.subheader("K线图与移动平均线")
        st.caption("📊 红色表示上涨，绿色表示下跌。图表显示选定日期范围内的完整数据")
        
        fig = create_stock_chart(df, title=f"{stock_name} K线图", show_volume=True)
        st.plotly_chart(fig, use_container_width=True, config=get_chinese_config())
    
    with tab3:
        st.subheader("技术指标分析")
        st.caption("📈 RSI > 70 为超买区域，RSI < 30 为超卖区域")
        
        fig = create_technical_chart(df)
        st.plotly_chart(fig, use_container_width=True, config=get_chinese_config())
        
        # RSI解读
        current_rsi = df['RSI'].iloc[-1]
        if current_rsi > 70:
            st.warning(f"⚠️ 当前RSI为 {current_rsi:.2f}，处于超买区域，可能面临回调风险")
        elif current_rsi < 30:
            st.success(f"✅ 当前RSI为 {current_rsi:.2f}，处于超卖区域，可能存在反弹机会")
        else:
            st.info(f"ℹ️ 当前RSI为 {current_rsi:.2f}，处于正常区域")
    
    with tab4:
        st.subheader("风险指标分析")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 风险指标卡片
            for key, value in risk_metrics.items():
                st.metric(key, value)
        
        with col2:
            # 收益率分布图
            if 'Daily_Return' in df.columns:
                returns = df['Daily_Return'].dropna()
                try:
                    fig = create_returns_distribution(returns)
                    st.plotly_chart(fig, use_container_width=True, config=get_chinese_config())
                except:
                    st.info("收益率分布图需要scipy库支持")
        
        # 风险提示
        st.warning("""
        ⚠️ **风险提示**
        
        1. 本系统仅供学习研究使用，不构成任何投资建议
        2. 股票市场存在风险，过往业绩不代表未来表现
        3. LSTM模型预测存在误差，请勿作为唯一投资依据
        4. 建议结合基本面分析和市场情况综合判断
        """)
    
    with tab5:
        st.subheader("模型训练详情")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**模型架构**")
            st.code("""
LSTM模型结构:
├── Input Layer (seq_length, 1)
├── LSTM Layer (50 units, return_sequences=True)
├── Dropout (0.2)
├── LSTM Layer (50 units)
├── Dropout (0.2)
├── Dense Layer (25 units, ReLU)
└── Output Layer (1 unit)
            """)
        
        with col2:
            st.markdown("**训练参数**")
            st.json({
                "序列长度": seq_length,
                "训练轮数": len(history.get('loss', [])),
                "批次大小": 32,
                "优化器": "Adam",
                "学习率": 0.001,
                "损失函数": "MSE"
            })
        
        # 训练曲线
        st.subheader("训练损失曲线")
        
        import plotly.graph_objects as go
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=history.get('loss', []),
            mode='lines',
            name='训练损失',
            line=dict(color='#2196f3')
        ))
        if 'val_loss' in history:
            fig.add_trace(go.Scatter(
                y=history['val_loss'],
                mode='lines',
                name='验证损失',
                line=dict(color='#ff9800')
            ))
        
        fig.update_layout(
            title=dict(text='模型训练过程', x=0.5),
            xaxis_title='训练轮次 (Epoch)',
            yaxis_title='损失值 (Loss)',
            template='plotly_dark',
            height=400,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        st.plotly_chart(fig, use_container_width=True, config=get_chinese_config())
        
        # 评估指标详情
        st.subheader("模型评估指标")
        metrics_df = pd.DataFrame([metrics])
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    with tab6:
        st.subheader("💡 智能趋势分析与投资建议")
        st.caption("⚠️ 以下分析仅供参考，不构成实际投资建议。投资有风险，入市需谨慎。")
        
        # 生成投资建议
        investment_advice = generate_investment_advice(df, future_predictions, risk_metrics)
        trend_info = investment_advice['趋势信息']
        
        # 使用Streamlit原生metric组件显示核心指标
        col1, col2, col3 = st.columns(3)
        
        with col1:
            score = investment_advice['综合评分']
            st.metric(
                label="📊 综合评分",
                value=f"{score} 分",
                delta="良好" if score >= 60 else "一般" if score >= 40 else "较差"
            )
        
        with col2:
            st.metric(
                label="💼 操作建议",
                value=investment_advice['操作建议'],
                delta="基于多维度分析"
            )
        
        with col3:
            st.metric(
                label="⚠️ 风险等级",
                value=investment_advice['风险等级'],
                delta="当前市场风险"
            )
        
        st.divider()
        
        # 趋势分析
        st.subheader("📊 趋势分析")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("当前趋势", trend_info['趋势'])
            st.metric("当前价格", f"${trend_info['当前价格']:.2f}")
            
        with col2:
            st.metric("1日涨跌", f"{trend_info['1日涨跌幅']:+.2f}%")
            st.metric("5日涨跌", f"{trend_info['5日涨跌幅']:+.2f}%")
            st.metric("20日涨跌", f"{trend_info['20日涨跌幅']:+.2f}%")
        
        st.divider()
        
        # 支撑位和压力位
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🟢 支撑位**")
            for level in trend_info.get('支撑位', []):
                st.success(level)
            if not trend_info.get('支撑位'):
                st.info("暂无支撑位")
        
        with col2:
            st.markdown("**🔴 压力位**")
            for level in trend_info.get('压力位', []):
                st.error(level)
            if not trend_info.get('压力位'):
                st.info("暂无压力位")
        
        st.divider()
        
        # 分析要点
        st.subheader("📝 分析要点")
        for point in investment_advice.get('分析要点', []):
            st.info(point)
        
        # 风险提示
        if investment_advice.get('注意事项'):
            st.subheader("⚠️ 风险提示")
            for warning in investment_advice['注意事项']:
                st.warning(warning)
        
        st.divider()
        
        # 免责声明
        st.warning("""
        **⚠️ 免责声明**  
        本系统提供的所有分析和预测仅供学习研究参考，不构成任何投资建议。
        股票市场具有高风险性，历史表现不代表未来收益。请投资者根据自身风险承受能力谨慎决策。
        """)
    
    with tab7:
        st.subheader("🔬 实验对比分析")
        st.caption("对比不同参数配置下的LSTM预测效果，为毕业设计提供实验数据支撑")
        
        st.markdown("### 📝 实验设置")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**序列长度对比**")
            seq_options = st.multiselect(
                "选择要对比的序列长度",
                options=[30, 45, 60, 90, 120],
                default=[30, 60, 90],
                help="序列长度表示用多少天的历史数据预测下一天"
            )
        
        with col2:
            st.markdown("**训练轮数对比**")
            epoch_options = st.multiselect(
                "选择要对比的训练轮数",
                options=[20, 30, 50, 80, 100],
                default=[30, 50],
                help="训练轮数越多，模型学习越充分，但耗时也越长"
            )
        
        experiment_type = st.radio(
            "选择实验类型",
            ["序列长度对比", "训练轮数对比"],
            horizontal=True
        )
        
        if st.button("🚀 开始实验", type="primary", key="start_experiment"):
            if 'df' not in st.session_state:
                st.error("请先进行股票预测，然后再进行实验对比")
                st.stop()
            
            df_exp = st.session_state['df']
            results = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            if experiment_type == "序列长度对比":
                if len(seq_options) < 2:
                    st.warning("请至少选择2个序列长度进行对比")
                    st.stop()
                
                fixed_epochs = 50
                total = len(seq_options)
                
                for i, seq_len in enumerate(seq_options):
                    status_text.text(f"正在训练 序列长度={seq_len}...")
                    
                    try:
                        # 数据预处理
                        from data.data_loader import preprocess_data, create_sequences, split_data
                        scaled_data, scaler = preprocess_data(df_exp)
                        X, y = create_sequences(scaled_data, seq_len)
                        X_train, X_test, y_train, y_test = split_data(X, y, train_ratio=0.8)
                        
                        # 构建和训练模型（禁用早停以确保训练完整轮数）
                        model = build_lstm_model(input_shape=(seq_len, 1))
                        train_model(model, X_train, y_train, X_test, y_test, epochs=fixed_epochs, batch_size=32, use_early_stopping=False)
                        
                        # 预测和评估
                        preds = predict(model, X_test)
                        preds_inv = scaler.inverse_transform(preds)
                        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
                        metrics = evaluate_model(y_test_inv, preds_inv)
                        
                        results.append({
                            '参数': f'序列长度={seq_len}',
                            'R²': metrics['R²'],
                            'RMSE': metrics['RMSE'],
                            'MAE': metrics['MAE'],
                            'MAPE': metrics['MAPE']
                        })
                    except Exception as e:
                        st.error(f"序列长度={seq_len} 训练失败: {str(e)}")
                    
                    progress_bar.progress((i + 1) / total)
            
            else:  # 训练轮数对比
                if len(epoch_options) < 2:
                    st.warning("请至少选择2个训练轮数进行对比")
                    st.stop()
                
                fixed_seq = 60
                total = len(epoch_options)
                
                for i, epochs_num in enumerate(epoch_options):
                    status_text.text(f"正在训练 轮数={epochs_num}...")
                    
                    try:
                        from data.data_loader import preprocess_data, create_sequences, split_data
                        scaled_data, scaler = preprocess_data(df_exp)
                        X, y = create_sequences(scaled_data, fixed_seq)
                        X_train, X_test, y_train, y_test = split_data(X, y, train_ratio=0.8)
                        
                        model = build_lstm_model(input_shape=(fixed_seq, 1))
                        train_model(model, X_train, y_train, X_test, y_test, epochs=epochs_num, batch_size=32, use_early_stopping=False)
                        
                        preds = predict(model, X_test)
                        preds_inv = scaler.inverse_transform(preds)
                        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
                        metrics = evaluate_model(y_test_inv, preds_inv)
                        
                        results.append({
                            '参数': f'训练轮数={epochs_num}',
                            'R²': metrics['R²'],
                            'RMSE': metrics['RMSE'],
                            'MAE': metrics['MAE'],
                            'MAPE': metrics['MAPE']
                        })
                    except Exception as e:
                        st.error(f"轮数={epochs_num} 训练失败: {str(e)}")
                    
                    progress_bar.progress((i + 1) / total)
            
            status_text.text("实验完成！")
            
            if results:
                # 显示结果表格
                st.markdown("### 📊 实验结果")
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)
                
                # 绘制对比图
                st.markdown("### 📈 指标对比图")
                
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                
                fig = make_subplots(rows=1, cols=2, subplot_titles=('R² 决定系数', 'RMSE 均方根误差'))
                
                params = [r['参数'] for r in results]
                r2_values = [r['R²'] for r in results]
                rmse_values = [r['RMSE'] for r in results]
                
                fig.add_trace(
                    go.Bar(x=params, y=r2_values, marker_color='#2196f3', text=[f'{v:.4f}' for v in r2_values], textposition='outside'),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Bar(x=params, y=rmse_values, marker_color='#ff9800', text=[f'{v:.2f}' for v in rmse_values], textposition='outside'),
                    row=1, col=2
                )
                
                fig.update_layout(template='plotly_dark', height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True, config=get_chinese_config())
                
                # 结论
                st.markdown("### 🏆 实验结论")
                best_idx = max(range(len(results)), key=lambda i: results[i]['R²'])
                best_result = results[best_idx]
                
                st.success(f"""
                **最优参数配置**: {best_result['参数']}
                
                - R² 决定系数: {best_result['R²']:.4f}
                - RMSE: {best_result['RMSE']:.4f}
                - MAPE: {best_result['MAPE']}
                
                该配置在本次实验中表现最佳，建议在实际预测中使用。
                """)


if __name__ == "__main__":
    main()
