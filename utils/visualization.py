# -*- coding: utf-8 -*-
"""
可视化工具模块
使用Plotly创建交互式图表 - 完整中文本地化版本
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Plotly中文配置 - 工具栏按钮翻译
PLOTLY_CONFIG_CN = {
    'locale': 'zh-CN',
    'displaylogo': False,
    'scrollZoom': True,  # 启用鼠标滚轮缩放
    'modeBarButtonsToRemove': [],
    'toImageButtonOptions': {
        'format': 'png',
        'filename': '股票图表',
        'height': 600,
        'width': 1200,
        'scale': 2
    }
}

# 中文工具栏提示
MODEBAR_BUTTONS_CN = {
    'toImage': '下载图片',
    'zoom2d': '缩放',
    'pan2d': '平移',
    'select2d': '框选',
    'lasso2d': '套索选择',
    'zoomIn2d': '放大',
    'zoomOut2d': '缩小',
    'autoScale2d': '自动缩放',
    'resetScale2d': '重置',
    'hoverClosestCartesian': '显示最近数据',
    'hoverCompareCartesian': '比较数据'
}


def get_chinese_config():
    """获取Plotly中文配置"""
    return PLOTLY_CONFIG_CN


def create_stock_chart(df: pd.DataFrame, 
                       title: str = "股票价格走势",
                       show_volume: bool = True) -> go.Figure:
    """
    创建股票K线图
    
    参数:
        df: 股票数据DataFrame
        title: 图表标题
        show_volume: 是否显示成交量
    
    返回:
        go.Figure: Plotly图表对象
    """
    if show_volume:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.15,  # 增加子图间距避免重叠
            row_heights=[0.75, 0.25],  # 调整比例
            subplot_titles=('价格走势', '成交量')
        )
    else:
        fig = go.Figure()
    
    # K线图
    fig.add_trace(
        go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='K线',
            increasing_line_color='#ef5350',  # 红色上涨
            decreasing_line_color='#26a69a',  # 绿色下跌
            increasing_fillcolor='#ef5350',
            decreasing_fillcolor='#26a69a'
        ),
        row=1 if show_volume else None,
        col=1 if show_volume else None
    )
    
    # 添加移动平均线
    if 'MA5' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['Date'], y=df['MA5'],
                mode='lines', name='5日均线',
                line=dict(color='#ff9800', width=1),
                hovertemplate='5日均线: %{y:.2f}<extra></extra>'
            ),
            row=1 if show_volume else None,
            col=1 if show_volume else None
        )
    
    if 'MA10' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['Date'], y=df['MA10'],
                mode='lines', name='10日均线',
                line=dict(color='#2196f3', width=1),
                hovertemplate='10日均线: %{y:.2f}<extra></extra>'
            ),
            row=1 if show_volume else None,
            col=1 if show_volume else None
        )
    
    if 'MA20' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['Date'], y=df['MA20'],
                mode='lines', name='20日均线',
                line=dict(color='#9c27b0', width=1),
                hovertemplate='20日均线: %{y:.2f}<extra></extra>'
            ),
            row=1 if show_volume else None,
            col=1 if show_volume else None
        )
    
    # 成交量
    if show_volume and 'Volume' in df.columns:
        colors = ['#ef5350' if row['Close'] >= row['Open'] else '#26a69a' 
                  for _, row in df.iterrows()]
        
        fig.add_trace(
            go.Bar(
                x=df['Date'],
                y=df['Volume'],
                name='成交量',
                marker_color=colors,
                showlegend=False,
                hovertemplate='成交量: %{y:,.0f}<extra></extra>'
            ),
            row=2, col=1
        )
    
    # 设置x轴范围为完整数据范围
    x_range = [df['Date'].min(), df['Date'].max()]
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        height=650,  # 增加高度
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12)
        ),
        hovermode='x unified',
        # 设置x轴完整显示
        xaxis=dict(
            range=x_range,
            title='',
            tickformat='%Y-%m-%d',
            tickangle=45
        ),
        yaxis=dict(title='价格', domain=[0.32, 1]),  # 调整domain避免重叠
        # 第二个子图的设置
        xaxis2=dict(
            range=x_range,
            title='日期',
            tickformat='%Y-%m-%d',
            tickangle=45
        ) if show_volume else None,
        yaxis2=dict(title='成交量', domain=[0, 0.22]) if show_volume else None,  # 调整domain
        # 流畅缩放配置
        uirevision='constant',  # 保持UI状态
        dragmode='zoom',
        modebar=dict(
            bgcolor='rgba(0,0,0,0.5)',
            orientation='h'
        )
    )
    
    # 添加缩放按钮配置
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1个月", step="month", stepmode="backward"),
                dict(count=3, label="3个月", step="month", stepmode="backward"),
                dict(count=6, label="6个月", step="month", stepmode="backward"),
                dict(count=1, label="1年", step="year", stepmode="backward"),
                dict(step="all", label="全部")
            ]),
            bgcolor='rgba(50,50,50,0.8)',
            font=dict(color='white', size=10),
            activecolor='#2196f3',
            y=1.0,
            x=0
        ),
        row=1, col=1
    )
    
    return fig


def create_prediction_chart(dates: list,
                           actual: np.ndarray,
                           predicted: np.ndarray,
                           future_dates: list = None,
                           future_predicted: np.ndarray = None,
                           title: str = "股票价格预测",
                           full_history_dates: list = None,
                           full_history_prices: np.ndarray = None) -> go.Figure:
    """
    创建预测结果图表
    
    参数:
        dates: 测试集日期列表
        actual: 测试集实际价格
        predicted: 测试集预测价格
        future_dates: 未来日期列表
        future_predicted: 未来预测价格
        title: 图表标题
        full_history_dates: 完整历史日期
        full_history_prices: 完整历史价格
    
    返回:
        go.Figure: Plotly图表对象
    """
    fig = go.Figure()
    
    # 如果提供了完整历史数据，先显示历史价格
    if full_history_dates is not None and full_history_prices is not None:
        fig.add_trace(
            go.Scatter(
                x=full_history_dates,
                y=full_history_prices,
                mode='lines',
                name='历史价格',
                line=dict(color='#78909c', width=1.5),
                hovertemplate='日期: %{x}<br>价格: %{y:.2f}<extra>历史价格</extra>'
            )
        )
    
    # 实际价格（测试集）
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=actual,
            mode='lines',
            name='实际价格',
            line=dict(color='#2196f3', width=2),
            hovertemplate='日期: %{x}<br>实际价格: %{y:.2f}<extra></extra>'
        )
    )
    
    # 预测价格（测试集）
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=predicted,
            mode='lines',
            name='模型预测',
            line=dict(color='#ff9800', width=2, dash='dash'),
            hovertemplate='日期: %{x}<br>预测价格: %{y:.2f}<extra></extra>'
        )
    )
    
    # 未来预测
    if future_dates is not None and future_predicted is not None:
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=future_predicted,
                mode='lines+markers',
                name='未来预测',
                line=dict(color='#4caf50', width=2),
                marker=dict(size=6),
                hovertemplate='日期: %{x}<br>预测价格: %{y:.2f}<extra>未来预测</extra>'
            )
        )
        
        # 添加预测区域阴影
        fig.add_vrect(
            x0=future_dates[0],
            x1=future_dates[-1],
            fillcolor="rgba(76, 175, 80, 0.1)",
            layer="below",
            line_width=0,
            annotation_text="预测区间",
            annotation_position="top left",
            annotation_font_size=12,
            annotation_font_color="#4caf50"
        )
    
    # 计算x轴范围，确保显示完整数据
    all_dates = list(dates) if not isinstance(dates, list) else dates
    if full_history_dates is not None:
        all_dates = list(full_history_dates) + list(all_dates)
    if future_dates is not None:
        all_dates = list(all_dates) + list(future_dates)
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        xaxis_title='日期',
        yaxis_title='价格 (美元)',
        template='plotly_dark',
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12)
        ),
        hovermode='x unified',
        xaxis=dict(
            tickformat='%Y-%m-%d',
            tickangle=45,
            dtick='M1'
        )
    )
    
    return fig


def create_technical_chart(df: pd.DataFrame) -> go.Figure:
    """
    创建技术指标图表 (RSI, MACD)
    
    参数:
        df: 包含技术指标的数据
    
    返回:
        go.Figure: Plotly图表对象
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('RSI 相对强弱指标', 'MACD 指数平滑异同移动平均线')
    )
    
    # 设置x轴范围
    x_range = [df['Date'].min(), df['Date'].max()]
    
    # RSI
    if 'RSI' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['Date'], y=df['RSI'],
                mode='lines', name='RSI',
                line=dict(color='#9c27b0', width=1.5),
                hovertemplate='日期: %{x}<br>RSI: %{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 超买超卖线
        fig.add_hline(y=70, line_dash="dash", line_color="red", 
                      annotation_text="超买线 (70)", annotation_position="right",
                      row=1, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green",
                      annotation_text="超卖线 (30)", annotation_position="right",
                      row=1, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=1, col=1)
    
    # MACD
    if 'MACD' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['Date'], y=df['MACD'],
                mode='lines', name='MACD线',
                line=dict(color='#2196f3', width=1.5),
                hovertemplate='日期: %{x}<br>MACD: %{y:.4f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        if 'Signal_Line' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['Date'], y=df['Signal_Line'],
                    mode='lines', name='信号线',
                    line=dict(color='#ff9800', width=1.5),
                    hovertemplate='日期: %{x}<br>信号线: %{y:.4f}<extra></extra>'
                ),
                row=2, col=1
            )
            
            # MACD柱状图
            macd_hist = df['MACD'] - df['Signal_Line']
            colors = ['#ef5350' if val >= 0 else '#26a69a' for val in macd_hist]
            
            fig.add_trace(
                go.Bar(
                    x=df['Date'], y=macd_hist,
                    name='MACD柱状图',
                    marker_color=colors,
                    showlegend=True,
                    hovertemplate='日期: %{x}<br>柱状值: %{y:.4f}<extra></extra>'
                ),
                row=2, col=1
            )
    
    fig.update_layout(
        template='plotly_dark',
        height=550,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11)
        ),
        hovermode='x unified',
        xaxis=dict(range=x_range),
        xaxis2=dict(
            range=x_range,
            title='日期',
            tickformat='%Y-%m-%d'
        ),
        yaxis=dict(title='RSI值'),
        yaxis2=dict(title='MACD值')
    )
    
    return fig


def create_risk_gauge(value: float, title: str, 
                      min_val: float = 0, max_val: float = 100) -> go.Figure:
    """
    创建风险仪表盘
    
    参数:
        value: 当前值
        title: 标题
        min_val: 最小值
        max_val: 最大值
    
    返回:
        go.Figure: Plotly图表对象
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 16}},
        gauge={
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': "#2196f3"},
            'steps': [
                {'range': [min_val, max_val * 0.33], 'color': "#4caf50"},
                {'range': [max_val * 0.33, max_val * 0.66], 'color': "#ff9800"},
                {'range': [max_val * 0.66, max_val], 'color': "#ef5350"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    fig.update_layout(
        template='plotly_dark',
        height=250,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def create_returns_distribution(returns: pd.Series) -> go.Figure:
    """
    创建收益率分布图
    
    参数:
        returns: 收益率序列
    
    返回:
        go.Figure: Plotly图表对象
    """
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=50,
        name='日收益率',
        marker_color='#2196f3',
        opacity=0.75,
        hovertemplate='收益率区间: %{x}<br>频数: %{y}<extra></extra>'
    ))
    
    # 添加正态分布曲线
    x_range = np.linspace(returns.min(), returns.max(), 100)
    mean = returns.mean()
    std = returns.std()
    
    try:
        from scipy import stats
        y_norm = stats.norm.pdf(x_range, mean, std) * len(returns) * (returns.max() - returns.min()) / 50
        
        fig.add_trace(go.Scatter(
            x=x_range,
            y=y_norm,
            mode='lines',
            name='正态分布曲线',
            line=dict(color='#ff9800', width=2)
        ))
    except ImportError:
        pass  # scipy未安装时跳过
    
    fig.update_layout(
        title=dict(text='日收益率分布', x=0.5),
        xaxis_title='收益率',
        yaxis_title='频数',
        template='plotly_dark',
        height=350,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def create_full_prediction_chart(df: pd.DataFrame,
                                  test_dates,
                                  y_test_actual: np.ndarray,
                                  test_predictions: np.ndarray,
                                  future_dates,
                                  future_predictions: np.ndarray,
                                  seq_length: int,
                                  title: str = "股票价格预测") -> go.Figure:
    """
    创建包含完整历史数据的预测图表
    
    参数:
        df: 完整股票数据DataFrame
        test_dates: 测试集日期
        y_test_actual: 测试集实际价格
        test_predictions: 测试集预测价格
        future_dates: 未来日期
        future_predictions: 未来预测价格
        seq_length: 序列长度
        title: 图表标题
    
    返回:
        go.Figure: Plotly图表对象
    """
    fig = go.Figure()
    
    # 计算分界点
    train_size = len(df) - len(y_test_actual) - seq_length
    
    # 1. 显示训练集历史数据
    train_dates = df['Date'].iloc[:train_size + seq_length].values
    train_prices = df['Close'].iloc[:train_size + seq_length].values
    
    fig.add_trace(
        go.Scatter(
            x=train_dates,
            y=train_prices,
            mode='lines',
            name='历史数据（训练集）',
            line=dict(color='#78909c', width=1.5),
            hovertemplate='日期: %{x|%Y-%m-%d}<br>收盘价: $%{y:.2f}<extra>训练数据</extra>'
        )
    )
    
    # 1.5 标注序列区间（测试前的 seq_length 天，即用于预测测试集第一天的输入数据）
    # 序列区间位于训练集末尾，从 (train_size) 到 (train_size + seq_length - 1)
    if train_size > 0 and seq_length > 0 and (train_size + seq_length) <= len(df):
        # 序列区间的起始和结束位置
        seq_start_idx = train_size
        seq_end_idx = train_size + seq_length - 1
        
        seq_start_date = pd.to_datetime(df['Date'].iloc[seq_start_idx]).strftime('%Y-%m-%d')
        seq_end_date = pd.to_datetime(df['Date'].iloc[seq_end_idx]).strftime('%Y-%m-%d')
        
        # 添加序列区间阴影
        fig.add_vrect(
            x0=seq_start_date,
            x1=seq_end_date,
            fillcolor="rgba(156, 39, 176, 0.15)",
            layer="below",
            line_width=1,
            line_color="rgba(156, 39, 176, 0.5)",
            line_dash="dash"
        )
        
        # 添加序列区间标注
        fig.add_annotation(
            x=seq_start_date,
            y=0.95,
            yref="paper",
            text=f"输入序列 ({seq_length}天)",
            showarrow=False,
            font=dict(size=10, color="#9c27b0"),
            bgcolor="rgba(156, 39, 176, 0.3)",
            bordercolor="#9c27b0",
            borderwidth=1
        )
        
    # 1.6 添加训练集区间标识（标注在训练集中间位置）
    if train_size > 0:
        # 训练集区间：从第0天到train_size+seq_length-1天（灰色线条区域）
        train_display_len = train_size + seq_length
        train_mid_idx = train_display_len // 2
        if train_mid_idx < len(df):
            fig.add_annotation(
                x=df['Date'].iloc[train_mid_idx],
                y=0.05,
                yref="paper",
                text=f"训练数据 ({train_display_len}天)",
                showarrow=False,
                font=dict(size=9, color="#78909c"),
                bgcolor="rgba(120, 144, 156, 0.3)"
            )
    
    # 添加测试集区间标识（标注在测试集中间位置，即黄色分界线之后）
    test_len = len(y_test_actual)
    if test_len > 0 and len(test_dates) > 0:
        # 测试集标注位置在测试数据的中间
        test_mid_idx = min(test_len // 2, len(test_dates) - 1)
        # 转换日期格式以确保正确显示
        test_mid_date = pd.to_datetime(test_dates[test_mid_idx]).strftime('%Y-%m-%d')
        fig.add_annotation(
            x=test_mid_date,
            y=0.05,
            yref="paper",
            text=f"测试数据 ({test_len}天)",
            showarrow=False,
            font=dict(size=9, color="#2196f3"),
            bgcolor="rgba(33, 150, 243, 0.3)"
        )
    
    # 2. 显示测试集实际价格
    fig.add_trace(
        go.Scatter(
            x=test_dates,
            y=y_test_actual.flatten(),
            mode='lines',
            name='实际价格（测试集）',
            line=dict(color='#2196f3', width=2),
            hovertemplate='日期: %{x|%Y-%m-%d}<br>实际价格: $%{y:.2f}<extra>测试数据</extra>'
        )
    )
    
    # 3. 显示测试集预测价格
    fig.add_trace(
        go.Scatter(
            x=test_dates,
            y=test_predictions.flatten(),
            mode='lines',
            name='模型预测（测试集）',
            line=dict(color='#ff9800', width=2, dash='dash'),
            hovertemplate='日期: %{x|%Y-%m-%d}<br>预测价格: $%{y:.2f}<extra>模型预测</extra>'
        )
    )
    
    # 4. 显示未来预测
    if future_dates is not None and future_predictions is not None:
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=future_predictions,
                mode='lines+markers',
                name='未来预测',
                line=dict(color='#4caf50', width=2),
                marker=dict(size=5, symbol='circle'),
                hovertemplate='日期: %{x|%Y-%m-%d}<br>预测价格: $%{y:.2f}<extra>未来预测</extra>'
            )
        )
        
        # 添加预测区域阴影
        fig.add_vrect(
            x0=future_dates[0],
            x1=future_dates[-1],
            fillcolor="rgba(76, 175, 80, 0.1)",
            layer="below",
            line_width=0,
            annotation_text="预测区间",
            annotation_position="top left",
            annotation_font_size=11,
            annotation_font_color="#4caf50"
        )
    
    # 添加训练/测试分界线
    if len(test_dates) > 0:
        # 将numpy datetime64转换为字符串以兼容Plotly
        split_date = pd.to_datetime(test_dates[0]).strftime('%Y-%m-%d')
        # 使用add_shape添加竖线（避免add_vline的annotation类型问题）
        fig.add_shape(
            type="line",
            x0=split_date, x1=split_date,
            y0=0, y1=1,
            yref="paper",
            line=dict(color="yellow", width=1, dash="dot")
        )
        # 单独添加标注
        fig.add_annotation(
            x=split_date,
            y=1.05,
            yref="paper",
            text="训练/测试分界",
            showarrow=False,
            font=dict(size=10, color="yellow"),
            bgcolor="rgba(0,0,0,0.5)"
        )
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        xaxis_title='日期',
        yaxis_title='价格 (美元)',
        template='plotly_dark',
        height=600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=11)
        ),
        hovermode='x unified',
        # 启用流畅动画和交互
        transition=dict(duration=300, easing='cubic-in-out'),
        uirevision='constant',
        dragmode='zoom',
        xaxis=dict(
            tickformat='%Y-%m-%d',
            tickangle=45,
            rangeslider=dict(visible=True, thickness=0.05),
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1个月", step="month", stepmode="backward"),
                    dict(count=3, label="3个月", step="month", stepmode="backward"),
                    dict(count=6, label="6个月", step="month", stepmode="backward"),
                    dict(count=1, label="1年", step="year", stepmode="backward"),
                    dict(step="all", label="全部")
                ]),
                bgcolor='rgba(50,50,50,0.8)',
                font=dict(color='white', size=10),
                activecolor='#2196f3',
                y=1.0,
                x=0
            )
        )
    )
    
    return fig


def create_model_comparison_chart(dates, actual_values, predictions_dict, title="模型预测对比"):
    """
    创建多模型预测对比图
    
    参数:
        dates: 日期数组
        actual_values: 实际值数组
        predictions_dict: 字典 {模型名称: 预测值数组}
        title: 图表标题
    
    返回:
        go.Figure: Plotly图表
    """
    # 颜色列表
    colors = ['#2196f3', '#4caf50', '#ff9800', '#e91e63', '#9c27b0', '#00bcd4']
    
    fig = go.Figure()
    
    # 添加实际值 - 使用醒目的红色
    fig.add_trace(go.Scatter(
        x=dates,
        y=actual_values,
        mode='lines',
        name='实际价格',
        line=dict(color='#ff4444', width=3),
        hovertemplate='日期: %{x|%Y-%m-%d}<br>实际价格: $%{y:.2f}<extra>实际值</extra>'
    ))
    
    # 添加各模型预测值
    for i, (model_name, predictions) in enumerate(predictions_dict.items()):
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=dates,
            y=predictions,
            mode='lines',
            name=model_name,
            line=dict(color=color, width=1.5, dash='dash'),
            hovertemplate=f'日期: %{{x|%Y-%m-%d}}<br>预测价格: $%{{y:.2f}}<extra>{model_name}</extra>'
        ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18, color='white')),
        template='plotly_dark',
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(title='日期', showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
        yaxis=dict(title='价格 ($)', showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    )
    
    return fig


def create_metrics_comparison_chart(metrics_dict):
    """
    创建模型性能指标对比图
    
    参数:
        metrics_dict: 字典 {模型名称: {'R²': float, 'RMSE': float, 'MAPE': float, ...}}
    
    返回:
        go.Figure: Plotly图表
    """
    model_names = list(metrics_dict.keys())
    
    # 提取各项指标
    r2_scores = []
    rmse_scores = []
    mape_scores = []
    
    for name in model_names:
        metrics = metrics_dict[name]
        r2_scores.append(metrics.get('R²', 0))
        rmse_scores.append(metrics.get('RMSE', 0))
        mape_str = str(metrics.get('MAPE', '0%')).replace('%', '')
        try:
            mape_scores.append(float(mape_str))
        except:
            mape_scores.append(0)
    
    # 创建子图
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('R² 决定系数', 'RMSE 均方根误差', 'MAPE 平均绝对百分比误差'),
        horizontal_spacing=0.1
    )
    
    colors = ['#2196f3', '#4caf50', '#ff9800', '#e91e63', '#9c27b0']
    bar_colors = [colors[i % len(colors)] for i in range(len(model_names))]
    
    # R² 对比（越高越好）
    fig.add_trace(
        go.Bar(x=model_names, y=r2_scores, marker_color=bar_colors, 
               text=[f'{v:.4f}' for v in r2_scores], textposition='outside',
               hovertemplate='%{x}<br>R²: %{y:.4f}<extra></extra>'),
        row=1, col=1
    )
    
    # RMSE 对比（越低越好）
    fig.add_trace(
        go.Bar(x=model_names, y=rmse_scores, marker_color=bar_colors,
               text=[f'{v:.2f}' for v in rmse_scores], textposition='outside',
               hovertemplate='%{x}<br>RMSE: %{y:.2f}<extra></extra>'),
        row=1, col=2
    )
    
    # MAPE 对比（越低越好）
    fig.add_trace(
        go.Bar(x=model_names, y=mape_scores, marker_color=bar_colors,
               text=[f'{v:.2f}%' for v in mape_scores], textposition='outside',
               hovertemplate='%{x}<br>MAPE: %{y:.2f}%<extra></extra>'),
        row=1, col=3
    )
    
    fig.update_layout(
        title=dict(text='模型性能指标对比', x=0.5, font=dict(size=18, color='white')),
        template='plotly_dark',
        height=400,
        showlegend=False
    )
    
    # 更新Y轴
    fig.update_yaxes(title_text="R²", row=1, col=1)
    fig.update_yaxes(title_text="RMSE", row=1, col=2)
    fig.update_yaxes(title_text="MAPE (%)", row=1, col=3)
    
    return fig


def generate_model_recommendation(metrics_dict):
    """
    根据性能指标生成模型推荐
    
    参数:
        metrics_dict: 字典 {模型名称: {'R²': float, 'RMSE': float, 'MAPE': float, ...}}
    
    返回:
        dict: 推荐结果
    """
    scores = {}
    
    for name, metrics in metrics_dict.items():
        r2 = metrics.get('R²', 0)
        rmse = metrics.get('RMSE', float('inf'))
        mape_str = str(metrics.get('MAPE', '100%')).replace('%', '')
        try:
            mape = float(mape_str)
        except:
            mape = 100
        
        # 综合评分：R²权重40%，RMSE权重30%（归一化取反），MAPE权重30%（取反）
        # 简化计算：R² * 40 + (1 - RMSE/max_rmse) * 30 + (1 - MAPE/100) * 30
        score = r2 * 40 + max(0, (100 - mape)) * 0.3 * 2
        scores[name] = {
            'score': score,
            'R²': r2,
            'RMSE': rmse,
            'MAPE': mape
        }
    
    # 按分数排序
    sorted_models = sorted(scores.items(), key=lambda x: x[1]['score'], reverse=True)
    
    best_model = sorted_models[0][0]
    best_score = sorted_models[0][1]
    
    recommendation = {
        'best_model': best_model,
        'best_score': best_score['score'],
        'best_metrics': best_score,
        'ranking': [(name, data['score']) for name, data in sorted_models],
        'reason': f"R²={best_score['R²']:.4f}，MAPE={best_score['MAPE']:.2f}%"
    }
    
    return recommendation
