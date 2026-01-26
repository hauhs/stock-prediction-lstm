# -*- coding: utf-8 -*-
"""
数据获取和预处理模块
使用yfinance获取股票历史数据，并进行预处理
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta


def get_stock_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取股票历史数据
    
    参数:
        symbol: 股票代码，如 'AAPL' (美股) 或 '000001.SS' (沪市)
        start_date: 开始日期，格式 'YYYY-MM-DD'
        end_date: 结束日期，格式 'YYYY-MM-DD'
    
    返回:
        DataFrame: 包含日期、开盘价、最高价、最低价、收盘价、成交量的数据
    """
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            raise ValueError(f"无法获取股票 {symbol} 的数据，请检查股票代码是否正确")
        
        # 重置索引，将日期作为列
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        
        # 只保留需要的列
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        return df
    
    except Exception as e:
        raise Exception(f"获取股票数据失败: {str(e)}")


def preprocess_data(df: pd.DataFrame, feature_col: str = 'Close') -> tuple:
    """
    预处理数据，进行归一化
    
    参数:
        df: 原始股票数据
        feature_col: 用于预测的特征列，默认为收盘价
    
    返回:
        tuple: (归一化后的数据, 归一化器)
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    data = df[feature_col].values.reshape(-1, 1)
    scaled_data = scaler.fit_transform(data)
    
    return scaled_data, scaler


def create_sequences(data: np.ndarray, seq_length: int = 60) -> tuple:
    """
    创建LSTM所需的时间序列数据集
    
    参数:
        data: 归一化后的数据
        seq_length: 序列长度（用于预测的历史天数）
    
    返回:
        tuple: (X, y) 训练数据和标签
    """
    X, y = [], []
    
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length:i, 0])
        y.append(data[i, 0])
    
    X = np.array(X)
    y = np.array(y)
    
    # 重塑数据为LSTM输入格式 [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y


def split_data(X: np.ndarray, y: np.ndarray, train_ratio: float = 0.8) -> tuple:
    """
    划分训练集和测试集
    
    参数:
        X: 特征数据
        y: 标签数据
        train_ratio: 训练集占比
    
    返回:
        tuple: (X_train, X_test, y_train, y_test)
    """
    train_size = int(len(X) * train_ratio)
    
    X_train = X[:train_size]
    X_test = X[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]
    
    return X_train, X_test, y_train, y_test


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算技术指标
    
    参数:
        df: 股票数据DataFrame
    
    返回:
        DataFrame: 添加了技术指标的数据
    """
    df = df.copy()
    
    # 移动平均线
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    
    # 日收益率
    df['Daily_Return'] = df['Close'].pct_change()
    
    # 波动率（20日滚动标准差）
    df['Volatility'] = df['Daily_Return'].rolling(window=20).std() * np.sqrt(252)
    
    # RSI (相对强弱指标)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    return df


def calculate_risk_metrics(df: pd.DataFrame) -> dict:
    """
    计算风险指标
    
    参数:
        df: 股票数据DataFrame（需包含Daily_Return列）
    
    返回:
        dict: 风险指标字典
    """
    if 'Daily_Return' not in df.columns:
        df['Daily_Return'] = df['Close'].pct_change()
    
    returns = df['Daily_Return'].dropna()
    
    # 年化收益率
    annual_return = returns.mean() * 252
    
    # 年化波动率
    annual_volatility = returns.std() * np.sqrt(252)
    
    # 夏普比率（假设无风险利率为3%）
    sharpe_ratio = (annual_return - 0.03) / annual_volatility if annual_volatility != 0 else 0
    
    # 最大回撤
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # VaR (95%置信度)
    var_95 = np.percentile(returns, 5)
    
    return {
        '年化收益率': f'{annual_return:.2%}',
        '年化波动率': f'{annual_volatility:.2%}',
        '夏普比率': f'{sharpe_ratio:.2f}',
        '最大回撤': f'{max_drawdown:.2%}',
        'VaR(95%)': f'{var_95:.2%}'
    }


def analyze_trend(df: pd.DataFrame) -> dict:
    """
    分析股票趋势
    
    参数:
        df: 股票数据DataFrame（需包含技术指标）
    
    返回:
        dict: 趋势分析结果
    """
    if len(df) < 20:
        return {'趋势': '数据不足', '强度': 0}
    
    # 获取最新数据
    current_price = df['Close'].iloc[-1]
    ma5 = df['MA5'].iloc[-1] if 'MA5' in df.columns else current_price
    ma10 = df['MA10'].iloc[-1] if 'MA10' in df.columns else current_price
    ma20 = df['MA20'].iloc[-1] if 'MA20' in df.columns else current_price
    
    # 趋势判断
    if current_price > ma5 > ma10 > ma20:
        trend = '强势上涨'
        strength = 5
        trend_color = '#4caf50'
    elif current_price > ma5 and current_price > ma20:
        trend = '温和上涨'
        strength = 4
        trend_color = '#8bc34a'
    elif current_price > ma20:
        trend = '震荡偏多'
        strength = 3
        trend_color = '#cddc39'
    elif current_price < ma5 < ma10 < ma20:
        trend = '强势下跌'
        strength = 1
        trend_color = '#f44336'
    elif current_price < ma5 and current_price < ma20:
        trend = '温和下跌'
        strength = 2
        trend_color = '#ff5722'
    else:
        trend = '震荡整理'
        strength = 3
        trend_color = '#9e9e9e'
    
    # 计算涨跌幅
    price_change_1d = (df['Close'].iloc[-1] / df['Close'].iloc[-2] - 1) * 100 if len(df) > 1 else 0
    price_change_5d = (df['Close'].iloc[-1] / df['Close'].iloc[-5] - 1) * 100 if len(df) > 5 else 0
    price_change_20d = (df['Close'].iloc[-1] / df['Close'].iloc[-20] - 1) * 100 if len(df) > 20 else 0
    
    # 均线支撑/压力
    support_levels = []
    resistance_levels = []
    
    if current_price > ma5:
        support_levels.append(f"MA5: ${ma5:.2f}")
    else:
        resistance_levels.append(f"MA5: ${ma5:.2f}")
    
    if current_price > ma10:
        support_levels.append(f"MA10: ${ma10:.2f}")
    else:
        resistance_levels.append(f"MA10: ${ma10:.2f}")
    
    if current_price > ma20:
        support_levels.append(f"MA20: ${ma20:.2f}")
    else:
        resistance_levels.append(f"MA20: ${ma20:.2f}")
    
    return {
        '趋势': trend,
        '趋势强度': strength,
        '趋势颜色': trend_color,
        '当前价格': current_price,
        '1日涨跌幅': price_change_1d,
        '5日涨跌幅': price_change_5d,
        '20日涨跌幅': price_change_20d,
        '支撑位': support_levels,
        '压力位': resistance_levels
    }


def generate_investment_advice(df: pd.DataFrame, 
                                future_predictions: np.ndarray,
                                risk_metrics: dict) -> dict:
    """
    生成投资建议
    
    参数:
        df: 股票数据DataFrame（需包含技术指标）
        future_predictions: 未来价格预测
        risk_metrics: 风险指标
    
    返回:
        dict: 投资建议
    """
    advice = {
        '综合评分': 0,
        '操作建议': '',
        '风险等级': '',
        '分析要点': [],
        '注意事项': []
    }
    
    score = 50  # 基础分
    points = []
    warnings = []
    
    # 1. 趋势分析
    trend_info = analyze_trend(df)
    if trend_info['趋势强度'] >= 4:
        score += 15
        points.append(f"📈 当前处于{trend_info['趋势']}趋势，动能较强")
    elif trend_info['趋势强度'] <= 2:
        score -= 15
        points.append(f"📉 当前处于{trend_info['趋势']}趋势，需谨慎")
    else:
        points.append(f"↔️ 当前处于{trend_info['趋势']}，建议观望")
    
    # 2. RSI分析
    if 'RSI' in df.columns:
        current_rsi = df['RSI'].iloc[-1]
        if current_rsi > 70:
            score -= 10
            warnings.append("⚠️ RSI > 70，处于超买区域，短期回调风险较高")
        elif current_rsi < 30:
            score += 10
            points.append("💡 RSI < 30，处于超卖区域，可能存在反弹机会")
        else:
            points.append(f"📊 RSI = {current_rsi:.1f}，处于正常区域")
    
    # 3. MACD分析
    if 'MACD' in df.columns and 'Signal_Line' in df.columns:
        macd = df['MACD'].iloc[-1]
        signal = df['Signal_Line'].iloc[-1]
        macd_prev = df['MACD'].iloc[-2] if len(df) > 1 else macd
        signal_prev = df['Signal_Line'].iloc[-2] if len(df) > 1 else signal
        
        # 金叉/死叉判断
        if macd > signal and macd_prev <= signal_prev:
            score += 10
            points.append("🔺 MACD金叉形成，买入信号")
        elif macd < signal and macd_prev >= signal_prev:
            score -= 10
            warnings.append("🔻 MACD死叉形成，卖出信号")
        elif macd > signal:
            points.append("📈 MACD位于信号线上方，多头趋势")
        else:
            points.append("📉 MACD位于信号线下方，空头趋势")
    
    # 4. 预测分析
    if future_predictions is not None and len(future_predictions) > 0:
        current_price = df['Close'].iloc[-1]
        pred_price = future_predictions[-1]
        pred_change = (pred_price / current_price - 1) * 100
        
        if pred_change > 5:
            score += 15
            points.append(f"🎯 模型预测未来上涨 {pred_change:.1f}%，看好后市")
        elif pred_change < -5:
            score -= 15
            warnings.append(f"⚠️ 模型预测未来下跌 {abs(pred_change):.1f}%，建议规避")
        else:
            points.append(f"📊 模型预测未来变化 {pred_change:+.1f}%，变化不大")
    
    # 5. 波动率分析
    try:
        volatility_str = risk_metrics.get('年化波动率', '0%').replace('%', '')
        volatility = float(volatility_str) / 100
        if volatility > 0.4:
            score -= 10
            warnings.append(f"⚠️ 年化波动率 {volatility:.0%}，波动较大，风险较高")
        elif volatility < 0.2:
            score += 5
            points.append(f"✅ 年化波动率 {volatility:.0%}，波动较小，适合稳健投资")
    except:
        pass
    
    # 确定综合评分（0-100）
    score = max(0, min(100, score))
    advice['综合评分'] = score
    
    # 确定操作建议
    if score >= 70:
        advice['操作建议'] = '建议买入'
        advice['建议颜色'] = '#4caf50'
    elif score >= 55:
        advice['操作建议'] = '建议持有'
        advice['建议颜色'] = '#2196f3'
    elif score >= 40:
        advice['操作建议'] = '观望为主'
        advice['建议颜色'] = '#ff9800'
    else:
        advice['操作建议'] = '建议卖出'
        advice['建议颜色'] = '#f44336'
    
    # 确定风险等级
    if score >= 60:
        advice['风险等级'] = '低风险'
        advice['风险颜色'] = '#4caf50'
    elif score >= 40:
        advice['风险等级'] = '中风险'
        advice['风险颜色'] = '#ff9800'
    else:
        advice['风险等级'] = '高风险'
        advice['风险颜色'] = '#f44336'
    
    advice['分析要点'] = points
    advice['注意事项'] = warnings
    advice['趋势信息'] = trend_info
    
    return advice


# 常用股票代码映射（用于中文显示）
STOCK_MAPPING = {
    # 美股
    'AAPL': '苹果 (Apple)',
    'GOOGL': '谷歌 (Google)',
    'MSFT': '微软 (Microsoft)',
    'AMZN': '亚马逊 (Amazon)',
    'TSLA': '特斯拉 (Tesla)',
    'META': 'Meta (Facebook)',
    'NVDA': '英伟达 (NVIDIA)',
    'BABA': '阿里巴巴',
    'JD': '京东',
    'PDD': '拼多多',
    'NIO': '蔚来汽车',
    # A股 (需要加后缀 .SS上海 .SZ深圳)
    '000001.SZ': '平安银行',
    '600519.SS': '贵州茅台',
    '000858.SZ': '五粮液',
    '601318.SS': '中国平安',
    '600036.SS': '招商银行',
}


if __name__ == "__main__":
    # 测试代码
    print("测试数据获取模块...")
    
    # 获取苹果公司股票数据
    df = get_stock_data('AAPL', '2023-01-01', '2024-01-01')
    print(f"获取到 {len(df)} 条数据")
    print(df.head())
    
    # 预处理数据
    scaled_data, scaler = preprocess_data(df)
    print(f"\n归一化后数据形状: {scaled_data.shape}")
    
    # 创建序列
    X, y = create_sequences(scaled_data, seq_length=60)
    print(f"X形状: {X.shape}, y形状: {y.shape}")
    
    # 计算技术指标
    df_with_indicators = calculate_technical_indicators(df)
    print(f"\n技术指标列: {df_with_indicators.columns.tolist()}")
    
    # 计算风险指标
    risk_metrics = calculate_risk_metrics(df_with_indicators)
    print(f"\n风险指标: {risk_metrics}")
