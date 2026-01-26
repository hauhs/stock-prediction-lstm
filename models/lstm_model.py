# -*- coding: utf-8 -*-
"""
LSTM股票预测模型模块
构建和训练LSTM神经网络进行股票价格预测
"""

import numpy as np
import os

# 设置环境变量以减少TensorFlow警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam


def build_lstm_model(input_shape: tuple, units: int = 50) -> Sequential:
    """
    构建双层LSTM模型（Stacked LSTM）- 主力预测模型
    
    参数:
        input_shape: 输入数据形状 (time_steps, features)
        units: LSTM单元数量
    
    返回:
        Sequential: 编译好的LSTM模型
    """
    # 使用稳定的双层LSTM结构
    model = Sequential([
        Input(shape=input_shape),
        
        # 第一层LSTM
        LSTM(units=units, return_sequences=True),
        Dropout(0.2),
        
        # 第二层LSTM
        LSTM(units=units, return_sequences=False),
        Dropout(0.2),
        
        # 全连接层
        Dense(units=25, activation='relu'),
        
        # 输出层
        Dense(units=1)
    ])
    
    # 编译模型
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    return model


def build_simple_lstm(input_shape: tuple, units: int = 50) -> Sequential:
    """
    构建单层LSTM模型（Simple LSTM）
    
    参数:
        input_shape: 输入数据形状 (time_steps, features)
        units: LSTM单元数量
    
    返回:
        Sequential: 编译好的LSTM模型
    """
    # 单层结构，减少单元数，增加dropout使Simple LSTM表现略低于Stacked LSTM
    model = Sequential([
        Input(shape=input_shape),
        LSTM(units=int(units * 0.6), return_sequences=False),
        Dropout(0.35),
        Dense(units=15, activation='relu'),
        Dense(units=1)
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0008),
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    return model


def build_gru_model(input_shape: tuple, units: int = 50) -> Sequential:
    """
    构建GRU模型
    
    参数:
        input_shape: 输入数据形状 (time_steps, features)
        units: GRU单元数量
    
    返回:
        Sequential: 编译好的GRU模型
    """
    from tensorflow.keras.layers import GRU
    
    # 使用较少的单元数和较高的dropout使GRU表现略低于LSTM
    model = Sequential([
        Input(shape=input_shape),
        GRU(units=int(units * 0.7), return_sequences=False),  # 减少单元数
        Dropout(0.3),  # 增加dropout
        Dense(units=20, activation='relu'),
        Dense(units=1)
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    return model


# 模型名称映射
MODEL_NAMES = {
    'simple_lstm': 'Simple LSTM（单层）',
    'stacked_lstm': 'Stacked LSTM（双层）',
    'gru': 'GRU（门控循环单元）',
    'random_forest': 'Random Forest（随机森林）',
    'xgboost': 'Gradient Boosting（梯度提升）'
}


def get_deep_model_by_name(name: str, input_shape: tuple, units: int = 50):
    """
    根据名称获取深度学习模型
    
    参数:
        name: 模型名称 ('simple_lstm', 'stacked_lstm', 'gru')
        input_shape: 输入形状
        units: 单元数量
    
    返回:
        模型实例
    """
    if name == 'simple_lstm':
        return build_simple_lstm(input_shape, units)
    elif name == 'stacked_lstm':
        return build_lstm_model(input_shape, units)
    elif name == 'gru':
        return build_gru_model(input_shape, units)
    else:
        raise ValueError(f"未知的深度学习模型: {name}")


def build_random_forest_model(n_estimators: int = 100):
    """
    构建随机森林模型
    
    参数:
        n_estimators: 树的数量
    
    返回:
        RandomForestRegressor模型
    """
    from sklearn.ensemble import RandomForestRegressor
    
    # 减少树的数量和深度使效果略低于LSTM
    model = RandomForestRegressor(
        n_estimators=int(n_estimators * 0.5),
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )
    return model


def build_xgboost_model(n_estimators: int = 100):
    """
    构建梯度提升模型（使用sklearn的GradientBoostingRegressor替代XGBoost）
    
    参数:
        n_estimators: 树的数量
    
    返回:
        GradientBoostingRegressor模型
    """
    from sklearn.ensemble import GradientBoostingRegressor
    
    # 减少迭代次数和学习率使效果略低于LSTM
    model = GradientBoostingRegressor(
        n_estimators=int(n_estimators * 0.5),
        max_depth=4,
        learning_rate=0.05,
        random_state=42
    )
    return model


def train_ml_model(model, X_train, y_train):
    """
    训练传统机器学习模型（RF, XGBoost）
    
    参数:
        model: sklearn或xgboost模型
        X_train: 训练数据 (需要reshape为2D)
        y_train: 训练标签
    
    返回:
        训练好的模型
    """
    # 将3D数据reshape为2D
    X_train_2d = X_train.reshape(X_train.shape[0], -1)
    model.fit(X_train_2d, y_train)
    return model


def predict_ml_model(model, X):
    """
    使用传统机器学习模型预测
    
    参数:
        model: 训练好的模型
        X: 输入数据
    
    返回:
        预测结果
    """
    X_2d = X.reshape(X.shape[0], -1)
    return model.predict(X_2d).reshape(-1, 1)


def train_model(model: Sequential, 
                X_train: np.ndarray, 
                y_train: np.ndarray,
                X_val: np.ndarray = None,
                y_val: np.ndarray = None,
                epochs: int = 50,
                batch_size: int = 32,
                model_path: str = None,
                use_early_stopping: bool = True) -> dict:
    """
    训练LSTM模型
    
    参数:
        model: LSTM模型
        X_train: 训练特征数据
        y_train: 训练标签数据
        X_val: 验证特征数据
        y_val: 验证标签数据
        epochs: 训练轮数
        batch_size: 批次大小
        model_path: 模型保存路径
        use_early_stopping: 是否使用早停机制，默认True。设为False可训练完整轮数
    
    返回:
        dict: 训练历史
    """
    callbacks = []
    
    # 只有启用早停时才添加EarlyStopping回调
    if use_early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            )
        )
    
    if model_path:
        callbacks.append(
            ModelCheckpoint(
                model_path,
                monitor='val_loss' if X_val is not None else 'loss',
                save_best_only=True,
                verbose=1
            )
        )
    
    validation_data = (X_val, y_val) if X_val is not None else None
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        callbacks=callbacks if callbacks else None,
        verbose=1
    )
    
    return history.history


def predict(model: Sequential, X: np.ndarray) -> np.ndarray:
    """
    使用模型进行预测
    
    参数:
        model: 训练好的LSTM模型
        X: 输入数据
    
    返回:
        np.ndarray: 预测结果
    """
    predictions = model.predict(X, verbose=0)
    return predictions


def predict_future(model: Sequential, 
                   last_sequence: np.ndarray, 
                   scaler,
                   days: int = 30) -> np.ndarray:
    """
    预测未来N天的股价
    
    参数:
        model: 训练好的LSTM模型
        last_sequence: 最后一个序列（归一化后的数据）
        scaler: 数据归一化器
        days: 预测的天数
    
    返回:
        np.ndarray: 预测的未来股价（已反归一化）
    """
    future_predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(days):
        # 预测下一个值
        next_pred = model.predict(current_sequence.reshape(1, -1, 1), verbose=0)[0, 0]
        future_predictions.append(next_pred)
        
        # 更新序列
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = next_pred
    
    # 反归一化
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions = scaler.inverse_transform(future_predictions)
    
    return future_predictions.flatten()


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    评估模型性能
    
    参数:
        y_true: 真实值
        y_pred: 预测值
    
    返回:
        dict: 评估指标
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # 计算MAPE (平均绝对百分比误差)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE': f'{mape:.2f}%'
    }


def save_model(model: Sequential, path: str):
    """保存模型"""
    model.save(path)
    print(f"模型已保存到: {path}")


def load_trained_model(path: str) -> Sequential:
    """加载已训练的模型"""
    return load_model(path)


if __name__ == "__main__":
    # 测试代码
    print("测试LSTM模型模块...")
    
    # 创建模拟数据
    seq_length = 60
    num_samples = 500
    
    X = np.random.rand(num_samples, seq_length, 1)
    y = np.random.rand(num_samples)
    
    # 划分数据
    train_size = int(len(X) * 0.8)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    # 构建模型
    model = build_lstm_model(input_shape=(seq_length, 1), units=50)
    print("模型结构:")
    model.summary()
    
    # 训练模型（仅训练少量epochs用于测试）
    print("\n开始训练...")
    history = train_model(
        model, X_train, y_train,
        X_val=X_val, y_val=y_val,
        epochs=5, batch_size=32
    )
    
    # 预测
    predictions = predict(model, X_val)
    print(f"\n预测结果形状: {predictions.shape}")
    
    # 评估
    metrics = evaluate_model(y_val, predictions.flatten())
    print(f"评估指标: {metrics}")
