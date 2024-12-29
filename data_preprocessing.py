import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def split_into_daily_samples(data, sample_size=1440):
    """
    将时间序列数据按照天进行分割
    """
    num_samples = len(data) // sample_size  # 计算能分割的天数
    data = data[:num_samples * sample_size]  # 确保数据可以整除
    samples = np.array(np.split(data, num_samples))  # 分割数据，每天一个样本
    return samples


def preprocess_data(file_path, time_range, sample_size=1440):
    """
    加载和预处理数据，包括时间索引设置、重采样、标准化等
    """
    df = pd.read_csv(file_path)

    # 转换时间列并设置为索引
    df['Time'] = pd.to_datetime(df['Time'])
    df.set_index('Time', inplace=True)

    # 重采样至每分钟一条数据
    df_resampled = df.resample('1T').mean().fillna(method='ffill')
    df_resampled = df_resampled.loc[time_range[0]:time_range[1]]

    # 提取‘Aggregate’列
    data = df_resampled['Aggregate'].values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df_resampled[['Aggregate']])

    # 分割数据
    samples_scaled = split_into_daily_samples(data_scaled, sample_size)
    samples = split_into_daily_samples(data, sample_size)
    samples = np.expand_dims(samples, axis=-1)  # 增加一个维度

    return samples, samples_scaled, scaler


def prepare_train_test_split(all_samples, test_size=0.2):
    """
    划分训练集和测试集
    """
    X_train, X_test = train_test_split(all_samples, test_size=test_size, random_state=42)

    return X_train, X_test
