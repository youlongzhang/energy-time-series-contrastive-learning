import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# 1. 设置文件路径并读取数据
file_path = r'C:\Users\15805\Desktop\CLEAN_House1.csv'
df = pd.read_csv(file_path)

# 2. 数据预处理：转换时间格式并设置索引
df['Time'] = pd.to_datetime(df['Time'])  # 假设时间列名为 'Time'
df.set_index('Time', inplace=True)

# 3. 设置开始时间和结束时间，并重新采样数据到每分钟
start_time = '2013-10-09 13:06:17'
end_time = '2015-07-10 11:56:32'
df_resampled = df.resample('1T').mean().fillna(method='ffill')  # 每分钟重新采样，向前填充缺失值

# 4. 选择指定时间范围内的数据
df_resampled = df_resampled.loc[start_time:end_time]

# 5. 数据切分：将每一天的数据作为一个样本，每天有 1440 个时间步（24小时 * 60分钟）
def split_into_daily_samples(data, sample_size=1440):
    num_samples = len(data) // sample_size  # 计算能分割的天数
    data = data[:num_samples * sample_size]  # 确保数据可以整除
    samples = np.array(np.split(data, num_samples))  # 分割数据，每天一个样本
    return samples

# 假设你想基于 'Aggregate' 列进行切分（总能耗列名可能不同）
data = df_resampled['Aggregate'].values  # 将数据转为数组
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df_resampled[['Aggregate']])
samples_scaled = split_into_daily_samples(data_scaled)
samples = split_into_daily_samples(data)

# 6. 扩展维度以匹配模型输入格式 (batch_size, time_steps, features)
samples = np.expand_dims(samples, axis=-1)  # 样本维度为 (num_samples, 1440, 1)

# 7. 划分训练集和测试集（80%训练，20%测试）
X_trains, X_tests = train_test_split(samples, test_size=0.2, random_state=42)

# 8. 使用MinMaxScaler标准化处理输入数据
scaler = MinMaxScaler(feature_range=(0, 1))

# 注意：我们需要对数据进行拟合和转换，以确保测试数据使用训练数据的范围进行标准化
# 拟合和转换训练数据
X_train = scaler.fit_transform(X_trains.reshape(-1, X_trains.shape[-1])).reshape(X_trains.shape)

# 仅转换测试数据（不重新拟合）
X_test = scaler.transform(X_tests.reshape(-1, X_tests.shape[-1])).reshape(X_tests.shape)
