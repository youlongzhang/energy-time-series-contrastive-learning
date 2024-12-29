import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.cluster import AffinityPropagation

# 1. 读取数据
file_path = r'C:\Users\15805\Desktop\CLEAN_House10.csv'
df = pd.read_csv(file_path)

# 2. 数据预处理：转换时间格式并设置索引
df['Time'] = pd.to_datetime(df['Time'])
df.set_index('Time', inplace=True)

# 3. 提取指定的时间范围
start_time = df.index.min()
end_time = df.index.max()

# 4. 按照周进行切分
weekly_groups = df.resample('M')  # 每周分组

# 5. 对每周的数据进行处理并保存开始和结束时间，并为每周分配编号
week_times = [(week_idx, group.index.min(), group.index.max()) for week_idx, (_, group) in enumerate(weekly_groups)]

# 打印每周的编号以及开始和结束时间
print("每周的编号、开始和结束时间：")
for week_idx, start, end in week_times:
    print(f"Week {week_idx}: Start: {start}, End: {end}")

def split_into_180_minute_samples(data, sample_size=180):
        num_samples = len(data) // sample_size
        data = data[:num_samples * sample_size]
        samples = np.array(np.split(data, num_samples))
        return samples
# 6. 定义函数，用于将数据切分成每日样本并进行标准化
def split_into_daily_samples(data, sample_size=1440):
    num_samples = len(data) // sample_size
    data = data[:num_samples * sample_size]
    samples = np.array(np.split(data, num_samples))
    return samples



# 7. 遍历每周的数据
for week_idx, start, end in week_times:
    df_week = df.loc[start:end]  # 提取一周的数据
    if pd.isna(start) and pd.isna(end):
        print(f"Week {week_idx} has no data (NaT start and end) and will be skipped.")
        continue  # 跳过这一周
    df_resampled = df_week.resample('1T').mean().fillna(method='ffill')  # 每分钟重新采样，向前填充缺失值

    minutes_per_day = 24 * 60  # 每天的分钟数
    total_minutes = len(df_resampled)
    num_days = total_minutes // minutes_per_day
    if total_minutes != num_days * minutes_per_day:
        print(f"Filtered data length is not a multiple of 1440. Adjusting to the nearest complete day.")
        total_minutes = num_days * minutes_per_day
        df_resampled = df_resampled.iloc[:total_minutes]
    # 分割每天的数据
    daily_data = np.array_split(df_resampled['Appliance3'].values, num_days)

    # 确保每个分割的数据长度为 1440
    for i, day in enumerate(daily_data):
        if len(day) != minutes_per_day:
            print(f"Day {i + 1} length: {len(day)}")
            daily_data[i] = np.pad(day, (0, minutes_per_day - len(day)), 'constant', constant_values=np.nan)

    # 将数据转换为合适的形状
    daily_data = np.array(daily_data).reshape((num_days, minutes_per_day, 1))

    # 扩展维度
    samples = daily_data



    # 8. 使用预训练的特征提取器
    feature_extractor = tf.keras.models.load_model('best_contrastive_model.h5')
    features = feature_extractor.predict(samples)

    # 9. Affinity Propagation聚类
    affinity_propagation = AffinityPropagation(preference=1.0, damping=0.8, random_state=42)
    clusters = affinity_propagation.fit_predict(features)

    # 关闭交互模式
    plt.ioff()
    start_str = start.strftime('%Y-%m-%d_%H-%M-%S')  # 格式化开始日期时间
    end_str = end.strftime('%Y-%m-%d_%H-%M-%S')  # 格式化结束日期时间
    # 10. t-SNE可视化
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)

    plt.figure(figsize=(10, 7))
    plt.scatter(features_2d[:, 0], features_2d[:, 1], c=clusters, cmap='viridis', s=50)
    plt.colorbar()
    plt.title(f"t-SNE Clustering of Week {week_idx}_Start_{start_str}_End_{end_str}")

    # 保存 t-SNE 可视化图像而不显示
    tsne_filename = f"tsne_week_{week_idx}.png"
    save_path = "D:\\visual_moth\\"  # 注意：Windows路径使用双反斜杠
    plt.savefig(save_path + tsne_filename)

    plt.close()  # 关闭图形以避免内存泄漏

    # 11. 聚类中心及类内样本可视化
    # 获取聚类中心
    cluster_centers_indices = affinity_propagation.cluster_centers_indices_
    n_clusters_ = len(cluster_centers_indices)

    for cluster_idx in range(n_clusters_):
        plt.figure(figsize=(10, 6))

        cluster_samples = samples[clusters == cluster_idx]
        cluster_center = features[cluster_centers_indices[cluster_idx]].flatten()  # 使用聚类中心

        plt.plot(cluster_center, label=f'Cluster {cluster_idx} Center', color='red', linewidth=2)

        for sample in cluster_samples[:10]:  # 展示部分样本
            plt.plot(sample.flatten(), color='blue', alpha=0.3)

        plt.title(f'Cluster {cluster_idx} - Week {start} to {end}')
        plt.legend()

        # 保存聚类中心及类内样本可视化图像而不显示
        cluster_filename = f"cluster_{cluster_idx}.png"

        save_path = "D:\\visual_moth\\"  # 注意：Windows路径使用双反斜杠
        plt.savefig(save_path + cluster_filename)

    plt.close()  # 关闭图形以避免内存泄漏