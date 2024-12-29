import pandas as pd
import os


# 定义时间段标签分配函数
def assign_time_labels(hour):
    """
    根据小时数为每个时间段分配标签。
    早晨：6-9点 -> 0；上午：9-12点 -> 1；下午：12-18点 -> 2；傍晚：18-21点 -> 3；夜间：21-24点 -> 4；深夜：0-6点 -> 5
    """
    if 6 <= hour < 9:
        return 0  # 早晨
    elif 9 <= hour < 12:
        return 1  # 上午
    elif 12 <= hour < 18:
        return 2  # 下午
    elif 18 <= hour < 21:
        return 3  # 傍晚
    elif 21 <= hour < 24:
        return 4  # 夜间
    else:
        return 5  # 深夜


# 文件夹路径
folder_path = r'C:\Users\15805\Desktop'

# 处理每个家庭的数据
for house_id in range(1, 14):  # 从1到13遍历所有家庭
    file_path = os.path.join(folder_path, f'CLEAN_House{house_id}.csv')

    # 加载数据
    df = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')

    # 重采样到每小时数据，假设数据是按分钟记录的
    df_resampled = df.resample('H').mean()

    # 提取小时信息
    df_resampled['Hour'] = df_resampled.index.hour

    # 应用时间段标签分配函数
    df_resampled['Time_Label'] = df_resampled['Hour'].apply(assign_time_labels)

    # 输出前几行查看标签
    print(f"House {house_id} labels:")
    print(df_resampled.head())

    # 保存带标签的数据
    df_resampled.to_csv(os.path.join(folder_path, f'CLEAN_House{house_id}_with_labels.csv'))

    # 如果有伪标签需要保存
    pseudo_labels = df_resampled['Time_Label'].values
    print(f"Generated pseudo labels for House {house_id}:")
    print(pseudo_labels)

    print(f"Processed and saved labels for House {house_id}")