import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
import random
import matplotlib.pyplot as plt
from Data_Augmentation.Dtw_loss import dtw_loss
from Data_Augmentation.self_attention_augmentation import create_transformer_augmented_model
from Contrastive_Learning.Contrastive_loss import contrastive_loss
from Contrastive_Learning.ENERGYNET import create_encoder, create_projection_head



# 数据增强方法：噪声添加
def add_noise(X, noise_factor=0.1):
    noise = np.random.normal(loc=0.0, scale=noise_factor, size=X.shape)
    X_noisy = X + noise
    return X_noisy


# 数据增强方法：缩放
def scale_data(X, scale_factor=1.5):
    X_scaled = X * scale_factor
    return X_scaled


# 数据增强方法：时间扭曲（简单示例）
def time_warping(X, warp_factor=0.1):
    X_warped = X.copy()
    for i in range(X.shape[0]):
        warp = np.random.uniform(1 - warp_factor, 1 + warp_factor, X.shape[1])
        X_warped[i] = X[i] * warp
    return X_warped


# 数据增强方法：时间裁剪（选择随机时间段）
def time_cropping(X, crop_size=1000):
    crop_start = random.randint(0, X.shape[1] - crop_size)
    X_cropped = X[:, crop_start:crop_start + crop_size]
    return X_cropped


# Transformer增强方法
def augment_data_with_transformer(X, input_shape, embed_dim=64, num_heads=4, ff_dim=128, num_blocks=2):
    transformer_model = create_transformer_augmented_model(input_shape, embed_dim, num_heads, ff_dim, num_blocks)
    augmented_data = transformer_model.predict(X)
    return augmented_data


# 创建对比学习模型
def create_contrastive_model(input_shape):
    input_1 = Input(shape=input_shape)  # 原始序列输入
    input_2 = Input(shape=input_shape)  # 增强视图输入

    # 使用ENERGYNET的特征提取器
    encoder = create_encoder()  # 创建特征提取器
    projection_head = create_projection_head()  # 创建投影头

    # 编码原始序列和增强视图
    encoded_1 = encoder(input_1)
    encoded_2 = encoder(input_2)

    # 对编码后的输出进行投影
    projected_1 = projection_head(encoded_1)
    projected_2 = projection_head(encoded_2)

    # 创建模型
    model = Model(inputs=[input_1, input_2], outputs=[projected_1, projected_2])
    return model


# 训练对比学习模型
def train_contrastive_model(X, augmented_X, model, epochs=10, batch_size=32):
    model.compile(optimizer='adam', loss=contrastive_loss)
    history = model.fit([X, augmented_X], X, epochs=epochs, batch_size=batch_size)
    return model


# 提取特征
def extract_features(model, X, augmented_X):
    features_1 = model.predict([X, augmented_X])
    return features_1


# 聚类与评估
def perform_clustering(features):
    ap = AffinityPropagation(random_state=42)
    y_pred = ap.fit_predict(features)
    return y_pred


def evaluate_clustering(y_true, y_pred):
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    silhouette = silhouette_score(X, y_pred)
    return nmi, ari, silhouette


# 不同的数据增强方法
augmentation_methods = ['original', 'noise', 'scale', 'time_warping', 'time_cropping', 'transformer']

nmi_scores = []
ari_scores = []
silhouette_scores = []

# 对不同的增强方法进行实验
for method in augmentation_methods:
    if method == 'original':
        X_augmented = X  # 无增强，使用原始数据
    elif method == 'noise':
        X_augmented = add_noise(X)
    elif method == 'scale':
        X_augmented = scale_data(X)
    elif method == 'time_warping':
        X_augmented = time_warping(X)
    elif method == 'time_cropping':
        X_augmented = time_cropping(X)
    elif method == 'transformer':
        X_augmented = augment_data_with_transformer(X, X.shape[1:])

    # 训练对比学习模型
    contrastive_model = create_contrastive_model(X.shape[1:])
    trained_model = train_contrastive_model(X, X_augmented, contrastive_model)

    # 提取特征并进行聚类
    features = extract_features(trained_model, X, X_augmented)

    # 聚类
    y_pred = perform_clustering(features)

    # 计算评价指标
    nmi, ari, silhouette = evaluate_clustering(y_true, y_pred)

    # 保存结果
    nmi_scores.append(nmi)
    ari_scores.append(ari)
    silhouette_scores.append(silhouette)

# 输出结果
for i, method in enumerate(augmentation_methods):
    print(f"Method: {method}")
    print(f"  NMI: {nmi_scores[i]:.4f}")
    print(f"  ARI: {ari_scores[i]:.4f}")
    print(f"  Silhouette Score: {silhouette_scores[i]:.4f}")
    print("-" * 40)

# 绘制评价指标
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

ax[0].bar(augmentation_methods, nmi_scores, color='blue')
ax[0].set_title('Normalized Mutual Information Score (NMI)')
ax[0].set_ylabel('Score')

ax[1].bar(augmentation_methods, ari_scores, color='green')
ax[1].set_title('Adjusted Rand Index (ARI)')
ax[1].set_ylabel('Score')

ax[2].bar(augmentation_methods, silhouette_scores, color='red')
ax[2].set_title('Silhouette Score')
ax[2].set_ylabel('Score')

plt.tight_layout()
plt.show()
