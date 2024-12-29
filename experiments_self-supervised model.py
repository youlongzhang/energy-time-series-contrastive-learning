import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
import random
import matplotlib.pyplot as plt
from Contrastive_Learning.Contrastive_loss import contrastive_loss
from Contrastive_Learning.ENERGYNET import create_encoder, create_projection_head


# 自编码器模型 (Autoencoder)
def create_autoencoder(input_shape):
    input_layer = Input(shape=input_shape)
    encoded = Dense(512, activation='relu')(input_layer)
    encoded = Dense(256, activation='relu')(encoded)
    decoded = Dense(512, activation='relu')(encoded)
    decoded = Dense(input_shape[0], activation='sigmoid')(decoded)

    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)  # 提取编码部分

    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder


# 对抗性无监督嵌入模型 (AUE)
def create_aue_model(input_shape):
    # AUE是对抗性训练模型，这里采用一个简单的示例架构
    input_layer = Input(shape=input_shape)
    encoded = Dense(512, activation='relu')(input_layer)
    encoded = Dense(256, activation='relu')(encoded)

    decoded = Dense(512, activation='relu')(encoded)
    decoded = Dense(input_shape[0], activation='sigmoid')(decoded)

    discriminator = Dense(1, activation='sigmoid')(encoded)

    aue_model = Model(input_layer, [decoded, discriminator])

    aue_model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return aue_model, encoded


# 时序卷积神经网络 (TCN)
def create_tcn_model(input_shape):
    input_layer = Input(shape=input_shape)
    x = Conv1D(64, kernel_size=3, activation='relu')(input_layer)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(128, kernel_size=3, activation='relu')(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(input_shape[0], activation='sigmoid')(x)

    tcn_model = Model(input_layer, x)
    tcn_model.compile(optimizer='adam', loss='mse')
    return tcn_model


# 对比学习模型
def create_contrastive_model(input_shape):
    input_1 = Input(shape=input_shape)
    input_2 = Input(shape=input_shape)

    # 假设已存在特征提取器和投影头
    encoder = create_encoder()
    projection_head = create_projection_head()

    encoded_1 = encoder(input_1)
    encoded_2 = encoder(input_2)

    projected_1 = projection_head(encoded_1)
    projected_2 = projection_head(encoded_2)

    model = Model(inputs=[input_1, input_2], outputs=[projected_1, projected_2])
    return model


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


# 实验流程：Autoencoder, AUE, TCN与对比学习
models = ['autoencoder', 'aue', 'tcn', 'contrastive']

nmi_scores = []
ari_scores = []
silhouette_scores = []

# 使用不同模型进行训练和评估
for model_name in models:
    if model_name == 'autoencoder':
        autoencoder, encoder = create_autoencoder(X.shape[1:])
        autoencoder.fit(X, X, epochs=10, batch_size=32)  # 假设目标是自编码器重构
        features = encoder.predict(X)

    elif model_name == 'aue':
        aue_model, encoder = create_aue_model(X.shape[1:])
        aue_model.fit(X, [X, np.zeros((X.shape[0], 1))], epochs=10, batch_size=32)  # 对抗训练
        features = encoder.predict(X)

    elif model_name == 'tcn':
        tcn_model = create_tcn_model(X.shape[1:])
        tcn_model.fit(X, X, epochs=10, batch_size=32)
        features = tcn_model.predict(X)

    elif model_name == 'contrastive':
        contrastive_model = create_contrastive_model(X.shape[1:])
        contrastive_model.compile(optimizer='adam', loss=contrastive_loss)
        contrastive_model.fit([X, X], X, epochs=10, batch_size=32)
        features = contrastive_model.predict([X, X])

    # 聚类
    y_pred = perform_clustering(features)

    # 计算评价指标
    nmi, ari, silhouette = evaluate_clustering(y_true, y_pred)

    # 保存结果
    nmi_scores.append(nmi)
    ari_scores.append(ari)
    silhouette_scores.append(silhouette)

# 输出结果
for i, model in enumerate(models):
    print(f"Model: {model}")
    print(f"  NMI: {nmi_scores[i]:.4f}")
    print(f"  ARI: {ari_scores[i]:.4f}")
    print(f"  Silhouette Score: {silhouette_scores[i]:.4f}")
    print("-" * 40)

# 绘制评价指标
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

ax[0].bar(models, nmi_scores, color='blue')
ax[0].set_title('Normalized Mutual Information Score (NMI)')
ax[0].set_ylabel('Score')

ax[1].bar(models, ari_scores, color='green')
ax[1].set_title('Adjusted Rand Index (ARI)')
ax[1].set_ylabel('Score')

ax[2].bar(models, silhouette_scores, color='red')
ax[2].set_title('Silhouette Score')
ax[2].set_ylabel('Score')

plt.tight_layout()
plt.show()
