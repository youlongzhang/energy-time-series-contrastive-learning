import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from combined_model import JointModel,combined_loss
import numpy as np
from Contrastive_Learning.Contrastive_loss import contrastive_loss
from Data_Augmentation.Dtw_loss import dtw_loss
import matplotlib.pyplot as plt

# 设置模型参数
embed_dim = 64  # Transformer嵌入维度
num_heads = 4  # Transformer中的多头注意力头数
ff_dim = 128  # Transformer中的前馈神经网络维度
num_blocks = 2  # Transformer块数
temperature = 0.1  # 对比学习的温度参数
epochs = 50  # 训练的轮数
batch_size = 128  # 每批次样本数
input_shape = (1440, 1)  # 输入形状

# 创建联合模型
joint_model = JointModel(input_shape, embed_dim, num_heads, ff_dim, num_blocks)

# 编译模型
joint_model.compile(optimizer=Adam(learning_rate=1e-4))

# 设置保存最佳模型的回调
checkpoint_contrastive = ModelCheckpoint('best_contrastive_model.h5',
                                         monitor='loss',
                                         save_best_only=True,
                                         mode='min',
                                         save_weights_only=True,
                                         verbose=1)

checkpoint_augmentation = ModelCheckpoint('best_augmentation_model.h5',
                                          monitor='loss',
                                          save_best_only=True,
                                          mode='min',
                                          save_weights_only=True,
                                          verbose=1)



# 创建列表用于存储损失
losses = []
contrastive_losses = []
dtw_losses = []


# 创建自定义回调记录每一轮的损失
class LossHistory(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # 记录损失
        losses.append(logs.get('loss'))
        contrastive_losses.append(logs.get('contrastive_loss'))
        dtw_losses.append(logs.get('dtw_loss'))


# 训练模型并记录损失
history = joint_model.fit(
    X_train,  # 原始序列
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[checkpoint_contrastive, checkpoint_augmentation, LossHistory()]
)


# 绘制损失图
def plot_losses():
    plt.figure(figsize=(12, 6))

    plt.plot(range(epochs), losses, label="Total Loss")
    plt.plot(range(epochs), contrastive_losses, label="Contrastive Loss")
    plt.plot(range(epochs), dtw_losses, label="DTW Loss")

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.show()


# 绘制损失图
plot_losses()

# 加载保存的最优对比学习模型
best_contrastive_model = tf.keras.models.load_model('best_contrastive_model.h5', custom_objects={'contrastive_loss': contrastive_loss})

# 加载保存的最优数据增强模型
best_dtw_model = tf.keras.models.load_model('best_dtw_model.h5', custom_objects={'dtw_loss': dtw_loss})
