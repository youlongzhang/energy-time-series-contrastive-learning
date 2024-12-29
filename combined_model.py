from Data_Augmentation.Dtw_loss import dtw_loss
from Data_Augmentation.self_attention_augmentation import create_transformer_augmented_model
from Contrastive_Learning.Contrastive_loss import contrastive_loss
from Contrastive_Learning.ENERGYNET import create_encoder,create_projection_head
import tensorflow as tf

def combined_loss(z1, z2, y_true, y_pred, temperature=0.1,contrastive_weight=0.6, dtw_weight=0.4):
    """
    计算联合损失：对比损失 + DTW损失

    参数：
        z1, z2：对比学习的输入表示
        y_true, y_pred：DTW损失的输入序列
        temperature：对比损失的温度系数
        dtw_weight：DTW损失的权重系数
        contrastive_weight:对比损失的权重系数
    返回：
        联合损失值
    """
    # 计算对比损失
    contrastive_loss_value = contrastive_loss(z1, z2, temperature)

    # 计算DTW损失
    dtw_loss_value = dtw_loss(y_true, y_pred)

    # 计算联合损失
    total_loss = contrastive_loss_value*contrastive_weight + dtw_weight * dtw_loss_value

    return total_loss


class JointModel(tf.keras.Model):
    def __init__(self, input_shape, embed_dim, num_heads, ff_dim, num_blocks):
        super(JointModel, self).__init__()
        self.input_shape = input_shape

        # 创建子模型
        self.transformer_model = create_transformer_augmented_model(input_shape, embed_dim, num_heads, ff_dim,
                                                                    num_blocks)
        self.encoder = create_encoder()
        self.projection_head = create_projection_head()

    def call(self, inputs):
        original_input = inputs  # 原始序列输入

        # 生成增强视图
        augmented_input = self.transformer_model(original_input)

        # 编码原始序列和增强序列
        encoded_original = self.encoder(original_input)
        encoded_augmented = self.encoder(augmented_input)

        # 投影
        projected_original = self.projection_head(encoded_original)
        projected_augmented = self.projection_head(encoded_augmented)

        return projected_original, projected_augmented, augmented_input

    def train_step(self, data):
        # 获取原始序列输入
        original_input = data  # 原始序列

        with tf.GradientTape() as tape:
            # 前向传播：生成增强视图，计算投影
            projected_original, projected_augmented, augmented_output = self(original_input)

            # 计算联合损失
            # 使用增强后的数据作为目标进行DTW计算
            loss = combined_loss(
                projected_original,
                projected_augmented,
                original_input,  # 原始序列作为目标
                augmented_output,  # 增强视图作为预测
                temperature=0.1,
                contrastive_weight=0.6,
                dtw_weight=0.4
            )

        # 计算梯度
        grads = tape.gradient(loss, self.trainable_variables)

        # 更新模型权重
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # 打印损失
        print(
            f"Combined Loss: {loss:.4f}, Contrastive Loss: {contrastive_loss(projected_original, projected_augmented):.4f}, DTW Loss: {dtw_loss(original_input, augmented_output):.4f}")

        return {"total_loss": loss}
