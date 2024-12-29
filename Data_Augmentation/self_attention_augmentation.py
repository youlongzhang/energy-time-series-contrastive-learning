import tensorflow as tf
from tensorflow.keras.layers import Dense, MultiHeadAttention, LayerNormalization, Dropout
from tensorflow.keras import Sequential

#注意力机制
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1,**kwargs):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential([Dense(ff_dim, activation="relu"), Dense(embed_dim)])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# 自动增强层
class AutoAugmentLayer(tf.keras.layers.Layer):
    def __init__(self, rate=0.1,**kwargs):
        super(AutoAugmentLayer, self).__init__()
        self.rate = rate

    def call(self, inputs, attention_scores):
        # attention_scores 表示注意力得分，指导增强的程度
        attention_scores = tf.expand_dims(attention_scores, axis=-1)
        augmented_inputs = inputs * (1 + attention_scores * self.rate)  # 根据得分对输入增强
        return augmented_inputs

    def get_config(self):
        config = super(AutoAugmentLayer, self).get_config()
        config.update({
            'rate': self.rate,
        })
        return config

# 端到端的增强模型
def create_transformer_augmented_model(input_shape, embed_dim, num_heads, ff_dim, num_blocks):
    inputs = tf.keras.Input(shape=input_shape)

    # Transformer编码
    embedding_layer = Dense(embed_dim)(inputs)
    x = embedding_layer
    for _ in range(num_blocks):
        x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)

    # 获取注意力得分
    attention_scores = tf.reduce_mean(x, axis=-1)  # 平均每个时间步的特征表示

    # 自动增强层，根据注意力得分对输入进行增强
    augmented_output = AutoAugmentLayer(rate=0.1)(inputs, attention_scores)

    # 输出增强后的时间序列
    outputs = Dense(input_shape[-1])(augmented_output)  # 输出维度与输入保持一致

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
