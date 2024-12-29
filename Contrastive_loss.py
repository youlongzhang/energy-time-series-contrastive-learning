import tensorflow as tf

# 定义对比学习损失函数
def contrastive_loss(z1, z2, temperature=0.1):
    batch_size = tf.shape(z1)[0]
    z1 = tf.math.l2_normalize(z1, axis=1)
    z2 = tf.math.l2_normalize(z2, axis=1)

    # 计算余弦相似度
    cosine_similarity_1 = tf.matmul(z1, z2, transpose_b=True) / temperature
    cosine_similarity_2 = tf.matmul(z2, z1, transpose_b=True) / temperature

    # 正样本对
    positive_pairs = tf.linalg.diag_part(cosine_similarity_1)

    # 负样本对（所有其他样本对）
    negative_pairs_1 = cosine_similarity_1 - tf.linalg.diag(positive_pairs)
    negative_pairs_2 = cosine_similarity_2 - tf.linalg.diag(positive_pairs)

    # 计算每个样本的损失
    loss = -tf.math.log(
        tf.exp(positive_pairs) /
        (tf.reduce_sum(tf.exp(negative_pairs_1), axis=1) + tf.reduce_sum(tf.exp(negative_pairs_2), axis=1))
    )

    return tf.reduce_mean(loss)