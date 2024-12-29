import tensorflow as tf

def dtw_loss(y_true, y_pred):
    """计算DTW距离，作为损失函数"""
    batch_size = tf.shape(y_true)[0]
    seq_len = tf.shape(y_true)[1]

    def compute_dtw(y_true_seq, y_pred_seq):
        # 计算每个序列对的DTW距离
        dtw_matrix = tf.TensorArray(dtype=tf.float32, size=seq_len)
        for i in range(seq_len):
            cost = tf.abs(y_true_seq[i] - y_pred_seq)
            dtw_matrix = dtw_matrix.write(i, cost)
        dtw_matrix = dtw_matrix.stack()
        dtw_matrix = tf.reduce_min(dtw_matrix, axis=0)# 动态规划找出最优路径
        return tf.reduce_sum(dtw_matrix)

    # 对每个批次中的序列计算DTW
    losses = tf.TensorArray(dtype=tf.float32, size=batch_size)
    for i in range(batch_size):
        loss = compute_dtw(y_true[i], y_pred[i])
        losses = losses.write(i, loss)

    return tf.reduce_mean(losses.stack())
