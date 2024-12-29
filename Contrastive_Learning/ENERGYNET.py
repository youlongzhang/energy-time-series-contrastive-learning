from tensorflow.keras.layers import Input, Conv1D, MaxPool1D, Bidirectional, LSTM, Dense
from tensorflow.keras.models import Model
# 定义输入数据的形状
input_shape = (1440, 1)
# 定义编码器模型
def create_encoder():
    inputs = Input(shape=input_shape)
    x = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
    x = MaxPool1D(pool_size=2)(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Bidirectional(LSTM(64, return_sequences=False))(x)
    encoder = Model(inputs, x, name='encoder')
    return encoder

# 定义投影头模型
def create_projection_head():
    inputs = Input(shape=(128,))
    x = Dense(64, activation='relu')(inputs)
    outputs = Dense(64)(x)
    projection_head = Model(inputs, outputs, name='projection_head')
    return projection_head
