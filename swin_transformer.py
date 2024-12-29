import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt

# 데이터 로드 및 전처리
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.2,
    zoom_range=1.0  # 데이터 증강: 확대
)


#  데이터 생성기 설정
train_generator = datagen.flow(x_train, y_train, batch_size=64, subset='training')
val_generator = datagen.flow(x_train, y_train, batch_size=64, subset='validation')

for x_batch, y_batch in train_generator:
    print(f"Train Generator: x_batch shape={x_batch.shape}, y_batch shape={y_batch.shape}")
    break

for x_batch, y_batch in val_generator:
    print(f"Validation Generator: x_batch shape={x_batch.shape}, y_batch shape={y_batch.shape}")
    break

# 윈도우 분할 함수 정의
def window_partition(x, window_size):
    B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    x = tf.pad(x, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]])
    H_padded, W_padded = H + pad_h, W + pad_w
    x = tf.reshape(x, [B, H_padded // window_size, window_size, W_padded // window_size, window_size, C])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, [-1, window_size * window_size, C])
    return x, H_padded, W_padded

#  윈도우 복원 함수 정의
def window_reverse(windows, window_size, H_padded, W_padded, H, W, C):
    B = tf.shape(windows)[0] // (H_padded // window_size * W_padded // window_size)
    x = tf.reshape(windows, [B, H_padded // window_size, W_padded // window_size, window_size, window_size, C])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, [B, H_padded, W_padded, C])
    x = x[:, :H, :W, :]
    return x

# 윈도우 어텐션 클래스 정의
class WindowAttention(layers.Layer):
    def __init__(self, dim, window_size, num_heads):
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        assert dim % num_heads == 0, 'dim must be divisible by num_heads'
        self.head_dim = dim // num_heads
        self.scale = tf.math.pow(tf.cast(self.head_dim, tf.float32), -0.5)
        self.query = layers.Dense(dim, use_bias=False)
        self.key = layers.Dense(dim, use_bias=False)
        self.value = layers.Dense(dim, use_bias=False)
        self.proj = layers.Dense(dim)

    def call(self, x, mask=None):
        B_, N, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        tf.debugging.assert_equal(C, self.dim, message=f'Input embedding dim ({C}) does not match layer embedding dim ({self.dim})')
        tf.debugging.assert_equal(N, self.window_size ** 2, message=f"Sequence length ({N}) does not match window size squared ({self.window_size ** 2})")
        
        q = tf.reshape(self.query(x), [B_, N, self.num_heads, self.head_dim])
        k = tf.reshape(self.key(x), [B_, N, self.num_heads, self.head_dim])
        v = tf.reshape(self.value(x), [B_, N, self.num_heads, self.head_dim])

        q = tf.transpose(q, [0, 2, 1, 3]) * self.scale
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])

        attn = tf.matmul(q, k, transpose_b=True)
        if mask is not None:
            attn += mask
        attn = tf.nn.softmax(attn, axis=-1)
        out = tf.matmul(attn, v)

        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, [B_, N, C])
        return self.proj(out)

#  SwinTransformerBlock 수정
class SwinTransformerBlock(layers.Layer):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0, dropout_rate=0.1):
        super(SwinTransformerBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.mlp = tf.keras.Sequential([
            layers.Dense(int(dim * mlp_ratio), activation='gelu'),
            layers.Dropout(dropout_rate),  # 드롭아웃 추가
            layers.Dense(dim)
        ])
        self.batch_norm = layers.BatchNormalization()  # Batch Normalization 추가

    def call(self, x):
        B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]

        shortcut = x
        # Layer Normalization + Attention
        x = self.norm1(x)
        x_windows, H_padded, W_padded = window_partition(x, self.window_size)
        attn_windows = self.attn(x_windows)
        x = window_reverse(attn_windows, self.window_size, H_padded, W_padded, H, W, C)
        x = x + shortcut
        x = self.batch_norm(x)  # Batch Normalization 적용

        # Layer Normalization + MLP
        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + shortcut
        x = self.batch_norm(x)  # Batch Normalization 적용
        return x

#  SwinTransformer 모델 수정
class SwinTransformer(models.Model):
    def __init__(self, img_size=32, patch_size=4, num_classes=10, depths=[2, 2, 5, 2], num_heads=[3, 6, 12, 24],
                 dropout_rate=0.3, l2_reg=1e-4):  # 드롭아웃과 L2 정규화 추가
        super(SwinTransformer, self).__init__()
        self.patch_embed = layers.Conv2D(96, kernel_size=patch_size, strides=patch_size, padding='valid',
                                         kernel_regularizer=tf.keras.regularizers.l2(l2_reg))  # L2 정규화 추가
        self.pos_drop = layers.Dropout(dropout_rate)

        self.blocks = []
        dim = 96
        for i in range(len(depths)):
            for j in range(depths[i]):
                shift_size = 0 if j % 2 == 0 else patch_size // 2
                self.blocks.append(
                    SwinTransformerBlock(
                        dim=dim, num_heads=num_heads[i], window_size=4, shift_size=shift_size, dropout_rate=dropout_rate
                    )
                )
            if i < len(depths) - 1:
                dim = dim * 2
                self.blocks.append(layers.Conv2D(dim, kernel_size=2, strides=2, padding='valid',
                                                 kernel_regularizer=tf.keras.regularizers.l2(l2_reg)))  # L2 정규화 추가

        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.avgpool = layers.GlobalAveragePooling2D()
        self.head = layers.Dense(num_classes, activation='softmax',
                                 kernel_regularizer=tf.keras.regularizers.l2(l2_reg))  # L2 정규화 추가

    def call(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.avgpool(x)
        x = self.head(x)
        return x

# 모델 학습
model = SwinTransformer(img_size=32, patch_size=4, num_classes=10)
model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

lr_reducer = ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=100,
    callbacks=[lr_reducer]
)
