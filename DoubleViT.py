import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import tensorflow_datasets as tfds
import numpy as np

# Load the data 'CIFAR10'
train_ds = tfds.load('cifar10', split='train', batch_size=128, as_supervised=True)
test_ds = tfds.load('cifar10', split='test', batch_size=128, as_supervised=True)

# Normalize the data and resize the data
def data_preparation(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.divide(image, 255)
    image = tf.image.resize(image, (32, 32))
    label = tf.one_hot(label, depth=10)
    return image, label

# Dataset augmentation
def data_augmentation(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_saturation(image, 0.7, 1.3)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_brightness(image, 0.1)
    return image, label 

# Data preprocessing stage
train = train_ds.map(data_preparation).cache().map(data_augmentation)
test = test_ds.map(data_preparation).cache().map(data_augmentation)


class MLP(layers.Layer):

    def __init__(self, mlp_expand_ratio, mlp_dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.mlp_expand_ratio = mlp_expand_ratio
        self.mlp_dropout_rate = mlp_dropout_rate

    def build(self, input_shape):
        input_channels = input_shape[-1]
        initial_filters = int(self.mlp_expand_ratio * input_channels)

        self.mlp = tf.keras.Sequential(
            [
                layers.Dense(
                    units=initial_filters,
                    activation=tf.nn.gelu,
                ),
                layers.Dropout(rate=self.mlp_dropout_rate),
                layers.Dense(units=input_channels),
                layers.Dropout(rate=self.mlp_dropout_rate),
            ]
        )

    def call(self, x):
        x = self.mlp(x)
        return x

class DropPath(layers.Layer):
    
    def __init__(self, drop_path_prob, **kwargs):
        super().__init__(**kwargs)
        self.drop_path_prob = drop_path_prob

    def call(self, x, training=False):
        if training:
            keep_prob = 1 - self.drop_path_prob
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x

# ShiftViT Block 
class ShiftViTBlock(layers.Layer):
    """
    This block uses Shift operation to the layer.
    The attention layer is substituted by the shift operations.
    It does not contain any parameter or arithmetic calculation.
    It exchanges a small portion of the channels between neighboring features.
    Refer Paper : When shift operation meets vision transformer: An extremely simple alternative to attention mechanism.
    Authors: Wang, G., Zhao, Y., Tang, C., Luo, C. and Zeng, W., 2022, June.
    In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 36, No. 2, pp. 2423-2430)
    """
    def __init__(
        self,
        epsilon,
        drop_path_prob,
        mlp_dropout_rate,
        num_div=12,
        shift_pixel=1,
        mlp_expand_ratio=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.shift_pixel = shift_pixel
        self.mlp_expand_ratio = mlp_expand_ratio
        self.mlp_dropout_rate = mlp_dropout_rate
        self.num_div = num_div
        self.epsilon = epsilon
        self.drop_path_prob = drop_path_prob

    def build(self, input_shape):
        self.H = input_shape[1]
        self.W = input_shape[2]
        self.C = input_shape[3]
        self.layer_norm = layers.LayerNormalization(epsilon=self.epsilon)
        self.drop_path = (
            DropPath(drop_path_prob=self.drop_path_prob)
            if self.drop_path_prob > 0.0
            else layers.Activation("linear")
        )
        self.mlp = MLP(
            mlp_expand_ratio=self.mlp_expand_ratio,
            mlp_dropout_rate=self.mlp_dropout_rate,
        )

    def get_shift_pad(self, x, mode):
        """Shifts the channels according to the mode chosen."""
        if mode == "left":
            offset_height = 0
            offset_width = 0
            target_height = 0
            target_width = self.shift_pixel
        elif mode == "right":
            offset_height = 0
            offset_width = self.shift_pixel
            target_height = 0
            target_width = self.shift_pixel
        elif mode == "up":
            offset_height = 0
            offset_width = 0
            target_height = self.shift_pixel
            target_width = 0
        else:
            offset_height = self.shift_pixel
            offset_width = 0
            target_height = self.shift_pixel
            target_width = 0
        crop = tf.image.crop_to_bounding_box(
            x,
            offset_height=offset_height,
            offset_width=offset_width,
            target_height=self.H - target_height,
            target_width=self.W - target_width,
        )
        shift_pad = tf.image.pad_to_bounding_box(
            crop,
            offset_height=offset_height,
            offset_width=offset_width,
            target_height=self.H,
            target_width=self.W,
        )
        return shift_pad

    def call(self, x, training=False):
        # Feature maps are being split
        x_splits = tf.split(x, num_or_size_splits=self.C // self.num_div, axis=-1)

        # Shift the feature maps
        x_splits[0] = self.get_shift_pad(x_splits[0], mode="left")
        x_splits[1] = self.get_shift_pad(x_splits[1], mode="right")
        x_splits[2] = self.get_shift_pad(x_splits[2], mode="up")
        x_splits[3] = self.get_shift_pad(x_splits[3], mode="down")

        # Concatenate the shifted and unshifted feature maps
        x = tf.concat(x_splits, axis=-1)

        # Add the residual connection
        shortcut = x
        x = shortcut + self.drop_path(self.mlp(self.layer_norm(x)), training=training)
        return x

# Parameters
patch_size = 3
projected_dim = 192
num_shift_blocks = 4
epsilon = 1e-5
training_epochs = 100
stochastic_depth_rate = 0.2
mlp_dropout_rate = 0.2
num_div = 12
shift_pixel = 1
mlp_expand_ratio = 2

def DoubleViT():
    """
    Proposed model DoubleViT: Pushing transformers towards the end because of convolutions.
    Network begins with convolutional layers and concludes with ShiftViT method.
    The convolutional layers and their depth are determined based on input shapes.
    Shift mechanism transforms the outputs of the convolution layers into patches before performing Shift operations.
    Shift method acts as substitute to the attention mechanism in ViTs.
    """
    img_input = layers.Input(shape=(32,32,3))
    x = Doublevit_stack(img_input)
    x = layers.Conv2D(
            filters=projected_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="same",
        )(x)
    
    dpr = [
        x
        for x in np.linspace(
            start=0, stop=stochastic_depth_rate, num=num_shift_blocks
        )
    ]
    for repeat in range(8):
        x = ShiftViTBlock(
            num_div=num_div,
            epsilon=epsilon,
            drop_path_prob=dpr[1],
            mlp_dropout_rate=mlp_dropout_rate,
            shift_pixel=shift_pixel,
            mlp_expand_ratio=mlp_expand_ratio,
        )(x)

    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)

    x = layers.Dense(
        10, activation='sigmoid', name="predictions"
    )(x)

    model = tf.keras.Model(img_input, x)

    return model


def Doublevit_block(x, filters, kernel_size=3, stride=1, conv_shortcut=True,name=None):
    if conv_shortcut:
        shortcut = layers.Conv2D(
            filters, 1, strides=stride, name=name + "_0_convsh"
        )(x)
        shortcut = layers.BatchNormalization(
            epsilon=1.001e-5, name=name + "_0_bnsh"
        )(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(filters, 1, strides=stride, name=name + "_1_conv")(x)
    x = layers.BatchNormalization(
        epsilon=1.001e-5, name=name + "_1_bn"
    )(x)
    x = layers.Activation("relu", name=name + "_1_relu")(x)

    x = layers.Conv2D(
        filters, kernel_size, padding="SAME", name=name + "_2_conv"
    )(x)
    x = layers.BatchNormalization(
        epsilon=1.001e-5, name=name + "_2_bn"
    )(x)
    x = layers.Activation("relu", name=name + "_2_relu")(x)

    x = layers.Add(name=name + "_add")([shortcut, x])
    x = layers.Activation("relu", name=name + "_out")(x)
    return x


def Doublevit_stack(x, stride1=2, name='Base'):
    # First 4 set of blocks consider input shapes and derive layers accordingly as mentioned in paper.
    filters=32
    blocks=4
    counter=1
    x = Doublevit_block(x, filters, stride=stride1,name=name+"_Doublevit_block_0")
    for i in range(1, blocks):
        x = Doublevit_block(
            x, filters, conv_shortcut=False, name=name + "_Doublevit_block_0" + str(counter)
        )
        counter+=1
    # Next set of 8 blocks performs with increased filters size.
    filters=64
    blocks=8
    x = Doublevit_block(x, filters, stride=stride1,name=name+"_Doublevit_block_1")
    for i in range(1, blocks):
        x = Doublevit_block(
            x, filters, conv_shortcut=False, name=name + "_Doublevit_block_1" + str(counter)
        )
        counter+=1
    return x


model = DoubleViT()
model.build(input_shape=(None,32,32,3))
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy'],
)

history = model.fit(train, epochs=training_epochs,validation_data = test)