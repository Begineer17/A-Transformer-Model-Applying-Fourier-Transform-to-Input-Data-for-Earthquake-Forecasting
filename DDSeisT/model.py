import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, initializers
import numpy as np

class ModelConfig:
    def __init__(self, batch_size=20, 
                 depths=6, filters_root=8, kernel_size=[3, 3], 
                 pool_size=[2, 2], dilation_rate=[1, 1], 
                 class_weights=[1.0, 1.0, 1.0], loss_type='mse', 
                 weight_decay=0.0, optimizer='adam', learning_rate=0.001, 
                 drop_rate=0.15, X_shape=[31, 1002, 2], n_channel=2, 
                 Y_shape=[31, 1002, 2], n_class=2):
        
        self.batch_size = batch_size
        self.depths = depths
        self.filters_root = filters_root
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.dilation_rate = dilation_rate
        self.class_weights = class_weights
        self.loss_type = loss_type
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.drop_rate = drop_rate
        self.X_shape = X_shape
        self.n_channel = n_channel
        self.Y_shape = Y_shape
        self.n_class = n_class

    def get_config(self):
        return {
            'batch_size': self.batch_size,
            'depths': self.depths,
            'filters_root': self.filters_root,
            'kernel_size': self.kernel_size,
            'pool_size': self.pool_size,
            'dilation_rate': self.dilation_rate,
            'class_weights': self.class_weights,
            'loss_type': self.loss_type,
            'weight_decay': self.weight_decay,
            'optimizer': self.optimizer,
            'learning_rate': self.learning_rate,
            'drop_rate': self.drop_rate,
            'X_shape': self.X_shape,
            'n_channel': self.n_channel,
            'Y_shape': self.Y_shape,
            'n_class': self.n_class
        }

def crop_and_concat(net1, net2):
    net1_shape = tf.shape(net1)
    net2_shape = tf.shape(net2)
    height_diff = (net2_shape[1] - net1_shape[1]) // 2
    width_diff = (net2_shape[2] - net1_shape[2]) // 2

    # Adjust net2 to match net1 by cropping
    net2_cropped = tf.slice(
        net2,
        [0, height_diff, width_diff, 0],
        [-1, net1_shape[1], net1_shape[2], -1]
    )
    return tf.concat([net1, net2_cropped], axis=-1)

class UNet(tf.keras.Model):
    def __init__(self, config=ModelConfig(), **kwargs):
        super(UNet, self).__init__(**kwargs)
        self.config = config
        self.depths = config.depths
        self.filters_root = config.filters_root
        self.kernel_size = config.kernel_size
        self.pool_size = config.pool_size
        self.dilation_rate = config.dilation_rate
        self.n_class = config.n_class
        self.drop_rate = config.drop_rate
        self.build_model()

    def get_config(self):
        config = super(UNet, self).get_config()
        config.update({
            'config': self.config.__dict__,  # Convert ModelConfig instance to dictionary
        })
        return config

    @classmethod
    def from_config(cls, config):
        config_obj = ModelConfig(**config['config'])  # Reconstruct ModelConfig instance from dictionary
        return cls(config=config_obj)

    def build_model(self):
        # Down-sampling (Encoder) Layers
        self.down_convs = []
        self.down_norms = []
        self.down_drops = []
        for depth in range(self.depths):
            filters = int(2 ** depth * self.filters_root)
            down_conv = layers.Conv2D(
                filters, self.kernel_size, padding='same',
                dilation_rate=self.dilation_rate,
                kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(self.config.weight_decay)
            )
            down_norm = layers.BatchNormalization()
            down_drop = layers.Dropout(self.drop_rate)

            self.down_convs.append(down_conv)
            self.down_norms.append(down_norm)
            self.down_drops.append(down_drop)

        # Up-sampling (Decoder) Layers
        self.up_convs = []
        self.up_norms = []
        self.up_drops = []
        self.concat_convs = []
        self.concat_norms = []
        self.concat_drops = []  # Additional Dropout layers for concatenated outputs
        for depth in range(self.depths - 2, -1, -1):
            filters = int(2 ** depth * self.filters_root)
            up_conv = layers.Conv2DTranspose(
                filters, self.kernel_size, strides=self.pool_size,
                padding='same', kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(self.config.weight_decay)
            )
            up_norm = layers.BatchNormalization()
            up_drop = layers.Dropout(self.drop_rate)

            concat_conv = layers.Conv2D(filters, self.kernel_size, padding='same')
            concat_norm = layers.BatchNormalization()
            concat_drop = layers.Dropout(self.drop_rate)

            self.up_convs.append(up_conv)
            self.up_norms.append(up_norm)
            self.up_drops.append(up_drop)
            self.concat_convs.append(concat_conv)
            self.concat_norms.append(concat_norm)
            self.concat_drops.append(concat_drop)

        # Final Output Layer
        self.output_conv = layers.Conv2D(self.n_class, (1, 1), activation=None, padding='same')

    def call(self, inputs, training=False):
        convs = []
        x = inputs
        print(f"Input Shape: {x.shape}")

        # Encoder: Down-sampling path
        for depth in range(self.depths):
            x = self.down_convs[depth](x)
            x = self.down_norms[depth](x, training=training)
            x = layers.ReLU()(x)
            x = self.down_drops[depth](x, training=training)
            convs.append(x)
            print(f"After down_conv {depth} shape: {x.shape}")

            # Only apply MaxPooling if we are not at the last down-sampling layer
            if len(convs) < self.depths:
                if x.shape[1] > 1 and x.shape[2] > 1:
                    x = layers.MaxPooling2D(self.pool_size, padding="same")(x)
                    print(f"After MaxPooling2D {depth} shape: {x.shape}")

        # Decoder: Up-sampling path
        for i in range(len(self.up_convs)):
            x = self.up_convs[i](x)
            x = self.up_norms[i](x, training=training)
            x = layers.ReLU()(x)
            x = self.up_drops[i](x, training=training)
            print(f"After up_conv {i} shape: {x.shape}")

            # Concatenate with corresponding encoder feature map
            x = crop_and_concat(convs[-(i + 2)], x)
            print(f"After concat {i} shape: {x.shape}")

            # Use predefined Conv2D, BatchNormalization, and Dropout layers after concatenation
            x = self.concat_convs[i](x)
            x = self.concat_norms[i](x, training=training)
            x = layers.ReLU()(x)
            x = self.concat_drops[i](x, training=training)
            print(f"After Conv-BatchNorm-ReLU-Dropout {i} shape: {x.shape}")

        # Final Output Layer
        x = self.output_conv(x)
        print(f"Output Shape: {x.shape}")
        return x

# Ensure that the UNet class is properly registered
tf.keras.utils.get_custom_objects().update({'UNet': UNet})