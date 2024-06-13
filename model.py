import tensorflow as tf
import tensorflow.keras as keras
import keras.backend as K
import tensorflow.keras.layers as L
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import DenseNet121
!pip install tensorflow-wavelets
import tensorflow_wavelets.Layers.DWT as DWT

class InstanceNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        # Create a scale parameter and a shift parameter for each channel
        self.scale = self.add_weight(
            name='scale',
            shape=(input_shape[-1],),
            initializer='ones',
            trainable=True
        )
        self.shift = self.add_weight(
            name='shift',
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True
        )

    def call(self, inputs):
        # Calculate mean and variance for each channel independently
        mean = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        variance = tf.reduce_mean(tf.square(inputs - mean), axis=[1, 2], keepdims=True)
        
        # Normalize the input
        normalized = (inputs - mean) / tf.sqrt(variance + self.epsilon)
        
        # Apply scale and shift
        output = self.scale * normalized + self.shift
        return output
    
class WeightedGlobalAveragePooling2D(tf.keras.layers.Layer):
    def __init__(self, num_channels, **kwargs):
        super(WeightedGlobalAveragePooling2D, self).__init__(**kwargs)
        self.num_channels = num_channels
        # Create a trainable weight variable for each channel
        self.channel_weights = self.add_weight(name='channel_weights',
                                              shape=(num_channels,),
                                              initializer='ones',
                                              trainable=True)

    def call(self, inputs):
        # Calculate weighted global average pooling
        weighted_sum = tf.reduce_sum(inputs * self.channel_weights, axis=[1, 2])
        weighted_average = weighted_sum / tf.reduce_sum(self.channel_weights)
        return weighted_average

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_channels)
    
class Gradients(L.Layer):
    def call(self, inputs):
        alpha = inputs
        gradients_alpha = tf.gradients(alpha, [alpha])[0]
        gradients_alpha = tf.reduce_mean(gradients_alpha, axis=[-1,-2, -3], keepdims=True)
        return gradients_alpha

def WGCAM(x):
    num_filters = x.shape[-1]
    wav = DWT.DWT(concat=0)(x)
    wav = Conv2DTranspose(num_filters*4, (2, 2), strides=2, padding="same")(wav)
    wav = SeparableConv2D(num_filters, (1,1), padding="same")(wav)
    x_sam = SeparableConv2D(num_filters, (1,1), padding="same")(x) 
    x_sam = wav+x_sam
    x_sam = SeparableConv2D(num_filters, (1,1), padding="same", activation='sigmoid')(x_sam)
    x_cam = WeightedGlobalAveragePooling2D(num_filters)(x)
    x_cam = Dense(num_filters/4, activation='relu')(x_cam)
    x_cam = Dense(num_filters, activation='sigmoid')(x_cam)
    x_cam = tf.keras.layers.Reshape((1, 1, x_cam.shape[-1]))(x_cam)
    x = x*x_sam
    x = tf.keras.layers.Multiply()([x, x_cam])
    return x

def CombinedUpsampleLayer(inputs):
    _,H,W,C = inputs.shape
    gaussian = UpSampling2D(size=(2, 2), interpolation="gaussian")(inputs)
    lanczos = UpSampling2D(size=(2, 2), interpolation="lanczos5")(inputs)
    combined = tf.keras.layers.Add()([gaussian, lanczos])
    combined_attn = Conv2D(C, 1, padding="same")(combined)

    # Assuming you want to upsample to the original input size
    upsampled = Conv2DTranspose(C, (2, 2), strides=2, padding="same")(inputs)

    x = Concatenate()([combined_attn,upsampled])
    return x

def conv_block(inputs, num_filters):
    x1 = Conv2D(num_filters//2, 5, padding="same")(inputs)
    x1 = InstanceNormalization()(x1)
    x1 = Activation("relu")(x1)
    
    x2 = Conv2D(num_filters//2, 3, padding="same")(inputs)
    x2 = InstanceNormalization()(x2)
    x2 = Activation("relu")(x2)
    
    x2 = Concatenate()([x1,x2])
    x2 = Conv2D(num_filters, 1, padding="same")(x2)
    
    x3 = Conv2D(num_filters, 1, padding="same")(inputs)
    x3 = InstanceNormalization()(x3)
    x3 = Activation("relu")(x3)
    
    x3 = Concatenate()([x2,x3])

    x = Conv2D(num_filters, 3, padding="same")(x3)
    x = InstanceNormalization()(x)
    x = Activation("relu")(x)

    return x

def decoder_block(inputs, skip_features, num_filters):
    x = CombinedUpsampleLayer(inputs)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_densenet121_unet(input_shape):
    """ Input """
    inputs = Input(input_shape)

    """ Pre-trained DenseNet121 Model """
    densenet = DenseNet121(include_top=False, weights="imagenet", input_tensor=inputs)

    """ Encoder """
    s1 = densenet.get_layer("input_1").output       ## 512
    s2 = densenet.get_layer("conv1/relu").output    ## 256
    s3 = densenet.get_layer("pool2_relu").output    ## 128
    s4 = densenet.get_layer("pool3_relu").output    ## 64

    """ Bridge """
    b1 = densenet.get_layer("pool4_relu").output  ## 32
    b1 = WGCAM(b1)
    
    """ Decoder """
    d1 = decoder_block(b1, WGCAM(s4), 512)             ## 64
    d2 = decoder_block(d1, WGCAM(s3), 256)             ## 128
    d3 = decoder_block(d2, WGCAM(s2), 128)             ## 256
    d4 = decoder_block(d3, WGCAM(s1), 64)              ## 512

    """ Outputs """
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name='TEACHER')
    return model
    
teacher_model = build_densenet121_unet((512, 512, 3))
optimizer = Adam(lr=0.0001)
teacher_model.compile(loss=combined_loss, metrics=["accuracy", dice_score, recall, precision, iou], optimizer=optimizer)
teacher_model.summary()
