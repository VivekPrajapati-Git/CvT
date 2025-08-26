import tensorflow as tf
from tensorflow.keras.layers import DepthwiseConv2D, BatchNormalization, Dense, Activation, Conv2D, LayerNormalization
from tensorflow.keras.layers import Add, Input, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import mixed_precision
import matplotlib.pyplot as plt

# Convolutional Embedding
def convolutional_embedding(x, filters , stride=1 , kernel_size = 3):
  x = Conv2D(filters , kernel_size , strides=stride , padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  return x

# Linear Projection using Convolution
class ConvAttention(tf.keras.layers.Layer):
  def __init__(self , num_heads , embed_dim):
    super(ConvAttention,self).__init__()
    self.num_heads = num_heads
    self.embed_dim = embed_dim
    self.scale = self.embed_dim ** -0.5

    self.query = DepthwiseConv2D(kernel_size = 3 , padding='same')
    self.key = DepthwiseConv2D(kernel_size = 3, padding='same')
    self.value = DepthwiseConv2D(kernel_size=3 , padding='same')

    self.projection = Dense(embed_dim)

  def call(self,x):
    batch_size , height , width , channels = tf.shape(x)[0] , tf.shape(x)[1] , tf.shape(x)[2] , tf.shape(x)[3]

    query = self.query(x)
    key = self.key(x)
    values = self.value(x)

    query = tf.reshape(query , (batch_size ,-1 , channels))
    values = tf.reshape(values , (batch_size ,-1 , channels))
    key = tf.reshape(key , (batch_size ,-1 , channels))

    attention_weights = tf.matmul(query , key , transpose_b = True) / self.scale
    att_weight = tf.nn.softmax(attention_weights , axis= -1)

    context = tf.matmul(att_weight , values)
    output = tf.reshape(context , [batch_size,height,width,channels])
    return self.projection(output)

# Transformer Block
def transformerBlock(x , embed_dim , num_heads , mlp_ratio = 4):
  shortcut = x

  x = LayerNormalization()(x)
  x = ConvAttention(num_heads,embed_dim)(x)
  x = Add()([x,shortcut])

  shortcut = x

  x = LayerNormalization()(x)
  x = Conv2D(embed_dim * mlp_ratio , 1 , activation='relu')(x)
  x = Conv2D(embed_dim , 1)(x)
  x = Add()([x,shortcut])

  return x

# Final Model of all layers
def built_cvt(input_shape = (224,224,3),num_classes = 1):
  input = Input(input_shape)

  x = convolutional_embedding(input, 64, stride=2)    # Stage 1
  for _ in range(1):
    x = transformerBlock(x, 64, num_heads=1)

  x = convolutional_embedding(x, 128, stride=2)        # Stage 2
  for _ in range(2):
    x = transformerBlock(x, 128, num_heads=2)

  x = convolutional_embedding(x, 256, stride=2)        # Stage 3
  for _ in range(4):
    x = transformerBlock(x, 256, num_heads=4)

  x = GlobalAveragePooling2D()(x)
  x = Dense(256, activation='relu')(x)
  x = Dropout(0.5)(x)
  x = Dense(num_classes , activation='sigmoid')(x)

  model = tf.keras.Model(input , x)
  return model
