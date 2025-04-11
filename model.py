import tf_keras as keras
import tensorflow_hub as hub
from tensorflow import string

class MyModel(keras.Model):
  def __init__(self):
    super().__init__()
    self.embed = hub.keras_layer(model = "https://tfhub.dev/google/nnlm-en-dim50/2",input_shape=[], dtype=string, trainable=True)
    self.d1 = keras.layers.Dense(16, activation='relu')
    self.d2 = keras.layers.Dense(1)

  def call(self, x):
    x = self.embed(x)
    x = self.d1(x)
    return self.d2(x)

