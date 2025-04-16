import tf_keras as keras
import tensorflow_hub as hub
import tensorflow as tf

class MyModel(keras.Model):
  def __init__(self):
    super().__init__()
    self.embed = hub.KerasLayer("https://tfhub.dev/google/nnlm-en-dim50/2")
    self.d1 = keras.layers.Dense(16, activation='relu')
    self.d2 = keras.layers.Dense(1)

  def call(self, inputs):
    # Ensure inputs are batched
    if len(inputs.shape) == 0:
      inputs = tf.expand_dims(inputs, axis=0)

    x = self.embed(inputs)
    x = self.d1(x)
    return self.d2(x)


