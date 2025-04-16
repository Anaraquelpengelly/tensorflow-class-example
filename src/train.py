from src.model import MyModel
import tf_keras as keras
import tensorflow as tf

model = MyModel()
loss_object = keras.losses.BinaryCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam()

train_loss = keras.metrics.Mean(name='train_loss')
train_accuracy = keras.metrics.BinaryAccuracy(name='train_accuracy')

test_loss = keras.metrics.Mean(name='test_loss')
test_accuracy = keras.metrics.BinaryAccuracy(name='test_accuracy')

@tf.function
def train_step(texts, labels):
    with tf.GradientTape() as tape:
        predictions = model(texts, training=True)
        loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

@tf.function
def test_step(texts, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = model(texts, training=False)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  # Convert logits to probabilities for accuracy
  test_accuracy(labels, tf.sigmoid(predictions))