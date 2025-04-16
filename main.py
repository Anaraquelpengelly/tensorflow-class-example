import tensorflow_datasets as tfds
from src.train import *

train_ds, val_ds, test_ds = tfds.load(
    name='imdb_reviews',
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True,
    batch_size=32,
)

accuracy_threshold = 0.90
best_accuracy = 0.0

EPOCHS = 10

for epoch in range(EPOCHS):
  # Reset the metrics at the start of the next epoch
  train_loss.reset_state()
  train_accuracy.reset_state()
  test_loss.reset_state()
  test_accuracy.reset_state()

  for texts, labels in train_ds:
    train_step(texts, labels)

  # Changed to use validation data during training
  for val_texts, val_labels in val_ds:
      test_step(val_texts, val_labels)

  print(
      f'Epoch {epoch + 1}, '
      f'Loss: {train_loss.result():0.2f}, '
      f'Accuracy: {train_accuracy.result() * 100:0.2f}%, '
      f'Val Loss: {test_loss.result():0.2f}, '
      f'Val Accuracy: {test_accuracy.result() * 100:0.2f}%'
  )
  val_accuracy = test_accuracy.result().numpy()
  if val_accuracy > accuracy_threshold and val_accuracy > best_accuracy:
      best_accuracy = val_accuracy
      model.save('models/my_best_model')
      print(f"Model saved at epoch {epoch + 1} with accuracy {val_accuracy:.2f}")

# Final evaluation on test set
test_loss.reset_state()
test_accuracy.reset_state()
for test_texts, test_labels in test_ds:
    test_step(test_texts, test_labels)
print(f'\nFinal Test Accuracy: {test_accuracy.result() * 100:.2f}%')


