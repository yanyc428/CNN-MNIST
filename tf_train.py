import tensorflow.keras as keras
from dataPreprocessor import *
import tensorflow as tf
import numpy as np
import visualization as v
import os

BATCH_SIZE = 128

train_data_file = './data/train/train-images-idx3-ubyte'
train_label_file = './data/train/train-labels-idx1-ubyte'
test_data_file = './data/test/t10k-images-idx3-ubyte'
test_label_file = './data/test/t10k-labels-idx1-ubyte'


def reader_dataset(data, label, idx3=True, batch_size=128, shuffle_buffer_size=10000):

    data_array = raw_file_idx3_process(data)
    label_array = raw_file_idx1_process(label)
    if idx3:
        data_array = data_array.reshape(-1, 1).reshape(60000, 28, 28, -1)
    else:
        data_array = data_array.reshape(-1, 1).reshape(10000, 28, 28, -1)
    dataset = tf.data.Dataset.from_tensor_slices((data_array, label_array))
    dataset = dataset.repeat()
    dataset = dataset.shuffle(shuffle_buffer_size).batch(batch_size)
    return dataset


train_set = reader_dataset(train_data_file, train_label_file)
test_set = reader_dataset(test_data_file, test_label_file, False)

model = keras.Sequential([
    keras.layers.Conv2D(filters=20, kernel_size=(5, 5), padding="VALID", input_shape=(28, 28, 1), strides=1, activation='relu'),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Conv2D(filters=50, kernel_size=(5, 5), padding="VALID", strides=1, activation='relu'),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(500, activation='relu'),
    keras.layers.Dense(10, activation="softmax"),
])

# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(28, 28, 1)),
#     keras.layers.Dense(500, activation='relu'),
#     keras.layers.Dense(100, activation='relu'),
#     keras.layers.Dense(50, activation='relu'),
#     keras.layers.Dense(10, activation="softmax"),
# ])

model.summary()

log_dir = "./callback"
if os.path.exists(log_dir):
    pass
else:
    os.mkdir(log_dir)

log_file = os.path.join(log_dir, 'mnist.h5')

callback = [
    tf.keras.callbacks.TensorBoard(log_dir),
    tf.keras.callbacks.ModelCheckpoint(log_file, save_best_only=True),
    tf.keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3)
]
model.compile(
    optimizer="sgd",
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

history = model.fit(train_set,
                    epochs=5,
                    steps_per_epoch=60000//BATCH_SIZE,
                    validation_steps=10000//BATCH_SIZE,
                    validation_data=test_set,
                    callbacks=callback)

model.evaluate(test_set,steps=10000//BATCH_SIZE, verbose=2)