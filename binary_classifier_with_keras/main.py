import numpy as np

from keras.datasets import imdb
from keras import models, layers, optimizers

import matplotlib.pyplot as plt


def vectorize(sequences, dim=10000):
    results = np.zeros((len(sequences), dim))
    for i, sequences in enumerate(sequences):
        results[i, sequences]=1
    return results


if __name__ == '__main__':
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
    print(train_data.shape, train_data[0], train_labels[0])

    # Conversion to get the name of the movies from numbers
    word_index = imdb.get_word_index()
    word_index = dict([(value, key) for (key, value) in word_index.items()])
    for _ in train_data[0]:
        print('-'*32)
        print(word_index.get(_ - 3))
        print('-'*32)
    x_train = vectorize(train_data)
    x_test = vectorize(test_data)

    y_train = np.asarray(train_labels).astype('float32')
    y_test = np.asarray(test_labels).astype('float32')

    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics='accuracy')

    x_val = x_train[:10000]
    partial_x_train = x_train[10000:]

    y_val = y_train[:10000]
    partial_y_val = y_train[10000:]

    history = model.fit(partial_x_train, partial_y_val, epochs=4, batch_size=512, validation_data=(x_val, y_val))

    history_dict = history.history

    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']

    epoch = range(1, len(loss_values)+1)
    plt.plot(epoch, loss_values, 'o', label= 'training')
    plt.plot(epoch, val_loss_values, '--', label= 'training')
    plt.legend()
    plt.show()

    print('-'*32)
    print(model.evaluate(x_test, y_test))
