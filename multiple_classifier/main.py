import numpy as np
from keras import layers, models
from keras.datasets import reuters
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt

def vectorize(sequences, dim=10000):
    restults = np.zeros((len(sequences),dim))
    for i, sequences in enumerate(sequences):
        restults[i,sequences]=1
    return restults


if __name__ == '__main__':
    (train_data, test_data), (train_labels, test_labels) = reuters.load_data(num_words=10000)

    print(train_data[0])

    word_index = reuters.get_word_index()
    word_index = dict([(value,key) for (key,value) in word_index.items()])

    for _ in train_data[0]:
        print(word_index.get( _ - 3))

    x_train = vectorize(train_data)
    x_test = vectorize(train_labels)

    y_train = to_categorical(test_data)
    y_test = to_categorical(test_labels)

    model = models.Sequential()

    model.add(layers.Dense(64, activation='relu', input_shape= (10000,)))
    model.add(layers.Dense(64, activation='relu'))
    # Here we are using 64 neurons because we have 64 different outputs
    # to classify and for the case of multiple clasiffier softmax works better
    # than sigmoid because sigmoid can only classify between 0 and 1
    model.add(layers.Dense(46, activation='softmax'))

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    x_val = x_train[:1000]
    partial_x_train = x_train[1000:]

    y_val = y_train[:1000]
    partial_y_train = y_train[1000:]

    history = model.fit(partial_x_train, partial_y_train,
              epochs=9,
              batch_size=512,
              validation_data=(x_val, y_val))

    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']

    fig = plt.figure(figsize=(10,10))
    epoch = range(1,len(loss_values)+1)
    plt.plot(epoch,loss_values, 'o',label='training')
    plt.plot(epoch,val_loss_values, '--',label='val')
    plt.legend()
    plt.show()

    print(model.evaluate(x_test, y_test))

    predictions = model.predict(x_test)

    # This one returns all of the 64 posibles results
    print(np.sum(predictions[0]))

    # This one return the position of the array where the best prediction is
    best = np.argmax(predictions[0])
    print(best)

    print(predictions[0][best])
