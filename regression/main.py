import pandas as pd
import numpy as np
from keras.datasets import boston_housing
from keras import layers, models, optimizers

import matplotlib.pyplot as plt


def build_model_regression(lr_var, input_data):
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(input_data,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))

    model.compile(optimizer=optimizers.RMSprop(lr=lr_var), loss='mse', metrics=['mae'])

    return model


if __name__ == '__main__':
    (train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()
    print(train_data.shape)
    print(train_labels.shape)

    # We have to normalize the data because there are some values that are to
    # high
    mean = train_data.mean(axis=0)
    train_data = train_data-mean
    std = train_data.std(axis=0)
    train_data = train_data/std

    test_data = test_data-mean
    test_data = test_data/std

    k = 4
    num_val_samples = len(train_data) // k
    num_epochs = 80
    all_history = []

    for i in range(k):
        print('Fold:', i)
        val_data = train_data[i*num_val_samples: (i+1) * num_val_samples]
        val_target = train_labels[i*num_val_samples: (i+1) * num_val_samples]

        partial_train_data = np.concatenate(
            [train_data[:i * num_val_samples],
             train_data[(i+1) * num_val_samples:]],
            axis=0)

        partial_train_targets = np.concatenate(
            [train_labels[:i * num_val_samples],
             train_labels[(i+1) * num_val_samples:]],
            axis=0)

        model = build_model_regression(0.001, 13)

        history = model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=16, validation_data=(val_data, val_target), verbose=0)

        all_history.append(history.history['val_mae'])

    print('-'*32)
    print(len(all_history[0]))

    print('-'*32)


    all_mae_avg = pd.DataFrame(all_history).mean(axis=0)
    print(all_mae_avg)

    plt.plot(range(1,len(all_mae_avg)+1), all_mae_avg)
    plt.show()

    print(model.evaluate(test_data, test_labels))
