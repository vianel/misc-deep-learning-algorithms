from keras import layers, models
from tensorflow.keras.utils import to_categorical
from keras.datasets import mnist
import matplotlib.pyplot as plt


if __name__ == '__main__':
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()
    plt.imshow(train_data[4])
    plt.show()
    print(train_labels[4])

    model = models.Sequential()
    model.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))

    # We decide to use 10 neural because we are guessing number from 0 to 10
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                  metrics='accuracy')

    print(model.summary())

    # Is easier for the neural networks work with 2 dimensions thats why we
    # reshape
    x_train = train_data.reshape((60000, 28*28))
    x_train = train_data.reshape((60000, 28*28))

    # Is easier to use float insteand of int in a neural network als owe divide
    # by 255 because thats the max number for each pixel
    x_train = x_train.astype('float32')/255

    x_test = test_data.reshape((10000, 28*28))
    x_test = x_test.astype('float32')/255

    # Is easier to the neural network use a vector instead of a number so we
    # need to use to categorical
    y_train = to_categorical(train_labels)
    y_test = to_categorical(test_labels)

    print(train_labels[0])
    print(y_train[0])

    # Training the network
    model.fit(x_train, y_train, epochs=5, batch_size=128)

    # Evaluate the model
    print(model.evaluate(x_test, y_test))
