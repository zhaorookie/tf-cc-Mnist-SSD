from __future__ import absolute_import, print_function
# import sys
# sys.path.append('/home/ubuntu/anaconda3/envs/tf2/lib/python3.6/site-packages')

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import imageio
import cv2 as cv


print(tf.__version__)

def export_mnist():
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1))
    test_images = test_images.reshape((10000, 28, 28, 1))

    train_images = train_images.astype(np.float32)
    test_images = test_images.astype(np.float32)

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=8)

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print("test loss: ", test_loss, "; test acc: ", test_acc)

    img = np.array(test_images[0]).reshape(28, 28)
    img.astype(np.float32)
    imageio.imwrite("img_0.png", img)

    result = model.predict(img.reshape(1, 28, 28, 1))

    print("before: ", result)
    result = np.argmax(result)
    print("after: ", result)

    tf.saved_model.save(model, 'tf_mnist/1/')


def load_mnist():
    # img = imageio.imread("img_0.png")
    img = cv.imread("img_0.png", cv.IMREAD_GRAYSCALE)
    img = np.asarray(img)
    img = img.reshape(1, 28, 28, 1)
    img_tensor = tf.convert_to_tensor(img, tf.float32)

    loaded = tf.saved_model.load('tf_mnist/1/')
    # 下面这里可用 saved_model_cli show --dir tf_mnist/1 --all
    infer = loaded.signatures['serving_default']
    result = infer(img_tensor)['dense_1']

    print("before: ", result)
    result = np.argmax(result)
    print("after: ", result)


if __name__ == "__main__":
    # export_mnist()
    load_mnist()