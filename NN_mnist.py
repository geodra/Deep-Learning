from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random as ran
from sklearn.utils import shuffle
import os
import time
import warnings
from contextlib import contextmanager
from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# Tensorflow saving and loading of models
def saveModel(sess,model_file):
    print("Saving model at: " + model_file)
    saver = tf.train.Saver()
    saver.save(sess, model_file)
    print("Saved")


def loadModel(sess, model_file):
    print("Loading model from: " + model_file)
    # load saved session
    loader = tf.train.Saver()
    loader.restore(sess, model_file)
    print("Loaded")

def load_mnist():
    return input_data.read_data_sets('MNIST_data', one_hot=True)

def train_set(mnist, num):
    print ('Total training images in dataset = ' + str(mnist.train.images.shape))
    x_train = mnist.train.images[:num,:]
    y_train = mnist.train.labels[:num,:]
    print ('Train examples loaded = ' + str(len(y_train)))
    return x_train, y_train

def test_set(mnist, num):
    print ('Total test images in dataset = ' + str(mnist.test.images.shape))
    x_test = mnist.test.images[:num,:]
    y_test = mnist.test.labels[:num,:]
    print ('Test examples Loaded = ' + str(len(y_test)))
    return x_test, y_test

def display_digit(num, x_train, y_train):
    label = y_train[num].argmax(axis=0)
    image = x_train[num].reshape([28,28])
    plt.title('Example: %d  Label: %d' % (num, label))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()

def display_mult_flat(start, stop, x_train):
    images = x_train[start].reshape([1,784])
    for i in range(start+1,stop):
        images = np.concatenate((images, x_train[i].reshape([1,784])))
    plt.imshow(images, cmap=plt.get_cmap('gray_r'))
    plt.show()

def model(x_train, y_train, x_test, y_test,  model_file, batch_size = 128, num_epochs = 20, LR = 0.001,training_mode = True, save=True):
    tf.reset_default_graph()

    ## Inputs ##
    x = tf.placeholder(tf.float32, shape=[None, 784])
    # None means that can be fed any multitude of 784-sized values.
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    ## Variables ##
    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))

    # Outputs
    y = tf.nn.softmax(tf.matmul(x,W) + b)

    # Loss & optimization
    loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    learning_rate = tf.placeholder("float32")
    opt = tf.train.AdamOptimizer(learning_rate = learning_rate) #tune
    train_op = opt.minimize(loss)

    N = x_train.shape[0]
    num_batches = int(N / batch_size)

    best_acc = 0
    losses_train, losses_test = [], []
    accuracies_train, accuracies_test = [], []
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        if not training_mode:
            loadModel(sess, model_file)

        else:
            for ep in range(num_epochs):
                print('EPOCH:', ep)
                # reshuffle training data every epoch
                X, Y = shuffle(x_train, y_train)
                for i in range(num_batches):
                    X_batch = X[i * batch_size:(i * batch_size + batch_size)]
                    Y_batch = Y[i * batch_size:(i * batch_size + batch_size)]

                    sess.run(train_op, feed_dict={x: X_batch, y_: Y_batch, learning_rate: LR})

                    # Check performance every some steps
                    if i % int(num_batches/4) == 0:
                        l_test = sess.run(loss, feed_dict={x: x_test, y_: y_test})
                        losses_test.append(l_test)

                        l_train = sess.run(loss, feed_dict={x: X, y_: Y})
                        losses_train.append(l_train)


                        acc_test = sess.run(accuracy, feed_dict={x: x_test, y_: y_test})
                        accuracies_test.append(acc_test)

                        acc_train = sess.run(accuracy, feed_dict={x: X, y_: Y})
                        accuracies_train.append(acc_train)

                        print("Epoch:", ep, "Batch:", i, "/", num_batches)
                        print('\n Loss and Accuracy of train set:', l_train, acc_train)
                        print('\n Loss and Accuracy of test set:', l_test, acc_test)
            if save:
                saveModel(sess, model_file)
            plt.figure(figsize=(15, 7))
            plt.title('Loss')
            plt.plot(losses_train, label='train')
            plt.plot(losses_test, label='test')
            plt.legend()
            plt.show()

            plt.figure(figsize=(15, 7))
            plt.title('Accuracy')
            plt.plot(accuracies_train, label = 'train')
            plt.plot(accuracies_test, label = 'test')
            plt.legend()
            plt.show()

        acc_train = sess.run(accuracy, feed_dict={x: x_train, y_: y_train})
        acc_test = sess.run(accuracy, feed_dict={x: x_test, y_: y_test})
        print('Accuracy of train is:' + str(acc_train) + ' and test set is:' + str(acc_test))

def main(model_file, train=True, n_train=1000, n_test=200, save=True):
    with timer("Load mnist dataset"):
        mnist = load_mnist()
    with timer("Load train & test dataset"):
        x_train, y_train = train_set(mnist, n_train)
        x_test, y_test = test_set(mnist, n_test)
    with timer("Display an image"):
        display_digit(ran.randint(0, x_train.shape[0]), x_train, y_train)
        display_mult_flat(0, 400, x_train)
    with timer("Run model"):
        model(x_train, y_train, x_test, y_test, model_file, batch_size=128, num_epochs=20, LR=0.001, training_mode=train, save=True)

if __name__ == "__main__":
    path = '/Users/Georgios.Drakos/Desktop'
    os.makedirs(path+'/tmp', exist_ok=True)
    model_file = path + '/tmp/model.ckpt'
    with timer("Full model run"):
        main(model_file, train=False, n_train=55000, n_test=10000, save=True)