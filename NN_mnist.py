from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random as ran
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import itertools
import os
import time
import warnings
from contextlib import contextmanager
from pandas.core.common import SettingWithCopyWarning
import sys

if not sys.warnoptions:
    warnings.simplefilter("ignore")

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

def plot_performance(train, test, title, save_plot, path):
    plt.figure(figsize=(15, 7))
    plt.title(title)
    plt.plot(train, label='train')
    plt.plot(test, label='test')
    plt.legend()
    if save_plot:
        plt.savefig(path + '/tmp/'+ title + '.png')
        plt.close
    else:
        plt.show()



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_my_confusion_matrix(y_true_labels,y_hat,title, save_plot, path):
            cf=confusion_matrix(y_true_labels, y_hat)
            class_names=np.array([0,1,2,3,4,5,6,7,8,9])
            plt.figure()
            plot_confusion_matrix(cf, class_names,title='Confusion matrix')
            plt.title(title)
            if save_plot:
                plt.savefig(path + '/tmp/' + title + '.png')
                plt.close
            else:
                plt.show()

def model(x_train, y_train, x_test, y_test,  model_file, path, batch_size = 128, num_epochs = 20, LR = 0.001,training_mode = True, save_model = True, save_plot =True):

    tf.reset_default_graph()

    # Comment below commands since data are already scaled
    # x_train /= 255
    # x_test /= 255

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

    # Metric
    y_p = tf.argmax(y, 1)

    N = x_train.shape[0]
    num_batches = int(N / batch_size)

    losses_train_batches, losses_test_batches = [], []
    accuracies_train_batches, accuracies_test_batches = [], []
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        best_test_acc = 0
        best_train_acc = 0
        if not training_mode:
            loadModel(sess, model_file)
            best_test_acc = sess.run(accuracy, feed_dict={x: x_test, y_: y_test})
            best_train_acc = sess.run(accuracy, feed_dict={x: x_train, y_: y_train})
            print('Best accuracy achieved on test set: ' + str(best_test_acc) + ' while on train set was: ' + str(
                best_train_acc))

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
                        l_test, acc_test = sess.run([loss, accuracy], feed_dict={x: x_test, y_: y_test})
                        losses_test_batches.append(l_test)
                        accuracies_test_batches.append(acc_test)

                        l_train, acc_train = sess.run([loss, accuracy], feed_dict={x: X, y_: Y})
                        losses_train_batches.append(l_train)
                        accuracies_train_batches.append(acc_train)

                        print("Epoch:", ep, "Batch:", i, "/", num_batches)
                        print('\n Loss and Accuracy of train set:', l_train, acc_train)
                        print('\n Loss and Accuracy of test set:', l_test, acc_test)

                print('End of '+str(ep+1) + ' /' + str(num_epochs))
                test_accuracy, y_hat = sess.run([accuracy, y_p], feed_dict={x: x_test, y_: y_test})
                train_accuracy = sess.run(accuracy, feed_dict={x: x_train, y_: y_train})
                if test_accuracy > best_test_acc:
                    best_test_acc = test_accuracy
                    best_train_acc = train_accuracy
                    # save the y_hat of the best trial to use it in confusion matrix
                    best_y_hat = y_hat
                    if save_model:
                        saveModel(sess, model_file)

            print('Best accuracy achieved on test set: '+str(best_test_acc)+ ' while on train set was: '+str(best_train_acc))

            title = 'Loss vs Batches'
            plot_performance(losses_train_batches, losses_test_batches, title, save_plot, path)

            title = 'Accuracy vs Batches'
            plot_performance(accuracies_train_batches, accuracies_test_batches, title, save_plot, path)

            # Plot confusion Matrix
            y_true_labels = np.argmax(y_test, 1)
            title='Confusion Matrix for Best Test Accuracy achieved'
            plot_my_confusion_matrix(y_true_labels, best_y_hat,title, save_plot, path)


def main(model_file, path, train=True, n_train=1000, n_test=200, save_model=True, save_plot = True):
    n_train = 55000 if n_train > 55000 else n_train
    n_test = 10000 if n_test > 10000 else n_test
    with timer("Load mnist dataset"):
        mnist = load_mnist()
    with timer("Load train & test dataset"):
        x_train, y_train = train_set(mnist, n_train)
        x_test, y_test = test_set(mnist, n_test)
    with timer("Display an image"):
        display_digit(ran.randint(0, x_train.shape[0]), x_train, y_train)
        display_mult_flat(0, 400, x_train)
    with timer("Run model"):
        model(x_train, y_train, x_test, y_test, model_file, path, batch_size=128, num_epochs=20, LR=0.001, training_mode=train, save_model = save_model, save_plot = save_plot)

if __name__ == "__main__":
    path = '/Users/Georgios.Drakos/Desktop'
    os.makedirs(path+'/tmp', exist_ok=True)
    model_file = path + '/tmp/model.ckpt'
    with timer("Full model run"):
        main(model_file, path=path, train=False, n_train=55000, n_test=10000, save_model=True, save_plot = True)