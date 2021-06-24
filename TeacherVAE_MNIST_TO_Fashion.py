from Multiple_GAN_codes.Basic_structure import *
from keras.datasets import mnist
import time
from utils import *
from scipy.misc import imsave as ims
from ops import *
from utils import *
from Utlis2 import *
import random as random
from glob import glob
import os, gzip
import keras as keras
from glob import glob
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

def file_name(file_dir):
    t1 = []
    file_dir = "F:/Third_Experiment/Multiple_GAN_codes/data/images_background/"
    for root, dirs, files in os.walk(file_dir):
        for a1 in dirs:
            b1 = "F:/Third_Experiment/Multiple_GAN_codes/data/images_background/" + a1 + "/renders/*.png"
            b1 = "F:/Third_Experiment/Multiple_GAN_codes/data/images_background/" + a1
            for root2, dirs2, files2 in os.walk(b1):
                for c1 in dirs2:
                    b2 = b1 + "/" + c1 + "/*.png"
                    img_path = glob(b2)
                    t1.append(img_path)

        print('root_dir:', root)  # 当前目录路径
        print('sub_dirs:', dirs)  # 当前路径下所有子目录
        print('files:', files)  # 当前路径下所有非目录子文件

    cc = []

    for i in range(len(t1)):
        a1 = t1[i]
        for p1 in a1:
            cc.append(p1)
    return cc


def sample_gumbel(shape, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = tf.random_uniform(shape, minval=0, maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)


def my_gumbel_softmax_sample(logits, cats_range, temperature=0.1):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(tf.shape(logits))
    logits_with_noise = tf.nn.softmax(y / temperature)
    return logits_with_noise


def load_mnist(dataset_name):
    data_dir = os.path.join("./data", dataset_name)

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    data = extract_data(data_dir + '/train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    trX = data.reshape((60000, 28, 28, 1))

    data = extract_data(data_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
    trY = data.reshape((60000))

    data = extract_data(data_dir + '/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    teX = data.reshape((10000, 28, 28, 1))

    data = extract_data(data_dir + '/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    teY = data.reshape((10000))

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1.0

    return X / 255., y_vec


def My_Encoder_mnist(image, name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        len_discrete_code = 4

        is_training = True
        z_dim = 32
        x = image
        net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='c_conv1'))
        net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='c_conv2'), is_training=is_training, scope='c_bn2'))
        net = tf.reshape(net, [batch_size, -1])
        net = lrelu(bn(linear(net, 1024, scope='c_fc3'), is_training=is_training, scope='c_bn3'))

        net = lrelu(bn(linear(net, 64, scope='e_fc11'), is_training=is_training, scope='c_bn11'))

        z_mean = linear(net, z_dim, 'e_mean')
        z_log_sigma_sq = linear(net, z_dim, 'e_log_sigma_sq')
        z_log_sigma_sq = tf.nn.softplus(z_log_sigma_sq)

        return z_mean, z_log_sigma_sq


def My_Classifier_mnist(image, name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        len_discrete_code = 4

        is_training = True
        z_dim = 32
        x = image
        net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='c_conv1'))
        net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='c_conv2'), is_training=is_training, scope='c_bn2'))
        net = tf.reshape(net, [batch_size, -1])
        net = lrelu(bn(linear(net, 1024, scope='c_fc3'), is_training=is_training, scope='c_bn3'))

        net = lrelu(bn(linear(net, 64, scope='e_fc11'), is_training=is_training, scope='c_bn11'))

        out_logit = linear(net, len_discrete_code, scope='e_fc22')
        softmaxValue = tf.nn.softmax(out_logit)

        return out_logit, softmaxValue

def MINI_Classifier(s, scopename,reuse=False):
    keep_prob = 1.0
    with tf.variable_scope(scopename, reuse=reuse):
        input = s
        n_output = 10
        n_hidden = 500
        # initializers
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)

        # 1st hidden layer
        w0 = tf.get_variable('w0', [input.get_shape()[1], n_hidden], initializer=w_init)
        b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
        h0 = tf.matmul(s, w0) + b0
        h0 = tf.nn.tanh(h0)
        h0 = tf.nn.dropout(h0, keep_prob)

        n_output = 10
        # output layer-mean
        wo = tf.get_variable('wo', [h0.get_shape()[1], n_output], initializer=w_init)
        bo = tf.get_variable('bo', [n_output], initializer=b_init)
        y1 = tf.matmul(h0, wo) + bo
        y = tf.nn.softmax(y1)

    return y1,y


# Create model of CNN with slim api
def Image_classifier(inputs,scopename, is_training=True,reuse=False):
    with tf.variable_scope(scopename, reuse=reuse):
        batch_norm_params = {'is_training': is_training, 'decay': 0.9, 'updates_collections': None}
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):
            x = tf.reshape(inputs, [-1, 28, 28, 1])

            # For slim.conv2d, default argument values are like
            # normalizer_fn = None, normalizer_params = None, <== slim.arg_scope changes these arguments
            # padding='SAME', activation_fn=nn.relu,
            # weights_initializer = initializers.xavier_initializer(),
            # biases_initializer = init_ops.zeros_initializer,
            net = slim.conv2d(x, 32, [5, 5], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.conv2d(net, 64, [5, 5], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.flatten(net, scope='flatten3')

            # For slim.fully_connected, default argument values are like
            # activation_fn = nn.relu,
            # normalizer_fn = None, normalizer_params = None, <== slim.arg_scope changes these arguments
            # weights_initializer = initializers.xavier_initializer(),
            # biases_initializer = init_ops.zeros_initializer,
            net = slim.fully_connected(net, 1024, scope='fc3')
            net = slim.dropout(net, is_training=is_training, scope='dropout3')  # 0.5 by default
            outputs = slim.fully_connected(net, 10, activation_fn=None, normalizer_fn=None, scope='fco')
    return outputs

class LifeLone_MNIST(object):
    def __init__(self):
        self.batch_size = 64
        self.input_height = 28
        self.input_width = 28
        self.c_dim = 1
        self.z_dim = 32
        self.len_discrete_code = 4
        self.epoch = 10

        self.learning_rate = 0.0002
        self.beta1 = 0.5

        # MNIST dataset
        mnistName = "mnist"
        fashionMnistName = "Fashion"

        data_X, data_y = load_mnist(mnistName)
        x_train = data_X[0:60000]
        x_test = data_X[60000:70000]
        y_train = data_y[0:60000]
        y_test = data_y[60000:70000]

        self.mnist_train_x = x_train
        self.mnist_train_y = np.zeros((np.shape(x_train)[0], 4))
        self.mnist_train_y[:, 0] = 1
        self.mnist_label = y_train
        self.mnist_label_test = y_test

        self.mnist_test_x = x_test
        self.mnist_test_y = y_test

        data_X, data_y = load_mnist(fashionMnistName)

        x_train1 = data_X[0:60000]
        x_test1 = data_X[60000:70000]
        y_train1 = data_y[0:60000]
        y_test1 = data_y[60000:70000]

        self.mnistFashion_train_x = x_train1
        self.mnistFashion_train_y = np.zeros((np.shape(x_train1)[0], 4))
        self.mnistFashion_train_y[:, 1] = 1
        self.mnistFashion_label = y_train1

        self.mnistFashion_test_x = x_test1
        self.mnistFashion_test_labels = y_test1

        '''
        files = file_name(1)
        data_files = files
        data_files = sorted(data_files)
        data_files = np.array(data_files)  # for tl.iterate.minibatches
        n_examples = np.shape(data_files)[0]
        batch = [get_image(batch_file, 105, 105,
                           resize_height=28, resize_width=28,
                           crop=False, grayscale=True) \
                 for batch_file in data_files]

        thirdX = np.array(batch)

        for t1 in range(n_examples):
            a1 = thirdX[t1]
            for p1 in range(28):
                for p2 in range(28):
                    if thirdX[t1, p1, p2] == 1.0:
                        thirdX[t1, p1, p2] = 0
                    else:
                        thirdX[t1, p1, p2] = 1

        myTest = thirdX[0:self.batch_size]
        self.thirdX = np.reshape(thirdX, (-1, 28, 28, 1))
        self.thirdY = np.zeros((np.shape(self.thirdX)[0], 4))
        self.thirdY[:, 2] = 1

        # ims("results/" + "gggg" + str(0) + ".jpg", merge(myTest[:64], [8, 8]))
        '''
        cc1 = 0

    def build_model(self):
        min_value = 1e-10
        # some parameters
        image_dims = [self.input_height, self.input_width, self.c_dim]
        bs = self.batch_size
        self.inputs = tf.placeholder(tf.float32, [bs] + image_dims, name='real_images')
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='z')
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.len_discrete_code])
        self.labels = tf.placeholder(tf.float32, [self.batch_size, 10])

        # GAN networks

        self.isPhase = 0

        # domain 1
        z_mean, z_log_sigma_sq = My_Encoder_mnist(self.inputs, "encoder1", batch_size=64, reuse=False)
        continous_variables = z_mean + z_log_sigma_sq * tf.random_normal(tf.shape(z_mean), 0, 1, dtype=tf.float32)

        #define the classifier
        label_logits = Image_classifier(self.inputs, "Mini_classifier", reuse=False)
        self.mini_classLoss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=label_logits, labels=self.labels))

        reco1 = Generator_mnist("generator1", continous_variables, reuse=False)
        reco2 = reco1

        z_mean2, z_log_sigma_sq2 = My_Encoder_mnist(self.inputs, "encoder2", batch_size=64, reuse=False)
        continous_variables2 = z_mean2 + z_log_sigma_sq2 * tf.random_normal(tf.shape(z_mean2), 0, 1, dtype=tf.float32)

        # define the classifier
        label_logits2 = Image_classifier(self.inputs, "Mini_classifier2", reuse=False)
        self.mini_classLoss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=label_logits, labels=self.labels))

        reco2 = Generator_mnist("generator2", continous_variables, reuse=False)

        # VAE loss
        reconstruction_loss1 = tf.reduce_mean(tf.reduce_sum(tf.square(reco1 - self.inputs), [1, 2, 3]))
        reconstruction_loss2 = tf.reduce_mean(tf.reduce_sum(tf.square(reco2 - self.inputs), [1, 2, 3]))

        KL_divergence1 = 0.5 * tf.reduce_sum(
            tf.square(z_mean) + tf.square(z_log_sigma_sq) - tf.log(1e-8 + tf.square(z_log_sigma_sq)) - 1, 1)
        KL_divergence1 = tf.reduce_mean(KL_divergence1)

        KL_divergence2 = 0.5 * tf.reduce_sum(
            tf.square(z_mean2) + tf.square(z_log_sigma_sq2) - tf.log(1e-8 + tf.square(z_log_sigma_sq2)) - 1, 1)
        KL_divergence2 = tf.reduce_mean(KL_divergence2)

        self.vae_loss1 = reconstruction_loss1 + KL_divergence1
        self.vae_loss2 = reconstruction_loss2 + KL_divergence2

        self.vaeLoss = self.vae_loss1

        """ Training """
        # divide trainable variables into a group for D and a group for G
        T_vars = tf.trainable_variables()
        encoder_vars1 = [var for var in T_vars if var.name.startswith('encoder1')]
        encoder_vars2 = [var for var in T_vars if var.name.startswith('encoder2')]

        generator1_vars = [var for var in T_vars if var.name.startswith('generator1')]
        generator2_vars = [var for var in T_vars if var.name.startswith('generator2')]

        MiniClassifier_vars = [var for var in T_vars if var.name.startswith('Mini_classifier')]

        self.output1 = reco1
        self.output2 = reco2

        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.vae1_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                .minimize(self.vae_loss1, var_list=encoder_vars1 + generator1_vars)
            self.vae1_optim2 = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                .minimize(self.vae_loss2, var_list=encoder_vars2 + generator2_vars)
            self.mini_classifier_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                .minimize(self.mini_classLoss, var_list=MiniClassifier_vars)
        b1 = 0

    def predict(self):
        # define the classifier
        label_logits = Image_classifier(self.inputs, "Mini_classifier", reuse=True)
        label_softmax = tf.nn.softmax(label_logits)
        predictions = tf.argmax(label_softmax, 1, name="predictions")
        return predictions

    def Give_predictedLabels(self,testX):
        totalN = np.shape(testX)[0]
        myN = int(totalN / self.batch_size)
        myPrediction = self.predict()
        totalPredictions = []
        myCount = 0
        for i in range(myN):
            my1 = testX[self.batch_size * i:self.batch_size * (i + 1)]
            predictions = self.sess.run(myPrediction, feed_dict={self.inputs: my1})
            for k in range(self.batch_size):
                totalPredictions.append(predictions[k])

        totalPredictions = np.array(totalPredictions)
        totalPredictions = keras.utils.to_categorical(totalPredictions)
        return totalPredictions

    def Calculate_accuracy(self,testX,testY):
        #testX = self.mnist_test_x
        totalN = np.shape(testX)[0]
        myN = int(totalN / self.batch_size)
        myPrediction = self.predict()
        totalPredictions = []
        myCount = 0
        for i in range(myN):
            my1 = testX[self.batch_size * i:self.batch_size * (i + 1)]
            predictions = self.sess.run(myPrediction, feed_dict={self.inputs: my1})
            for k in range(self.batch_size):
                totalPredictions.append(predictions[k])

        totalPredictions = np.array(totalPredictions)

        testLabels = testY[0:np.shape(totalPredictions)[0]]
        testLabels = np.argmax(testLabels, 1)
        trueCount = 0
        for k in range(np.shape(testLabels)[0]):
            if testLabels[k] == totalPredictions[k]:
                trueCount = trueCount + 1

        accuracy = (float)(trueCount / np.shape(testLabels)[0])

        return accuracy

    def test(self):
        with tf.Session() as sess:
            self.saver = tf.train.Saver()

            self.sess = sess
            sess.run(tf.global_variables_initializer())
            self.saver.restore(sess, 'models/TeacherStudent_MNIST_TO_Fashion')

            myPredict = self.predict()

            myIndex = 2
            mnist_x_test = self.mnistFashion_train_x[myIndex * self.batch_size: (myIndex + 1) * self.batch_size]
            mnist_y_test = self.mnist_test_y

            testX = self.mnistFashion_train_x
            testY = self.mnistFashion_train_y

            mnistFashion_x_test = self.mnistFashion_train_x[0:self.batch_size]
            mnistFashion_y_test = self.mnistFashion_train_y

            r2 = np.reshape(mnist_x_test, (-1, 28, 28))
            ims("results/" + "Real" + str(0) + ".jpg", merge(r2[:64], [8, 8]))

            my11 = self.sess.run(self.output1, feed_dict={self.inputs: self.mnistFashion_test_x[0:self.batch_size]})
            my11 = np.reshape(my11,(-1,28,28))
            ims("results/" + "myRe" + str(0) + ".jpg", merge(my11[:64], [8, 8]))

            mnistAccuracy = self.Calculate_accuracy(self.mnist_test_x, self.mnist_label_test)
            mnistFashionAccuracy = self.Calculate_accuracy(self.mnistFashion_test_x, self.mnistFashion_test_labels)

            bc = 0

    def Generate_GAN_Samples(self, n_samples, classN):

        reco1 = Generator_mnist("generator1", self.z, reuse=True)

        myArr = []
        for tt in range(classN):
            y1 = np.zeros((self.batch_size, 4))
            y1[:, 0] = 1
            num1 = int(n_samples / self.batch_size)
            for i in range(num1):
                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

                g_outputs = self.sess.run(
                    reco1,
                    feed_dict={self.z: batch_z})
                for t1 in range(self.batch_size):
                    myArr.append(g_outputs[t1])

        myArr = np.array(myArr)
        return myArr

    def train_classifier(self):
        isFirstStage = True
        with tf.Session() as sess:
            self.sess = sess
            sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()

            #self.saver.restore(sess, 'models/TeacherVAE_MNIST_TO_Fashion')

            # saver to save model
            self.saver = tf.train.Saver()

            old_Nsamples = 60000
            oldX = self.Generate_GAN_Samples(old_Nsamples, 1)
            oldY = np.zeros((np.shape(oldX)[0], 4))
            oldY[:, 0] = 1

            oldLabels = self.Give_predictedLabels(oldX)
            oldX = oldX[0:np.shape(oldLabels)[0]]
            oldY = oldY[0:np.shape(oldLabels)[0]]

            # define combination of old and new datasets
            # second state
            dataX = np.concatenate((self.mnistFashion_train_x, oldX), axis=0)
            dataY = np.concatenate((self.mnistFashion_train_y, oldY), axis=0)
            dataLabels = np.concatenate((self.mnistFashion_label, oldLabels),axis=0)
            labelsY = dataLabels

            # third stage
            # dataX = np.concatenate((self.thirdX, oldX), axis=0)
            # dataY = np.concatenate((self.thirdY, oldY), axis=0)

            # First stage
            if isFirstStage:
                dataX = self.mnist_train_x
                dataY = self.mnist_train_y
                labelsY = self.mnist_label
                '''
                dataX = self.mnistFashion_train_x
                dataY = self.mnistFashion_train_y
                labelsY = self.mnistFashion_label
                '''

            x_train = dataX
            y_train = labelsY

            x_test = np.concatenate((self.mnistFashion_test_x,self.mnist_test_x),axis=0)
            y_test = np.concatenate((self.mnistFashion_test_labels,self.mnistFashion_test_labels),axis=0)

            '''
            input_shape = (28, 28, 1)
            model = Sequential()
            model.add(Conv2D(32, kernel_size=(3, 3),
                             activation='relu',
                             input_shape=input_shape))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))
            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(10, activation='softmax'))

            model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=keras.optimizers.Adadelta(),
                          metrics=['accuracy'])

            epochs = 20
            batch_size = 64
            model.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=1,
                      validation_data=(x_test, y_test))

            score = model.evaluate(x_test, y_test, verbose=0)
            counter = 0
            '''

            n_examples = np.shape(dataX)[0]

            start_epoch = 0
            start_batch_id = 0
            self.num_batches = int(n_examples / self.batch_size)

            # loop for epoch
            start_time = time.time()
            for epoch in range(start_epoch, self.epoch):
                count = 0
                # Random shuffling
                index = [i for i in range(n_examples)]
                random.shuffle(index)
                dataX = dataX[index]
                dataY = dataY[index]
                labelsY = labelsY[index]

                # get batch data
                for idx in range(start_batch_id, self.num_batches):
                    batch_images = dataX[idx * self.batch_size:(idx + 1) * self.batch_size]
                    batch_y = dataY[idx * self.batch_size:(idx + 1) * self.batch_size]
                    batch_labels = labelsY[idx * self.batch_size:(idx + 1) * self.batch_size]

                    # Train a classifier
                    _, c_class = self.sess.run([self.mini_classifier_optim, self.mini_classLoss],
                                               feed_dict={self.inputs: batch_images, self.y: batch_y,
                                                          self.labels: batch_labels})
                    print(c_class)

                mnistAccuracy = self.Calculate_accuracy(self.mnist_test_x, self.mnist_label_test)
                mnistFashionAccuracy = self.Calculate_accuracy(self.mnistFashion_test_x, self.mnistFashion_label)

            self.saver.save(self.sess, "models/TeacherStudent_MNIST_TO_SVHN")

    def train(self):

        isFirstStage = False
        with tf.Session() as sess:
            self.sess = sess
            sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()

            self.saver.restore(sess, 'models/TeacherVAE_MNIST_TO_Fashion')

            # saver to save model
            self.saver = tf.train.Saver()

            old_Nsamples = 60000
            oldX = self.Generate_GAN_Samples(old_Nsamples, 1)
            oldY = np.zeros((np.shape(oldX)[0], 4))
            oldY[:, 0] = 1

            oldLabels = self.Give_predictedLabels(oldX)
            oldX = oldX[0:np.shape(oldLabels)[0]]
            oldY = oldY[0:np.shape(oldLabels)[0]]

            # define combination of old and new datasets
            # second state
            dataX = np.concatenate((self.mnistFashion_train_x, oldX), axis=0)
            dataY = np.concatenate((self.mnistFashion_train_y, oldY), axis=0)
            dataLabels = np.concatenate((self.mnistFashion_label,oldLabels))
            labelsY = dataLabels

            # third stage
            #dataX = np.concatenate((self.thirdX, oldX), axis=0)
            #dataY = np.concatenate((self.thirdY, oldY), axis=0)

            # First stage
            if isFirstStage:
                dataX = self.mnist_train_x
                dataY = self.mnist_train_y
                labelsY = self.mnist_label

            counter = 0

            n_examples = np.shape(dataX)[0]

            start_epoch = 0
            start_batch_id = 0
            self.num_batches = int(n_examples / self.batch_size)

            mnistAccuracy_list = []
            mnistFashionAccuracy_list = []

            # loop for epoch
            start_time = time.time()
            for epoch in range(start_epoch, self.epoch):
                count = 0
                # Random shuffling
                index = [i for i in range(n_examples)]
                random.shuffle(index)
                dataX = dataX[index]
                dataY = dataY[index]
                labelsY = labelsY[index]

                # get batch data
                for idx in range(start_batch_id, self.num_batches):
                    batch_images = dataX[idx * self.batch_size:(idx + 1) * self.batch_size]
                    batch_y = dataY[idx * self.batch_size:(idx + 1) * self.batch_size]
                    batch_labels = labelsY[idx * self.batch_size:(idx + 1) * self.batch_size]

                    # update GAN
                    batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

                    # update VAE
                    _, loss1 = self.sess.run([self.vae1_optim, self.vaeLoss],
                                             feed_dict={self.inputs: batch_images, self.y: batch_y})

                    _, loss2 = self.sess.run([self.vae1_optim2, self.vae_loss2],
                                             feed_dict={self.inputs: batch_images, self.y: batch_y})


                    # Train a classifier
                    _ ,c_class= self.sess.run([self.mini_classifier_optim, self.mini_classLoss],
                                               feed_dict={self.inputs: batch_images, self.y: batch_y,self.labels:batch_labels})


                    outputs1, outputs2 = self.sess.run(
                        [self.output1, self.output2],
                        feed_dict={self.inputs: batch_images, self.y: batch_y})

                    # display training status
                    counter += 1
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, vae_loss:%.8f. c_loss:%.8f" \
                          % (epoch, idx, self.num_batches, time.time() - start_time, 0, 0, loss1, c_class))

                y_RPR1 = np.reshape(outputs1, (-1, 28, 28,1))
                ims("results/" + "MNIST" + str(epoch) + ".jpg", merge(y_RPR1[:64], [8, 8]))

                if isFirstStage==False:
                    mnistAccuracy = self.Calculate_accuracy(self.mnist_test_x, self.mnist_label_test)
                    mnistFashionAccuracy = self.Calculate_accuracy(self.mnistFashion_test_x, self.mnistFashion_test_labels)
                    mnistAccuracy_list.append(mnistAccuracy)
                    mnistFashionAccuracy_list.append(mnistFashionAccuracy)
                else:
                    mnistAccuracy = self.Calculate_accuracy(self.mnist_test_x, self.mnist_label_test)
                    mnistAccuracy_list.append(mnistAccuracy)

            lossArr1 = np.array(mnistAccuracy_list).astype('str')
            f = open("results/TeacherVAE_MnistToFashion_mnistAccuracy.txt", "w", encoding="utf-8")
            for i in range(np.shape(lossArr1)[0]):
                f.writelines(lossArr1[i])
                f.writelines('\n')
            f.flush()
            f.close()

            lossArr1 = np.array(mnistFashionAccuracy_list).astype('str')
            f = open("results/TeacherVAE_MnistToFashion_fashionAccuracy.txt", "w", encoding="utf-8")
            for i in range(np.shape(lossArr1)[0]):
                f.writelines(lossArr1[i])
                f.writelines('\n')
            f.flush()
            f.close()

            self.saver.save(self.sess, "models/TeacherVAE_MNIST_TO_Fashion2")


infoMultiGAN = LifeLone_MNIST()
infoMultiGAN.build_model()
infoMultiGAN.train()
#infoMultiGAN.train_classifier()
#infoMultiGAN.test()
