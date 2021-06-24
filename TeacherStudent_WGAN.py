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
import os,gzip
import keras as keras
from glob import glob


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
    return  cc

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

def My_Encoder_mnist(image,name, batch_size=64, reuse=False):
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

def My_Classifier_mnist(image,name, batch_size=64, reuse=False):
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

        return out_logit,softmaxValue

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
        self.mnist_train_y = np.zeros((np.shape(x_train)[0],4))
        self.mnist_train_y[:,0] = 1

        self.mnist_test_x = x_test
        self.mnist_test_y = y_test

        data_X, data_y = load_mnist(fashionMnistName)

        x_train1 = data_X[0:60000]
        x_test1 = data_X[60000:70000]
        y_train1 = data_y[0:60000]
        y_test1 = data_y[60000:70000]

        self.mnistFashion_train_x = x_train1
        self.mnistFashion_train_y = np.zeros((np.shape(x_train1)[0],4))
        self.mnistFashion_train_y[:,1] = 1

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
                    if thirdX[t1,p1,p2] == 1.0:
                        thirdX[t1,p1,p2] = 0
                    else:
                        thirdX[t1, p1, p2] = 1

        myTest = thirdX[0:self.batch_size]
        self.thirdX = np.reshape(thirdX,(-1,28,28,1))
        self.thirdY = np.zeros((np.shape(self.thirdX)[0],4))
        self.thirdY[:,2] = 1

        #ims("results/" + "gggg" + str(0) + ".jpg", merge(myTest[:64], [8, 8]))

        cc1 = 0

    def build_model(self):
        min_value = 1e-10
        # some parameters
        image_dims = [self.input_height, self.input_width, self.c_dim]
        bs = self.batch_size
        self.inputs = tf.placeholder(tf.float32, [bs] + image_dims, name='real_images')
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='z')
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.len_discrete_code])

        #GAN networks
        gan_code = tf.concat((self.z,self.y),axis=1)
        G1 = Generator_mnist("GAN_generator",gan_code, reuse=False)

        ## 1. GAN Loss
        # output of D for real images
        D_real, D_real_logits, _ = Discriminator_Mnist(self.inputs, "discriminator", reuse=False)

        # output of D for fake images
        D_fake, D_fake_logits, input4classifier_fake = Discriminator_Mnist(G1, "discriminator", reuse=True)

        self.g_loss = tf.reduce_mean(D_fake_logits)
        self.d_loss = tf.reduce_mean(D_real_logits) - tf.reduce_mean(D_fake_logits)

        epsilon = tf.random_uniform([], 0.0, 1.0)
        x_hat = epsilon * self.inputs + (1 - epsilon) * G1
        _,d_hat,_ = Discriminator_Mnist(x_hat, "discriminator", reuse=True)
        scale = 10.0
        ddx = tf.gradients(d_hat, x_hat)[0]
        ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=1))
        ddx = tf.reduce_mean(tf.square(ddx - 1.0) * scale)
        self.d_loss = self.d_loss + ddx

        # losses
        '''
        d_r_loss = tf.losses.mean_squared_error(tf.ones_like(D_real_logits), D_real_logits)
        d_f_loss = tf.losses.mean_squared_error(tf.zeros_like(D_fake_logits), D_fake_logits)
        self.d_loss = (d_r_loss + d_f_loss) / 2.0
        self.g_loss = tf.losses.mean_squared_error(tf.ones_like(D_fake_logits), D_fake_logits)
        '''
        """ Graph Input """
        # images

        self.isPhase = 0

        #domain 1
        z_mean, z_log_sigma_sq = My_Encoder_mnist(self.inputs,"encoder1", batch_size=64, reuse=False)
        out_logit,softmaxValue = My_Classifier_mnist(self.inputs,"classifier", batch_size=64, reuse=False)

        continous_variables = z_mean + z_log_sigma_sq * tf.random_normal(tf.shape(z_mean), 0, 1, dtype=tf.float32)

        log_y = tf.log(softmaxValue + 1e-10)
        discrete_real = my_gumbel_softmax_sample(log_y, np.arange(self.len_discrete_code))

        y_labels = tf.argmax(softmaxValue,1)
        y_labels = tf.cast(y_labels,dtype=tf.float32)
        y_labels = tf.reshape(y_labels,(-1,1))

        code1 = tf.concat((continous_variables,discrete_real),axis=1)
        reco1 = Generator_mnist("generator1",code1, reuse=False)
        reco2 = reco1

        #VAE loss
        reconstruction_loss1 = tf.reduce_mean(tf.reduce_sum(tf.square(reco1 - self.inputs), [1, 2, 3]))

        KL_divergence1 = 0.5 * tf.reduce_sum(
            tf.square(z_mean - y_labels) + tf.square(z_log_sigma_sq) - tf.log(1e-8 + tf.square(z_log_sigma_sq)) - 1, 1)
        KL_divergence1 = tf.reduce_mean(KL_divergence1)

        self.vae_loss1 = reconstruction_loss1 + KL_divergence1

        self.vaeLoss = self.vae_loss1

        #classification loss
        self.classifier_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=out_logit, labels=self.y))

        """ Training """
        # divide trainable variables into a group for D and a group for G
        T_vars = tf.trainable_variables()
        encoder_vars1 = [var for var in T_vars if var.name.startswith('encoder1')]
        encoderClassifier_vars1 = [var for var in T_vars if var.name.startswith('classifier')]
        generator1_vars = [var for var in T_vars if var.name.startswith('generator1')]
        discriminator_vars1 = [var for var in T_vars if var.name.startswith('discriminator')]
        GAN_generator_vars = [var for var in T_vars if var.name.startswith('GAN_generator')]

        self.output1 = reco1
        self.output2 = reco2
        self.GAN_output = G1

        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.vae1_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                .minimize(self.vaeLoss, var_list=encoder_vars1 + generator1_vars)
            self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                .minimize(self.d_loss, var_list=discriminator_vars1)
            self.g_optim = tf.train.AdamOptimizer(self.learning_rate * 5, beta1=self.beta1) \
                .minimize(self.g_loss, var_list=GAN_generator_vars)
            self.classifier_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                .minimize(self.classifier_loss, var_list=encoderClassifier_vars1)

        b1 = 0

    def predict(self):
        out_logit1, softmaxValue1 = My_Classifier_mnist(self.inputs, "classifier", batch_size=64,
                                                                         reuse=True)

        predictions = tf.argmax(softmaxValue1, 1, name="predictions")
        return predictions

    def test(self):
        with tf.Session() as sess:
            self.saver = tf.train.Saver()

            self.sess = sess
            sess.run(tf.global_variables_initializer())
            self.saver.restore(sess, 'models/TeacherStudent_WGAN_1')

            myIndex = 2
            mnist_x_test = self.mnistFashion_train_x[myIndex*self.batch_size: (myIndex+1)*self.batch_size]
            mnist_y_test = self.mnist_test_y

            testX = self.mnistFashion_train_x

            testY = self.mnistFashion_train_y

            mnistFashion_x_test = self.mnistFashion_train_x[0:self.batch_size]
            mnistFashion_y_test = self.mnistFashion_train_y

            r2 = np.reshape(mnist_x_test,(-1,28,28))
            ims("results/" + "Real" + str(0) + ".jpg", merge(r2[:64], [8, 8]))

            myTestX = np.concatenate((self.mnist_train_x,self.mnistFashion_train_x,self.thirdX),axis=0)
            index = [i for i in range(np.shape(myTestX)[0])]
            random.shuffle(index)
            myTestX = myTestX[index]
            myTestX = myTestX[0:self.batch_size]
            #myTestX = np.concatenate((self.mnist_train_x[0:32],self.mnistFashion_train_x[0:32]),axis=0)

            predictions = sess.run(self.output1, feed_dict={self.inputs: myTestX})
            predictions = np.reshape(predictions,(-1,28,28))
            ims("results/" + "myResults" + str(0) + ".png", merge(predictions[:64], [8, 8]))
            myTestX = np.reshape(myTestX,(-1,28,28))
            ims("results/" + "myReal" + str(0) + ".png", merge(myTestX[:64], [8, 8]))
 
            totalN = np.shape(testX)[0]
            myN = int(totalN/self.batch_size)
            myPrediction = self.predict()
            totalPredictions = []
            myCount = 0
            for i in range(myN):
                my1 = testX[self.batch_size*i:self.batch_size*(i+1)]
                predictions = sess.run(myPrediction,feed_dict={self.inputs:my1})
                for k in range(self.batch_size):
                    totalPredictions.append(predictions[k])
                    if predictions[k] == 1:
                        myCount = myCount+1

            totalPredictions = np.array(totalPredictions)
            print(totalPredictions)

            p = myCount
            b1 = totalPredictions

            r = sess.run(self.output1, feed_dict={self.inputs: mnist_x_test})
            r = np.reshape(r,(-1,28,28))
            ims("results/" + "my" + str(0) + ".jpg", merge(r[:64], [8, 8]))


    def Generate_GAN_Samples(self,n_samples,classN):
        myArr = []
        for tt in range(classN):
            y1 = np.zeros((self.batch_size,4))
            y1[:,0] = 1
            num1 = int(n_samples/self.batch_size)
            for i in range(num1):
                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

                g_outputs = self.sess.run(
                    self.GAN_output,
                    feed_dict={self.z: batch_z,self.y:y1})
                for t1 in range(self.batch_size):
                    myArr.append(g_outputs[t1])

        myArr = np.array(myArr)
        return myArr

    def train(self):

        isFirstStage = False
        with tf.Session() as sess:
            self.sess = sess
            sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()

            self.saver.restore(sess, 'models/TeacherStudent_WGAN')

            # saver to save model
            self.saver = tf.train.Saver()

            old_Nsamples = 30000
            oldX = self.Generate_GAN_Samples(old_Nsamples,2)
            oldY = np.zeros((np.shape(oldX)[0],4))
            oldY[:,0] = 1

            b1 = oldX[0:self.batch_size]
            b1 = np.reshape(b1,(-1,28,28))
            ims("results/" + "b1" + str(0) + ".jpg", merge(b1[:64], [8, 8]))

            testX = oldX

            totalN = np.shape(testX)[0]
            myN = int(totalN / self.batch_size)
            myPrediction = self.predict()
            totalPredictions = []
            myCount = 0
            for i in range(myN):
                my1 = testX[self.batch_size * i:self.batch_size * (i + 1)]
                predictions = sess.run(myPrediction, feed_dict={self.inputs: my1})
                for k in range(self.batch_size):
                    totalPredictions.append(predictions[k])

            totalPredictions = np.array(totalPredictions)
            totalPredictions = keras.utils.to_categorical(totalPredictions,4)
            oldY = totalPredictions

            #define combination of old and new datasets
            #second state
            dataX = np.concatenate((self.mnistFashion_train_x,oldX),axis=0)
            dataY = np.concatenate((self.mnistFashion_train_y,oldY),axis=0)

            #third stage
            dataX = np.concatenate((self.thirdX, oldX), axis=0)
            dataY = np.concatenate((self.thirdY, oldY), axis=0)

            #First stage
            if isFirstStage:
                dataX = self.mnist_train_x
                dataY = self.mnist_train_y

            counter = 0

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

                # get batch data
                for idx in range(start_batch_id, self.num_batches):
                    batch_images = dataX[idx*self.batch_size:(idx+1)*self.batch_size]
                    batch_y = dataY[idx*self.batch_size:(idx+1)*self.batch_size]

                    # update GAN
                    batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

                    # update D network
                    _, d_loss = self.sess.run([self.d_optim, self.d_loss],
                                                          feed_dict={self.inputs: batch_images,
                                                                     self.z: batch_z,self.y:batch_y})

                    # update G and Q network
                    _, g_loss = self.sess.run(
                        [self.g_optim, self.g_loss],
                        feed_dict={self.inputs: batch_images, self.z: batch_z,self.y:batch_y})

                    # update VAE
                    _,loss1 = self.sess.run([self.vae1_optim,self.vaeLoss],feed_dict={self.inputs:batch_images,self.y:batch_y})
                    class_loss = 0

                    #Update VAE by classification loss
                    _,c_class = self.sess.run([self.classifier_optim,self.classifier_loss],feed_dict={self.inputs:batch_images,self.y:batch_y})

                    outputs1,outputs2 = self.sess.run(
                            [self.output1,self.output2],
                            feed_dict={self.inputs: batch_images,self.y:batch_y})

                    g_outputs = self.sess.run(
                        self.GAN_output,
                        feed_dict={self.inputs: batch_images, self.z: batch_z,self.y:batch_y})

                    # display training status
                    counter += 1
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, vae_loss:%.8f. c_loss:%.8f" \
                              % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, g_loss,loss1,c_class))

                y_RPR1 = np.reshape(outputs1, (-1, 28, 28))
                g_output = np.reshape(g_outputs, (-1, 28, 28))
                ims("results/" + "MNIST" + str(epoch) + ".jpg", merge(y_RPR1[:64], [8, 8]))
                ims("results/" + "GAN" + str(epoch) + ".jpg", merge(g_output[:64], [8, 8]))

            self.saver.save(self.sess, "models/TeacherStudent_WGAN_1")

infoMultiGAN = LifeLone_MNIST()
infoMultiGAN.build_model()
#infoMultiGAN.train()
infoMultiGAN.test()


