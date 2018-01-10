#!/usr/bin/python
"""
This code implements the LeNet5 on MNIST by Theano.
"""
import cPickle
import gzip
import os
import sys
import time
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv
import keras
from keras.datasets import mnist

class LeNetConvPoolLayer(object):
    """
    The combination of a conv + pooling layer.
    rng: random generator used to initialize W.
    input: 4 dimensional tensor, theano.tensor.dtensor4.
    filter_shape: (number of filters, num_input_feature_maps, filter height, filter width)
    image_shape: (batch_size, num_input_feature_mapes, image_height, image_width)
    poolsize: (#rows, #cols)
    """
    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        #image_shape[1] and filter_shape[1] are both num_input_feature_maps, they should be the same
        assert image_shape[1] == filter_shape[1]
        self.input = input
        #Every hidden neuron (pixel) is connected to the following number of previous layers neurons:
        #num_input_feature_maps * filter_height * filter_width
        fan_in = np.prod(filter_shape[1:])
        #The gradient of every neuron of the lower layer comes from (num_output_feature_maps * filter_height * filter_width)/pooling_size
        fan_out = (filter_shape[0]*np.prod(filter_shape[2:]))/np.prod(poolsize)
        #Initialize the weights W, need to be shared, so that it can be trained.
        W_bound = np.sqrt(6./(fan_in+fan_out))
        self.W = theano.shared(np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape), dtype=theano.config.floatX), borrow=True)
        #The bias is a 1D tensor, one bias per output feature map. The number of outputed feature maps is dertermined
        #by the number of filters, so we use filter_shape[0], i.e., the number_of_filters to initialize.
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)
        
        #Convolution between input image and filters. NOTE: we do not add b and then activate by sigmoid.
        conv_out = conv.conv2d(input=input, filters=self.W, filter_shape=filter_shape, image_shape=image_shape)
        #Max pooling
        #pooled_out = pool.max_pool_2d(input=conv_out, ds=poolsize, ignore_border=True)
        pooled_out = pool.pool_2d(input=conv_out, ds=poolsize, ignore_border=True, mode='max')
        #Add bias and then activate. Since b is 1D vector, we use dimshuffle to reshape. Ex. if b is (10,)
        #then b.dimshuffle('x', 0, 'x', 'x') reshapes b to the dimension (1, 10, 1, 1)
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        #The parameters of this block
        self.params = [self.W, self.b]
    
class HiddenLayer(object):
    """
    Fully connected layer.
    The class of the hidden layer: INPUT: input; OUTPUT: the neurons in the hidden layer. The input and hidden layers are fully connected.
    Suppose input is an n_in dimensional vector (i.e., n_in neurons), hidden layer has n_out neurons. Then, there are:
    n_in*n_out connections (parameters). So the size of W is (n_in, n_out), each column corresponds to the weights of a neuron in hidden layer.
    b is bias, the dimension is n_out.
    rng: used to initialize W.
    input: the input to train the model. The size of input is (n_example, n_in), each row is a sample, i.e., the input of MLP.
    activation: activation function. Here we use tanh.
    """
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
        self.input = input
        if W is None:
            W_values = np.asarray(rng.uniform(low=-np.sqrt(6./(n_in+n_out)), high=np.sqrt(6./(n_in+n_out)), size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
            W = theano.shared(value=W_values, name='W', borrow=True)
        
        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
        
        #Initialize W, b in the hidden layer
        self.W = W
        self.b = b
        
        #Output of the hidden layer
        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None else activation(lin_output))
        
        #Parameters of the hidden layer
        self.params = [self.W, self.b]

class Softmax(object):
    """
    Input: (n_example, n_in) where n_example is the batch size.
    n_in: the output of the previous hidden layer (FC layer).
    n_out: the output, i.e., number of classes.
    """
    def __init__(self, input, n_in, n_out):
        #W is of size (n_in, n_out), b is a vector of dim n_out.
        self.W = theano.shared(value=np.zeros((n_in, n_out), dtype=theano.config.floatX), name='W', borrow=True)
        self.b = theano.shared(value=np.zeros((n_out,), dtype=theano.config.floatX), name='b', borrow=True)
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        #The parameters of the softmax regressor
        self.params = [self.W, self.b]
    
    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
    
    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred', ('y', y.type, 'y_pred', self.y_pred.type))
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()



def load_data(dataset):
    """
    Load mnist data.
    """
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        #Check if the dataset is in the data directory.
        new_path = os.path.join(os.path.split(__file__)[0], "..", "data", dataset)
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path
    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = ('http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz')
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)
    
    print '... loading data'
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    
    #Set the data to be shared variables, only shared variables can be saved to GPU memory.
    #GPU can only save float fata.
    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow) #np array to theano tensor
        shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)  
        return shared_x, T.cast(shared_y, 'int32')
    
    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)
    
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    return rval

def evaluate_lenet5(learning_rate=0.1, n_epochs=200,
                    dataset='mnist.pkl.gz',
                    nkerns=[20, 50], batch_size=500):
    """ Demonstrates lenet on MNIST dataset
    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)
    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer
    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)
    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    rng = np.random.RandomState(23455)

    #Load data
    #datasets = load_data(dataset)
    #train_set_x, train_set_y = datasets[0]
    #valid_set_x, valid_set_y = datasets[1]
    #test_set_x, test_set_y = datasets[2]
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    img_rows, img_cols = 28, 28
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape: ', x_train.shape)
    print(x_train.shape[0], 'train_samples')
    
    train_set_x = x_train[:50000, :, :, :]; train_set_y = y_train[:50000]
    valid_set_x = x_train[50000:, :, :, :]; valid_set_y = y_train[50000:]
    test_set_x = x_test; test_set_y = y_test
    
    #Set the data to be shared variables, only shared variables can be saved to GPU memory.
    #GPU can only save float fata.
    def shared_dataset(datax, datay, borrow=True):
        data_x, data_y = datax, datay
        shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow) #np array to theano tensor
        shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)  
        return shared_x, T.cast(shared_y, 'int32')
    test_set_x, test_set_y = shared_dataset(test_set_x, test_set_y)
    valid_set_x, valid_set_y = shared_dataset(valid_set_x, valid_set_y)
    train_set_x, train_set_y = shared_dataset(train_set_x, train_set_y)

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    layer0_input = x.reshape((batch_size, 1, 28, 28))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, 28, 28),
        filter_shape=(nkerns[0], 1, 5, 5),
        poolsize=(2, 2)
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 12, 12),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(2, 2)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 4 * 4,
        n_out=500,
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    layer3 = Softmax(input=layer2.output, n_in=500, n_out=10)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print 'training @ iter = ', iter
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in xrange(n_test_batches)
                    ]
                    test_score = np.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
' ran for %.2fm' % ((end_time - start_time) / 60.))

'''
def evaluate_lenet5(learning_rate=0.1, n_epochs=200, dataset='mnist.pkl.gz', nkerns=[20, 50], batch_size=100):
    """
    learning rate: the parameter in front of SGD.
    batch_size: the size of each minibatch.
    nkerns=[20, 50], the number of kernels in each convolutional layer.
    """
    rng = np.random.RandomState(23455)
    #Load data
    #datasets = load_data(dataset)
    #train_set_x, train_set_y = datasets[0]
    #valid_set_x, valid_set_y = datasets[1]
    #test_set_x, test_set_y = datasets[2]
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    img_rows, img_cols = 28, 28
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape: ', x_train.shape)
    print(x_train.shape[0], 'train_samples')
    
    train_set_x = x_train[:50000, :, :, :]; train_set_y = y_train[:50000]
    valid_set_x = x_train[50000:, :, :, :]; valid_set_y = y_train[50000:]
    test_set_x = x_test; test_set_y = y_test
    
    #Set the data to be shared variables, only shared variables can be saved to GPU memory.
    #GPU can only save float fata.
    def shared_dataset(datax, datay, borrow=True):
        data_x, data_y = datax, datay
        shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow) #np array to theano tensor
        shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)  
        return shared_x, T.cast(shared_y, 'int32')
    test_set_x, test_set_y = shared_dataset(test_set_x, test_set_y)
    valid_set_x, valid_set_y = shared_dataset(valid_set_x, valid_set_y)
    train_set_x, train_set_y = shared_dataset(train_set_x, train_set_y)
    
    
    #Compute the number of batches
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] #Theano tensor to np array
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size
    
    #Define a few variables, index is index of the batch. x is the training feature, y is the label.
    index = T.lscalar()
    x = T.matrix('x')  #Define a theano matrix
    y = T.ivector('y') #Define a theano vector
    
    print('---Build the model.')
    
    #The loaded batch data shape is (batch_size, 28*28), while LeNetConvPooling's input is 4D, so we need reshape.
    layer0_input = x.reshape((batch_size, 1, 28, 28))
    
    #layer0 is the first LeNetConvPoolLayer, input is (28, 28), after conv we get (28-5+1, 28-5+1) = (24, 24).
    #After pooling, we get (24/2, 24/2) = (12, 12).
    #Each batch containd batch_size images, the first block has nkerns[0] kernel, so the outout of layer0 is (batch_size, nkern[0], 12, 12).
    layer0 = LeNetConvPoolLayer(rng, input=layer0_input, image_shape=(batch_size, 1, 28, 28), filter_shape=(nkerns[0], 1, 5, 5), poolsize=(2, 2))
    
    #Layer1 is the second LeNetConvPoolLayer, input is the output of layer0. Each FM size is (12, 12), after conv we get size (12-5+1, 12-5+1)=(8, 8).
    #After maxpooling we get (8/2, 8/2) = (4, 4). Each batch has batch_size FMs, the second block has nkerns[1] kernels.
    #So layer1's output is (batcj_size, nkerns[1], 4, 4)
    layer1 = LeNetConvPoolLayer(rng, input=layer0.output, image_shape=(batch_size, nkerns[0], 12, 12), filter_shape=(nkerns[1], nkerns[0], 5, 5), poolsize=(2, 2))
    
    #Flat the output of the conv layers to a 2D array. layer1's output's shape is (batch_size, nkerns[1], 4, 4) and
    #it is flat to (batch_size, nkerns[1]*4*4) = (500, 800) which is used as the input of layer2.
    layer2_input = layer1.output.flatten(2)  #Test output the theano tensor, and shape
    layer2_input_np = layer2_input.eval()   #Get the value of a theano tensor
    print('Shape: ', layer2_input_np.shape)
    
    layer2 = HiddenLayer(rng, input=layer2_input, n_in=nkerns[1]*4*4, n_out=500, activation=T.tanh)
    
    #The last layer, softmax. Input is the output of layer2 which is (500, 500). The output of layer 3 is (batch_size, n_out)=(500, 10)
    layer3 = Softmax(input=layer2.output, n_in=500, n_out=10)
    
    #The loss function
    cost = layer3.negative_log_likelihood(y)
    
    #test_model computes the error, x, y is given by the index. Then call layer3, layer3 will sequentially calls layer2, layer1, layer0.
    #Input: data; output: the error
    test_model = theano.function([index], layer3.errors(y), givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        })
    #Validation model
    validate_model = theano.function([index], layer3.errors(y), givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],  
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]  
        })
    #Train the model, use SGD
    #The parameters
    params = layer3.params + layer2.params + layer1.params + layer0.params
    #Compute the gradient of each parameters
    grads = T.grads(cost, params)
    #SGD update the parameters.
    updates = [(param_i, param_i-learning_rate*grad_i) for param_i, grad_i in zip(params, grads)]
    
    #Train the model
    train_model = theano.function([index], cost, updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        })
    
    #Start training
    print '... training'
    patience = 10000
    patience_increase = 2
    improvement_threshold = 0.995
    
    #Set validation_frequency such that it will validate on every epoch
    validation_frequency = min(n_train_batches, patience/2)
    
    best_validation_loss = np.inf #Best loss, i.e., the minimum of loss
    best_iter = 0 #Best number of iteration, unit is batch. if best_iter=1000, which means after 1000 batch, the best_validation achieves.
    test_score = 0.
    start_time = time.clock()
    epoch = 0
    done_looping = False
    
    while (epoch < n_epochs) and (not done_looping):
        epoch = epocj + 1
        for minibatch_index in xrange(n_train_batches):
            iter = (epoch-1) * n_train_batches + minbatch_index  #The iter-th iteration (minibatch-wise)
            if iter % 100 == 0:
                print('training @ iter = ', iter)
                cost_ij = train_model(minibatch_index)
                
                if (iter+1) % validation_frequency == 0:
                    #Compute zero-one loss on validation set.
                    validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
                    this_validation_loss = np.mean(validation_losses)
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                        (epoch, minibatch_index + 1, n_train_batches, this_validation_loss * 100.))
                    if this_validation_loss < best_validation_loss:
                        if this_validation_loss < best_validation_loss*improvement_threshold:  
                            patience = max(patience, iter * patience_increase)
                        best_validation_loss = this_validation_loss
                        best_iter = iter
                        test_losses = [test_model(i) for i in xrange(n_test_batches)]
                        test_score = np.mean(test_losses)
                        print(('epoch %i, minibatch %i/%i, test error of best model %f %%') % 
                            (epoch, minibatch_index + 1, n_train_batches, test_score * 100.))
            if patience <= iter:
                done_looping = True
                break
    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' + os.path.split(__file__)[1] + ' ran for %.2fm' % ((end_time - start_time) / 60.))
'''



if __name__ == '__main__':
    evaluate_lenet5()



"""
Test theano tensor and np array transformation.
import numpy as np
import theano
import theano.tensor as T
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
xT = theano.shared(x)
y = xT.get_value()
#Or
y = xT.eval() #transfer tensor to np array
"""
