import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample

from hw3_utils import shared_dataset, load_data, load_data_cifar
from hw3_nn import LogisticRegression, HiddenLayer, myMLP, LeNetConvPoolLayer, train_nn, DropConnect
import cPickle, gzip, numpy

def relu(inp):
    out = T.switch(T.gt(inp,0),inp,0)
    return out
def tanh(inp):
    out = T.tanh(inp)
    return out
def sigmoid(inp):
    out = T.nnet.sigmoid(inp)
    return out


def test_dropconnect3(learning_rate=0.1, n_epochs=1000, nkerns=[16,64,20],
            batch_size=20, verbose=True, fileName = 'predictionsDropConnect3_Cifar',activation=tanh,fullyconnected=300,p=0.5):
    """
    Wrapper function for testing LeNet on SVHN dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer

    :type batch_size: int
    :param batch_szie: number of examples in minibatch.

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to.

    """

    rng = numpy.random.RandomState(23455)

    datasets = load_data_cifar()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels
    learning_rate = theano.shared(learning_rate)
    #testing = T.lscalar('testing')
    testing = T.iscalar('testing')
    testValue = testing
    getTestValue = theano.function([testing],testValue)   

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 3 * 32 * 32)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 3, 32, 32))

    # TODO: Construct the first convolutional pooling layer
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size,3,32,32),
        filter_shape=(nkerns[0],3,5,5),
        poolsize=(2,2),
        activation=tanh
    )

    # TODO: Construct the second convolutional pooling layer
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size,nkerns[0],14,14),
        filter_shape=(nkerns[1],nkerns[0],5,5),
        poolsize=(2,2),
        activation=tanh
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    
    layer2 = LeNetConvPoolLayer(
        rng,
        input=layer1.output,
        image_shape=(batch_size,nkerns[1],5,5),
        filter_shape=(nkerns[2],nkerns[1],2,2),
        poolsize=(2,2),
        activation=tanh
    )

    layer3_input = layer2.output.flatten(2)
    
    layer3 = DropConnect(
        rng,
        input=layer3_input,
        n_in=nkerns[2]*2*2,
        n_out=fullyconnected,
        testing=testing,
        activation=activation,
        p=p
    )

    # TODO: classify the values of the fully-connected sigmoidal layer
    layer4 = LogisticRegression(
         input=layer3.output,
         n_in=fullyconnected,
         n_out=10)

    # the cost we minimize during training is the NLL of the model
    cost = layer4.negative_log_likelihood(y)

    print("Model building complete")

   
    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size],
            testing: getTestValue(1)
        },
        on_unused_input='ignore'
    )
    
    getPredictedValue = theano.function(        
        [index],
        layer4.predictedValue(),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size],
            testing: getTestValue(1)
        },
        on_unused_input='ignore'
    )
    
    validate_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size],
            testing: getTestValue(1)
        },
        on_unused_input='ignore'
    )

    # TODO: create a list of all model parameters to be fit by gradient descent
    params = layer4.params+layer3.params  + layer2.params + layer1.params + layer0.params
    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)
    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    
    #updates = [
    #    (param_i, param_i - learning_rate * layer2.maskW.get_value() * grad_i) if (param_i.name == 'WDrop') else (param_i, param_i - learning_rate * layer2.maskb.get_value() * grad_i) if(param_i.name == 'bDrop') else (param_i, param_i - learning_rate * grad_i)
    #    for param_i, grad_i in zip(params, grads)
    #]
    
    updates = []
    momentum = 0.9
    for param in params:
        param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
        if (param.name == 'WDrop'):
            updates.append((param,param - learning_rate.get_value().item() * layer3.maskW.get_value() * param_update))
        elif(param.name == 'bDrop'):
            updates.append((param,param - learning_rate.get_value().item() * layer3.maskb.get_value() * param_update))
        else:
            updates.append((param,param - learning_rate.get_value().item() * param_update))
        updates.append((param_update, momentum*param_update + (1. - momentum)*T.grad(cost, param)))
    '''
    updates = [
        (param_i, param_i - learning_rate.get_value().item() * grad_i) if ((param_i.name == 'WDrop') or (param_i.name == 'bDrop')) else (param_i, param_i - learning_rate.get_value().item() * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]
    '''
    print("Commpiling the train model function")

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size],
            testing : getTestValue(0)
        },
        on_unused_input='ignore',
        allow_input_downcast=True
    )
    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    predictions = train_nn(train_model, validate_model, test_model, getPredictedValue,
        n_train_batches, n_valid_batches, n_test_batches, n_epochs, learning_rate, verbose)

    f = open(fileName, 'wb')
    cPickle.dump(predictions, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()