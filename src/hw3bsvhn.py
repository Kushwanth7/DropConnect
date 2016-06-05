"""
Source Code for Homework 3.b of ECBM E6040, Spring 2016, Columbia University

Instructor: Prof. Aurel A. Lazar

This code is based on
[1] http://deeplearning.net/tutorial/logreg.html
[2] http://deeplearning.net/tutorial/mlp.html
[3] http://deeplearning.net/tutorial/lenet.html
"""
import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
import scipy
import scipy.misc
from scipy import misc
from scipy import ndimage
from hw3_utils import shared_dataset, load_data
from hw3_nn import LogisticRegression, HiddenLayer, myMLP, LeNetConvPoolLayer, train_nn, DropConnect,DropOut
import cPickle, gzip, numpy
import matplotlib.pyplot as plt

def relu(inp):
    out = T.switch(T.gt(inp,0),inp,0)
    return out
def tanh(inp):
    out = T.tanh(inp)
    return out
def sigmoid(inp):
    out = T.nnet.sigmoid(inp)
    return out

#from testing
def rotate90(xn,m):
    r=scipy.ndimage.interpolation.rotate(xn[:,:,0],3,axes=(0,1))
    g=scipy.ndimage.interpolation.rotate(xn[:,:,1],3,axes=(0,1))
    b=scipy.ndimage.interpolation.rotate(xn[:,:,2],3,axes=(0,1))
    r=scipy.misc.imresize(r,28.0/(r.shape[0]))
    g=scipy.misc.imresize(g,28.0/(g.shape[0]))
    b=scipy.misc.imresize(b,28.0/(b.shape[0]))
    xn[:,:,0]=r
    xn[:,:,1]=g
    xn[:,:,2]=b
    return xn
def scale(xn):
    xn2=scipy.ndimage.interpolation.zoom(xn, zoom=(2,2,1))
    xn=scipy.misc.imresize(xn2,28.0/(xn2.shape[0]))
    return xn
def crop(xn,start):
    xcrop=xn[start[0]:start[0]+28,start[1]:start[1]+28,:]
    return xcrop
def cropimg(x):
    start=numpy.random.randint(4, size=2)
    xnew=numpy.zeros((x.shape[0],28*28*3))
    temp2=numpy.zeros((x.shape[0],28,28,3))
    for i in range(x.shape[0]):         
        temp=numpy.zeros((32,32,3))
        temp = numpy.reshape(x[i],(32,32,3))   
        xnew[i]=crop(temp,start).flatten()
    return xnew
def translate_image(xvals,typ):
    for x in range(xvals.shape[0]):    
        old=xvals[x].shape
        temp=numpy.zeros((28,28,3))
        temp = numpy.reshape(xvals[x],(28,28,3))
        if typ==1:
            xvals[x]=rotate90(temp,1).flatten()
        if typ==2:
            xvals[x]=flipimg(temp).flatten()
        if typ==3:
            xvals[x]=scale(temp).flatten()
    return xvals
#from testing

# TODO
def test_dropout(learning_rate=0.01, n_epochs=1000, nkerns=[32, 32,64],
            batch_size=120, verbose=False, fileName = 'predictions',fullyconnected=20,activation=tanh,p=0.5):
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

    # Load the dataset
    # Load the dataset
    datasets = load_data()        

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
    learning_rate = theano.shared(learning_rate)

    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels
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
        poolsize=(2,2)
    )

    # TODO: Construct the second convolutional pooling layer
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size,nkerns[0],14,14),
        filter_shape=(nkerns[1],nkerns[0],5,5),
        poolsize=(2,2)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    
    layer2 = LeNetConvPoolLayer(
        rng,
        input=layer1.output,
        image_shape=(batch_size,nkerns[1],5,5),
        filter_shape=(nkerns[2],nkerns[1],2,2),
        poolsize=(2,2)
    )

    layer3_input = layer2.output.flatten(2)


    # TODO: construct a fully-connected sigmoidal layer
    layer3 = DropOut(
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
    params =layer4.params+ layer3.params  + layer2.params + layer1.params + layer0.params
    
    grads = T.grad(cost,params)
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
            updates.append((param,param - learning_rate.get_value().item() * param_update))
        elif(param.name == 'bDrop'):
            updates.append((param,param - learning_rate.get_value().item() * param_update))
        else:
            updates.append((param,param - learning_rate.get_value().item() * param_update))
        updates.append((param_update, momentum*param_update + (1. - momentum)*T.grad(cost, param)))
                
            
    '''
    updates = [
        (param_i, param_i - learning_rate * grad_i) if ((param_i.name == 'WDrop') or (param_i.name == 'bDrop')) else (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]
    '''

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size],
            testing: getTestValue(1)
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

def test_dropout_crop(learning_rate=0.01, n_epochs=1000, nkerns=[32, 32,64],
            batch_size=120, verbose=False, fileName = 'predictions',fullyconnected=20,activation=tanh,p=0.5):
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

    # Load the dataset in raw Format, since we need to preprocess it
    train_set, valid_set, test_set = load_data(theano_shared=False)

    #from testing
    new_train_setx=cropimg(train_set[0])
    print('crop over')
    new_train_sety=train_set[1]
    print('y is intialized')
    new_train_setx1=cropimg(train_set[0])
    print('crop over')
    
    new_train_setx = numpy.vstack((new_train_setx,
                       new_train_setx1,
                                   ))
    print('x is stacked')
   
    new_train_sety = numpy.vstack((numpy.reshape(train_set[1],(train_set[1].shape[0],1)),
                                   numpy.reshape(train_set[1],(train_set[1].shape[0],1))                                                           
                                   ))  
    
    print('y is stacked')  
    
    print new_train_setx.shape
    print new_train_sety.shape     
    new_train_sety=new_train_sety.flatten()
    print new_train_sety.shape 
    new_train_set=(new_train_setx,new_train_sety)
    print('all is well') 
    
    
    new_test_setx=cropimg(test_set[0])
    print('crop over')
    new_valid_setx=cropimg(valid_set[0])
    print('crop over')
    new_test_set = (new_test_setx,test_set[1])
    new_valid_set = (new_valid_setx,valid_set[1])
    
    
    # Convert raw dataset to Theano shared variables.
    test_set_x, test_set_y = shared_dataset(new_test_set)
    valid_set_x, valid_set_y = shared_dataset(new_valid_set)
    train_set_x, train_set_y = shared_dataset(new_train_set)
    #from testing
    #train_set_x, train_set_y = shared_dataset(train_set)
    #valid_set_x, valid_set_y = shared_dataset(valid_set)
    #test_set_x, test_set_y = shared_dataset(test_set)

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
    testing = T.iscalar('testing')
    testValue = testing
    getTestValue = theano.function([testing],testValue)

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 3 * 32 * 32)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 3, 28, 28))

    # TODO: Construct the first convolutional pooling layer
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size,3,28,28),
        filter_shape=(nkerns[0],3,5,5),
        poolsize=(2,2)
    )

    # TODO: Construct the second convolutional pooling layer
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size,nkerns[0],12,12),
        filter_shape=(nkerns[1],nkerns[0],5,5),
        poolsize=(2,2)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    
    layer2 = LeNetConvPoolLayer(
        rng,
        input=layer1.output,
        image_shape=(batch_size,nkerns[1],4,4),
        filter_shape=(nkerns[2],nkerns[1],3,3),
        poolsize=(2,2)
    )

    layer3_input = layer2.output.flatten(2)


    # TODO: construct a fully-connected sigmoidal layer
    layer3 = DropOut(
        rng,
        input=layer3_input,
        n_in=nkerns[2]*1*1,
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
    
    grads = T.grad(cost,params)
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
            updates.append((param,param - learning_rate.get_value().item() * param_update))
        elif(param.name == 'bDrop'):
            updates.append((param,param - learning_rate.get_value().item() * param_update))
        else:
            updates.append((param,param - learning_rate.get_value().item() * param_update))
        updates.append((param_update, momentum*param_update + (1. - momentum)*T.grad(cost, param)))
                
            
    '''
    updates = [
        (param_i, param_i - learning_rate * grad_i) if ((param_i.name == 'WDrop') or (param_i.name == 'bDrop')) else (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]
    '''

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size],
            testing: getTestValue(1)
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


def test_dropout_cropscalrot(learning_rate=0.01, n_epochs=1000, nkerns=[32, 32,64],
            batch_size=120, verbose=False, fileName = 'predictions',fullyconnected=20,activation=tanh,p=0.5):
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

    # Load the dataset in raw Format, since we need to preprocess it
    train_set, valid_set, test_set = load_data(theano_shared=False)
    
    #from testing
    num=int(0.15*train_set[1].shape[0])
    start=numpy.random.choice(train_set[1].shape[0], num, replace=False).tolist()
    print('random 15')
    
    new_train_setx=cropimg(train_set[0])
    print('crop over')
    new_train_setx1=numpy.array(new_train_setx[start], copy=True)
    new_train_setx2=numpy.array(new_train_setx[start], copy=True)
    print('x is intialized')
    
    new_train_sety=train_set[1]    
    new_train_sety=train_set[1][start]
    print('y is intialized')  
    
    
    new_train_setx = numpy.vstack((new_train_setx,new_train_setx1,new_train_setx2))                         
    print('x is stacked')
   
    new_train_sety = numpy.vstack((numpy.reshape(train_set[1],(train_set[1].shape[0],1)),numpy.reshape(new_train_sety,(new_train_sety.shape[0],1)),numpy.reshape(new_train_sety,(new_train_sety.shape[0],1))))  
    
    print('y is stacked')  
    
    print new_train_setx.shape
    print new_train_sety.shape     
    new_train_sety=new_train_sety.flatten()
    print new_train_sety.shape 
    new_train_set=(new_train_setx,new_train_sety)
    print('all is well') 
    
    
    new_test_setx=cropimg(test_set[0])
    print('crop over')
    new_valid_setx=cropimg(valid_set[0])
    print('crop over')
    new_test_set = (new_test_setx,test_set[1])
    new_valid_set = (new_valid_setx,valid_set[1])

    
    
    # Convert raw dataset to Theano shared variables.
    test_set_x, test_set_y = shared_dataset(new_test_set)
    valid_set_x, valid_set_y = shared_dataset(new_valid_set)
    train_set_x, train_set_y = shared_dataset(new_train_set)
    #from testing
    #train_set_x, train_set_y = shared_dataset(train_set)
    #valid_set_x, valid_set_y = shared_dataset(valid_set)
    #test_set_x, test_set_y = shared_dataset(test_set)

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
    testing = T.iscalar('testing')
    testValue = testing
    getTestValue = theano.function([testing],testValue)

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 3 * 32 * 32)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 3, 28, 28))

    # TODO: Construct the first convolutional pooling layer
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size,3,28,28),
        filter_shape=(nkerns[0],3,5,5),
        poolsize=(2,2)
    )

    # TODO: Construct the second convolutional pooling layer
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size,nkerns[0],12,12),
        filter_shape=(nkerns[1],nkerns[0],5,5),
        poolsize=(2,2)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    
    layer2 = LeNetConvPoolLayer(
        rng,
        input=layer1.output,
        image_shape=(batch_size,nkerns[1],4,4),
        filter_shape=(nkerns[2],nkerns[1],3,3),
        poolsize=(2,2)
    )

    layer3_input = layer2.output.flatten(2)


    # TODO: construct a fully-connected sigmoidal layer
    layer3 = DropOut(
        rng,
        input=layer3_input,
        n_in=nkerns[2]*1*1,
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
    
    grads = T.grad(cost,params)
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
            updates.append((param,param - learning_rate.get_value().item() * param_update))
        elif(param.name == 'bDrop'):
            updates.append((param,param - learning_rate.get_value().item() * param_update))
        else:
            updates.append((param,param - learning_rate.get_value().item() * param_update))
        updates.append((param_update, momentum*param_update + (1. - momentum)*T.grad(cost, param)))
                
            
    '''
    updates = [
        (param_i, param_i - learning_rate * grad_i) if ((param_i.name == 'WDrop') or (param_i.name == 'bDrop')) else (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]
    '''

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size],
            testing: getTestValue(1)
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



def test_dropconnect(learning_rate=0.1, n_epochs=1000, nkerns=[32, 32,64],
            batch_size=120, verbose=False, fileName = 'predictions',fullyconnected=20,activation=tanh,p=0.5):
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

    # Load the dataset
    # Load the dataset
    datasets = load_data()        

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
    learning_rate = theano.shared(learning_rate)

    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels
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
        poolsize=(2,2)
    )

    # TODO: Construct the second convolutional pooling layer
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size,nkerns[0],14,14),
        filter_shape=(nkerns[1],nkerns[0],5,5),
        poolsize=(2,2)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    
    layer2 = LeNetConvPoolLayer(
        rng,
        input=layer1.output,
        image_shape=(batch_size,nkerns[1],5,5),
        filter_shape=(nkerns[2],nkerns[1],2,2),
        poolsize=(2,2)
    )

    layer3_input = layer2.output.flatten(2)


    # TODO: construct a fully-connected sigmoidal layer
    layer3 = DropConnect(
        rng,
        input=layer3_input,
        n_in=nkerns[2]*2*2,
        n_out=fullyconnected,
        testing=testing,
        activation = activation,
        p = p
    )

    # TODO: classify the values of the fully-connected sigmoidal layer
    layer4 = LogisticRegression(
         input=layer3.output,
         n_in=fullyconnected,
         n_out=10)

    # the cost we minimize during training is the NLL of the model
    cost = layer4.negative_log_likelihood(y)


   
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
    
    grads = T.grad(cost,params)
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
        (param_i, param_i - learning_rate * grad_i) if ((param_i.name == 'WDrop') or (param_i.name == 'bDrop')) else (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]
    '''

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size],
            testing: getTestValue(1)
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
    
    
def test_dropconnect_crop(learning_rate=0.1, n_epochs=1000, nkerns=[32, 32,64],
            batch_size=120, verbose=False, fileName = 'predictions',activation=tanh,fullyconnected=20,p=0.5):
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

    # Load the dataset in raw Format, since we need to preprocess it
    train_set, valid_set, test_set = load_data(theano_shared=False)

    #from testing
    new_train_setx=cropimg(train_set[0])
    print('crop over')
    new_train_sety=train_set[1]
    print('y is intialized')
    new_train_setx1=cropimg(train_set[0])
    print('crop over')
    
    new_train_setx = numpy.vstack((new_train_setx,
                       new_train_setx1,
                                   ))
    print('x is stacked')
   
    new_train_sety = numpy.vstack((numpy.reshape(train_set[1],(train_set[1].shape[0],1)),
                                   numpy.reshape(train_set[1],(train_set[1].shape[0],1))                                                           
                                   ))  
    
    print('y is stacked')  
    
    print new_train_setx.shape
    print new_train_sety.shape     
    new_train_sety=new_train_sety.flatten()
    print new_train_sety.shape 
    new_train_set=(new_train_setx,new_train_sety)
    print('all is well') 
    
    
    new_test_setx=cropimg(test_set[0])
    print('crop over')
    new_valid_setx=cropimg(valid_set[0])
    print('crop over')
    new_test_set = (new_test_setx,test_set[1])
    new_valid_set = (new_valid_setx,valid_set[1])
    
    
    # Convert raw dataset to Theano shared variables.
    test_set_x, test_set_y = shared_dataset(new_test_set)
    valid_set_x, valid_set_y = shared_dataset(new_valid_set)
    train_set_x, train_set_y = shared_dataset(new_train_set)

    
    
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
    testing = T.iscalar('testing')
    testValue = testing
    getTestValue = theano.function([testing],testValue)

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 3 * 32 * 32)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 3, 28, 28))

    # TODO: Construct the first convolutional pooling layer
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size,3,28,28),
        filter_shape=(nkerns[0],3,5,5),
        poolsize=(2,2)
    )

    # TODO: Construct the second convolutional pooling layer
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size,nkerns[0],12,12),
        filter_shape=(nkerns[1],nkerns[0],5,5),
        poolsize=(2,2)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    
    layer2 = LeNetConvPoolLayer(
        rng,
        input=layer1.output,
        image_shape=(batch_size,nkerns[1],4,4),
        filter_shape=(nkerns[2],nkerns[1],3,3),
        poolsize=(2,2)
    )

    layer3_input = layer2.output.flatten(2)


    # TODO: construct a fully-connected sigmoidal layer
    layer3 = DropConnect(
        rng,
        input=layer3_input,
        n_in=nkerns[2]*1*1,
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
    
    grads = T.grad(cost,params)
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
        (param_i, param_i - learning_rate * grad_i) if ((param_i.name == 'WDrop') or (param_i.name == 'bDrop')) else (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]
    '''

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size],
            testing: getTestValue(1)
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

def test_dropconnect_cropscalrot(learning_rate=0.1, n_epochs=1000, nkerns=[32, 32,64],
            batch_size=120, verbose=False, fileName = 'predictions',activation=tanh,fullyconnected=20,p=0.5):
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

    # Load the dataset in raw Format, since we need to preprocess it
    train_set, valid_set, test_set = load_data(theano_shared=False)
    
    #from testing
    num=int(0.15*train_set[1].shape[0])
    start=numpy.random.choice(train_set[1].shape[0], num, replace=False).tolist()
    print('random 15')
    
    new_train_setx=cropimg(train_set[0])
    print('crop over')
    new_train_setx1=numpy.array(new_train_setx[start], copy=True)
    new_train_setx2=numpy.array(new_train_setx[start], copy=True)
    print('x is intialized')
    
    new_train_sety=train_set[1]    
    new_train_sety=train_set[1][start]
    print('y is intialized')  
    
    
    new_train_setx = numpy.vstack((new_train_setx,new_train_setx1,new_train_setx2))                         
    print('x is stacked')
   
    new_train_sety = numpy.vstack((numpy.reshape(train_set[1],(train_set[1].shape[0],1)),numpy.reshape(new_train_sety,(new_train_sety.shape[0],1)),numpy.reshape(new_train_sety,(new_train_sety.shape[0],1))))  
    
    print('y is stacked')  
    
    print new_train_setx.shape
    print new_train_sety.shape     
    new_train_sety=new_train_sety.flatten()
    print new_train_sety.shape 
    new_train_set=(new_train_setx,new_train_sety)
    print('all is well') 
    
    
    new_test_setx=cropimg(test_set[0])
    print('crop over')
    new_valid_setx=cropimg(valid_set[0])
    print('crop over')
    new_test_set = (new_test_setx,test_set[1])
    new_valid_set = (new_valid_setx,valid_set[1])

    
    
    # Convert raw dataset to Theano shared variables.
    test_set_x, test_set_y = shared_dataset(new_test_set)
    valid_set_x, valid_set_y = shared_dataset(new_valid_set)
    train_set_x, train_set_y = shared_dataset(new_train_set)

    #from testing
    #train_set_x, train_set_y = shared_dataset(train_set)
    #valid_set_x, valid_set_y = shared_dataset(valid_set)
    #test_set_x, test_set_y = shared_dataset(test_set)
    
    
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
    testing = T.iscalar('testing')
    testValue = testing
    getTestValue = theano.function([testing],testValue)

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 3 * 32 * 32)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # Reshape matrix of rasterized images of shape (batch_size, 3 * 32 * 32)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 3, 28, 28))

    # TODO: Construct the first convolutional pooling layer
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size,3,28,28),
        filter_shape=(nkerns[0],3,5,5),
        poolsize=(2,2)
    )

    # TODO: Construct the second convolutional pooling layer
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size,nkerns[0],12,12),
        filter_shape=(nkerns[1],nkerns[0],5,5),
        poolsize=(2,2)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    
    layer2 = LeNetConvPoolLayer(
        rng,
        input=layer1.output,
        image_shape=(batch_size,nkerns[1],4,4),
        filter_shape=(nkerns[2],nkerns[1],3,3),
        poolsize=(2,2)
    )

    layer3_input = layer2.output.flatten(2)


    # TODO: construct a fully-connected sigmoidal layer
    layer3 = DropConnect(
        rng,
        input=layer3_input,
        n_in=nkerns[2]*1*1,
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
    
    grads = T.grad(cost,params)
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
        (param_i, param_i - learning_rate * grad_i) if ((param_i.name == 'WDrop') or (param_i.name == 'bDrop')) else (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]
    '''

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size],
            testing: getTestValue(1)
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

    
def test_lenet(learning_rate=0.1, n_epochs=200, nkerns=[32,32,64],
            batch_size=20, verbose=True, fileName = 'predictions',activation=relu,fullyconnected=20):
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

    # Load the dataset
    # Load the dataset
    datasets = load_data()        

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
    learning_rate = theano.shared(learning_rate)

    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

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
        poolsize=(2,2)
    )

    # TODO: Construct the second convolutional pooling layer
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size,nkerns[0],14,14),
        filter_shape=(nkerns[1],nkerns[0],5,5),
        poolsize=(2,2)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    
    layer2 = LeNetConvPoolLayer(
        rng,
        input=layer1.output,
        image_shape=(batch_size,nkerns[1],5,5),
        filter_shape=(nkerns[2],nkerns[1],2,2),
        poolsize=(2,2)
    )

    layer3_input = layer2.output.flatten(2)


    # TODO: construct a fully-connected sigmoidal layer
    layer3 = HiddenLayer(
        rng,
        input=layer3_input,
        n_in=nkerns[2]*2*2,
        n_out=fullyconnected,
        activation=activation
    )

    # TODO: classify the values of the fully-connected sigmoidal layer
    layer4 = LogisticRegression(
         input=layer3.output,
         n_in=fullyconnected,
         n_out=10)

    # the cost we minimize during training is the NLL of the model
    cost = layer4.negative_log_likelihood(y)


   
    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        },
        on_unused_input='ignore'
    )
    
    getPredictedValue = theano.function(        
        [index],
        layer4.predictedValue(),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        },
        on_unused_input='ignore'
    )

    validate_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        },
        on_unused_input='ignore'
    )
    # TODO: create a list of all model parameters to be fit by gradient descent
    params =layer4.params+ layer3.params  + layer2.params + layer1.params + layer0.params
    
    grads = T.grad(cost,params)
    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    
    #updates = [
    #    (param_i, param_i - learning_rate * layer2.maskW.get_value() * grad_i) if (param_i.name == 'WDrop') else (param_i, param_i - learning_rate * layer2.maskb.get_value() * grad_i) if(param_i.name == 'bDrop') else (param_i, param_i - learning_rate * grad_i)
    #    for param_i, grad_i in zip(params, grads)
    #]
    '''
    updates = []
    momentum = 0.9
    for param in params:
        param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
        if (param.name == 'WDrop'):
            updates.append((param,param - learning_rate.get_value().item() * param_update))
        elif(param.name == 'bDrop'):
            updates.append((param,param - learning_rate.get_value().item() * param_update))
        else:
            updates.append((param,param - learning_rate.get_value().item() * param_update))
        updates.append((param_update, momentum*param_update + (1. - momentum)*T.grad(cost, param)))
                
            
    '''
    updates = [
        (param_i, param_i - learning_rate.get_value().item() * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]
    

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
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
    
def test_lenet_crop(learning_rate=0.01, n_epochs=1000, nkerns=[32, 32,64],
            batch_size=120, verbose=False, fileName = 'predictions',fullyconnected=20,activation=relu):
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

    # Load the dataset in raw Format, since we need to preprocess it
    train_set, valid_set, test_set = load_data(theano_shared=False)

    #from testing
    new_train_setx=cropimg(train_set[0])
    print('crop over')
    new_train_sety=train_set[1]
    print('y is intialized')
    new_train_setx1=cropimg(train_set[0])
    print('crop over')
    
    new_train_setx = numpy.vstack((new_train_setx,
                       new_train_setx1,
                                   ))
    print('x is stacked')
   
    new_train_sety = numpy.vstack((numpy.reshape(train_set[1],(train_set[1].shape[0],1)),
                                   numpy.reshape(train_set[1],(train_set[1].shape[0],1))                                                           
                                   ))  
    
    print('y is stacked')  
    
    print new_train_setx.shape
    print new_train_sety.shape     
    new_train_sety=new_train_sety.flatten()
    print new_train_sety.shape 
    new_train_set=(new_train_setx,new_train_sety)
    print('all is well') 
    
    
    new_test_setx=cropimg(test_set[0])
    print('crop over')
    new_valid_setx=cropimg(valid_set[0])
    print('crop over')
    new_test_set = (new_test_setx,test_set[1])
    new_valid_set = (new_valid_setx,valid_set[1])
    
    
    # Convert raw dataset to Theano shared variables.
    test_set_x, test_set_y = shared_dataset(new_test_set)
    valid_set_x, valid_set_y = shared_dataset(new_valid_set)
    train_set_x, train_set_y = shared_dataset(new_train_set)
    #from testing
    #train_set_x, train_set_y = shared_dataset(train_set)
    #valid_set_x, valid_set_y = shared_dataset(valid_set)
    #test_set_x, test_set_y = shared_dataset(test_set)

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

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 3 * 32 * 32)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 3, 28, 28))

    # TODO: Construct the first convolutional pooling layer
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size,3,28,28),
        filter_shape=(nkerns[0],3,5,5),
        poolsize=(2,2)
    )

    # TODO: Construct the second convolutional pooling layer
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size,nkerns[0],12,12),
        filter_shape=(nkerns[1],nkerns[0],5,5),
        poolsize=(2,2)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    
    layer2 = LeNetConvPoolLayer(
        rng,
        input=layer1.output,
        image_shape=(batch_size,nkerns[1],4,4),
        filter_shape=(nkerns[2],nkerns[1],3,3),
        poolsize=(2,2)
    )

    layer3_input = layer2.output.flatten(2)


    # TODO: construct a fully-connected sigmoidal layer
    layer3 = HiddenLayer(
        rng,
        input=layer3_input,
        n_in=nkerns[2]*1*1,
        n_out=fullyconnected,
        activation=activation
    )

    # TODO: classify the values of the fully-connected sigmoidal layer
    layer4 = LogisticRegression(
         input=layer3.output,
         n_in=fullyconnected,
         n_out=10)

    # the cost we minimize during training is the NLL of the model
    cost = layer4.negative_log_likelihood(y)


   
    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        },
        on_unused_input='ignore'
    )
    
    getPredictedValue = theano.function(        
        [index],
        layer4.predictedValue(),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        },
        on_unused_input='ignore'
    )

    validate_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        },
        on_unused_input='ignore'
    )
    # TODO: create a list of all model parameters to be fit by gradient descent
    params = layer4.params+layer3.params  + layer2.params + layer1.params + layer0.params
    
    grads = T.grad(cost,params)
    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    
    #updates = [
    #    (param_i, param_i - learning_rate * layer2.maskW.get_value() * grad_i) if (param_i.name == 'WDrop') else (param_i, param_i - learning_rate * layer2.maskb.get_value() * grad_i) if(param_i.name == 'bDrop') else (param_i, param_i - learning_rate * grad_i)
    #    for param_i, grad_i in zip(params, grads)
    #]
    '''
    updates = []
    momentum = 0.9
    for param in params:
        param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
        if (param.name == 'WDrop'):
            updates.append((param,param - learning_rate.get_value().item() * param_update))
        elif(param.name == 'bDrop'):
            updates.append((param,param - learning_rate.get_value().item() * param_update))
        else:
            updates.append((param,param - learning_rate.get_value().item() * param_update))
        updates.append((param_update, momentum*param_update + (1. - momentum)*T.grad(cost, param)))
                
            
    '''
    updates = [
        (param_i, param_i - learning_rate.get_value().item() * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]


    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
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

def test_lenet_cropscalrot(learning_rate=0.01, n_epochs=200, nkerns=[32, 32,64],
            batch_size=20, verbose=True, fileName = 'predictions',fullyconnected=20,activation=relu):
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

    # Load the dataset in raw Format, since we need to preprocess it
    train_set, valid_set, test_set = load_data(theano_shared=False)
    
    #from testing
    num=int(0.15*train_set[1].shape[0])
    start=numpy.random.choice(train_set[1].shape[0], num, replace=False).tolist()
    print('random 15')
    
    new_train_setx=cropimg(train_set[0])
    print('crop over')
    new_train_setx1=numpy.array(new_train_setx[start], copy=True)
    new_train_setx2=numpy.array(new_train_setx[start], copy=True)
    print('x is intialized')
    
    new_train_sety=train_set[1]    
    new_train_sety=train_set[1][start]
    print('y is intialized')  
    
    
    new_train_setx = numpy.vstack((new_train_setx,new_train_setx1,new_train_setx2))                         
    print('x is stacked')
   
    new_train_sety = numpy.vstack((numpy.reshape(train_set[1],(train_set[1].shape[0],1)),numpy.reshape(new_train_sety,(new_train_sety.shape[0],1)),numpy.reshape(new_train_sety,(new_train_sety.shape[0],1))))  
    
    print('y is stacked')  
    
    print new_train_setx.shape
    print new_train_sety.shape     
    new_train_sety=new_train_sety.flatten()
    print new_train_sety.shape 
    new_train_set=(new_train_setx,new_train_sety)
    print('all is well') 
    
    
    new_test_setx=cropimg(test_set[0])
    print('crop over')
    new_valid_setx=cropimg(valid_set[0])
    print('crop over')
    new_test_set = (new_test_setx,test_set[1])
    new_valid_set = (new_valid_setx,valid_set[1])

    
    
    # Convert raw dataset to Theano shared variables.
    test_set_x, test_set_y = shared_dataset(new_test_set)
    valid_set_x, valid_set_y = shared_dataset(new_valid_set)
    train_set_x, train_set_y = shared_dataset(new_train_set)
    #from testing
    #train_set_x, train_set_y = shared_dataset(train_set)
    #valid_set_x, valid_set_y = shared_dataset(valid_set)
    #test_set_x, test_set_y = shared_dataset(test_set)

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

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 3 * 32 * 32)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 3, 28, 28))

    # TODO: Construct the first convolutional pooling layer
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size,3,28,28),
        filter_shape=(nkerns[0],3,5,5),
        poolsize=(2,2)
    )

    # TODO: Construct the second convolutional pooling layer
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size,nkerns[0],12,12),
        filter_shape=(nkerns[1],nkerns[0],5,5),
        poolsize=(2,2)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    
    layer2 = LeNetConvPoolLayer(
        rng,
        input=layer1.output,
        image_shape=(batch_size,nkerns[1],4,4),
        filter_shape=(nkerns[2],nkerns[1],3,3),
        poolsize=(2,2)
    )

    layer3_input = layer2.output.flatten(2)


    # TODO: construct a fully-connected sigmoidal layer
    layer3 = HiddenLayer(
        rng,
        input=layer3_input,
        n_in=nkerns[2]*1*1,
        n_out=fullyconnected,
        activation=activation,
    )

    # TODO: classify the values of the fully-connected sigmoidal layer
    layer4 = LogisticRegression(
         input=layer3.output,
         n_in=fullyconnected,
         n_out=10)

    # the cost we minimize during training is the NLL of the model
    cost = layer4.negative_log_likelihood(y)


   
    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        },
        on_unused_input='ignore'
    )
    
    getPredictedValue = theano.function(        
        [index],
        layer4.predictedValue(),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        },
        on_unused_input='ignore'
    )

    validate_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        },
        on_unused_input='ignore'
    )
    # TODO: create a list of all model parameters to be fit by gradient descent
    params = layer4.params+layer3.params  + layer2.params + layer1.params + layer0.params
    
    grads = T.grad(cost,params)
    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    
    #updates = [
    #    (param_i, param_i - learning_rate * layer2.maskW.get_value() * grad_i) if (param_i.name == 'WDrop') else (param_i, param_i - learning_rate * layer2.maskb.get_value() * grad_i) if(param_i.name == 'bDrop') else (param_i, param_i - learning_rate * grad_i)
    #    for param_i, grad_i in zip(params, grads)
    #]
    '''
    updates = []
    momentum = 0.9
    for param in params:
        param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
        if (param.name == 'WDrop'):
            updates.append((param,param - learning_rate.get_value().item() * param_update))
        elif(param.name == 'bDrop'):
            updates.append((param,param - learning_rate.get_value().item() * param_update))
        else:
            updates.append((param,param - learning_rate.get_value().item() * param_update))
        updates.append((param_update, momentum*param_update + (1. - momentum)*T.grad(cost, param)))
                
            
    '''
    updates = [
        (param_i, param_i - learning_rate.get_value().item() * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]
    

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
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

    
    
def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             batch_size=20, n_hidden=500, 
             verbose=True, fileName='predictionsMLP',fullyconnected=20,activation=tanh):
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

    # Load the dataset
    # Load the dataset
    f = gzip.open('/home/ubuntu/mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    
    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

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
    testing = T.lscalar('testing')
    testValue = testing
    getTestValue = theano.function([testing],testValue) 
    
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 3 * 32 * 32)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 1, 28, 28))
    layer0_input = layer0_input.flatten(2)
    # TODO: Construct the first convolutional pooling layer
    layer0 = HiddenLayer(
        rng,
        input=layer0_input,
        n_in=28*28*1,
        n_out=n_hidden,
        activation=activation
    )
    
    layer1 = HiddenLayer(
        rng,
        input=layer0.output,
        n_in=n_hidden,
        n_out=n_hidden,
        activation=activation
    )
    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    
    
    # TODO: construct a fully-connected sigmoidal layer
    layer2 = DropConnect(
        rng,
        input=layer1.output,
        n_in=n_hidden,
        n_out=fullyconnected,
        testing=testing,
        activation=activation
    )
    
    # TODO: classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(
         input=layer2.output,
         n_in=fullyconnected,
         n_out=10)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

    print("Model building complete")

   
    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size],
            testing: getTestValue(1)
        },
        on_unused_input='ignore'
    )
    
    getPredictedValue = theano.function(        
        [index],
        layer3.predictedValue(),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size],
            testing: getTestValue(1)
        },
        on_unused_input='ignore'
    )

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size],
            testing: getTestValue(1)
        },
        on_unused_input='ignore'
    )

    # TODO: create a list of all model parameters to be fit by gradient descent
    params = layer3.params  + layer2.params + layer1.params + layer0.params
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
            updates.append((param,param - learning_rate.get_value().item() * layer2.maskW.get_value() * param_update))
        elif(param.name == 'bDrop'):
            updates.append((param,param - learning_rate.get_value().item() * layer2.maskb.get_value() * param_update))
        else:
            updates.append((param,param - learning_rate.get_value().item() * param_update))
        updates.append((param_update, momentum*param_update + (1. - momentum)*T.grad(cost, param)))
    '''
    updates = [
        (param_i, param_i - learning_rate * grad_i) if ((param_i.name == 'WDrop') or (param_i.name == 'bDrop')) else (param_i, param_i - learning_rate * grad_i)
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


if __name__ == "__main__":
    #test_lenet(verbose=True)
    # test_convnet(verbose=True)
    # test_CDNN()
    pass