import argparse

from mlp.layers import AffineLayer, ReluLayer,DropoutLayer, ReshapeLayer, MaxPoolingLayer, ConvolutionalLayer
from mlp.errors import CrossEntropySoftmaxError
from mlp.models import MultipleLayerModel
from mlp.initialisers import ConstantInit, GlorotUniformInit
from mlp.learning_rules import GradientDescentLearningRule
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Flatten, ELU, Activation, Lambda, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam, RMSprop
from keras.initializers import glorot_uniform
from random import randrange
from keras.utils.np_utils import to_categorical

from common import load_data, load_data_hog, train_and_save_results, L2Penalty

#########################################################################################
stats_interval = 1
seed=10102016

def get_model_1(learning):
        
    image_shape = (28, 28, 1)

    print(image_shape)
    model = Sequential()
    model.add(Lambda(lambda x: x, input_shape=image_shape, output_shape=image_shape))
    
    model.add(Convolution2D(
        5, 
        5, 
        kernel_initializer='random_uniform',
        bias_initializer='zeros',
        border_mode="valid"))
    model.add(MaxPooling2D())
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(47, kernel_initializer=glorot_uniform(seed), activation='softmax'))

    optimizer = Adam(lr=learning)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    return model

def get_model_2(learning):
        
    image_shape = (28, 28, 1)

    print(image_shape)
    model = Sequential()
    model.add(Lambda(lambda x: x, input_shape=image_shape, output_shape=image_shape))
    
    model.add(Convolution2D(
        5, 
        5, 
        kernel_initializer='random_uniform',
        bias_initializer='zeros',
        border_mode="valid"))
    model.add(MaxPooling2D())
    model.add(Activation('relu'))

    model.add(Convolution2D(
        10, 
        5, 
        kernel_initializer='random_uniform',
        bias_initializer='zeros',
        border_mode="valid"))
    model.add(MaxPooling2D())
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(47, kernel_initializer=glorot_uniform(seed), activation='softmax'))

    optimizer = Adam(lr=learning)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    return model

def cnn_1(exp_name, num_epochs):
    hyper = OrderedDict()
    hyper["learning_rate"] = 0.001
    hyper["dropout_prob"] = 0.5
    hyper["batch_size"] = 50
    hyper["num_epochs"] = num_epochs

    input_dim, output_dim, hidden_dim = 784, 47, 100

    rng = np.random.RandomState(seed)

    weights_init = GlorotUniformInit(rng=rng, gain=2.**0.5)
    biases_init = ConstantInit(0.)

    model = MultipleLayerModel([
        ReshapeLayer((1, 28, 28)),
        
        ConvolutionalLayer(1, 5, 28, 28, 5, 5), # 5, 24, 24
        MaxPoolingLayer(), # 5, 12, 12
        ReluLayer(), # 720
        
        ReshapeLayer(),
        AffineLayer(5*12*12, output_dim, weights_init, biases_init)
        # AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
    ])

    error = CrossEntropySoftmaxError()
    
    learning_rule = GradientDescentLearningRule(hyper["learning_rate"])

    train_data, valid_data, test_data = load_data(rng, batch_size=hyper["batch_size"])

    train_and_save_results(
        exp_name + "_cnn_1",
        model,
        error,
        learning_rule,
        hyper,
        train_data,
        valid_data,
        test_data,
        stats_interval
        )

def cnn_2(exp_name, num_epochs):
    hyper = OrderedDict()
    hyper["learning_rate"] = 0.001
    hyper["dropout_prob"] = 0.5
    hyper["batch_size"] = 50
    hyper["num_epochs"] = num_epochs

    input_dim, output_dim, hidden_dim = 784, 47, 100

    rng = np.random.RandomState(seed)

    weights_init = GlorotUniformInit(rng=rng, gain=2.**0.5)
    biases_init = ConstantInit(0.)

    model = MultipleLayerModel([
        ReshapeLayer((1, 28, 28)),
        
        ConvolutionalLayer(1, 5, 28, 28, 5, 5), # 5, 24, 24
        MaxPoolingLayer(), # 5, 12, 12
        ReluLayer(), 

        ConvolutionalLayer(5, 10, 12, 12, 5, 5), # 10, 8, 8
        MaxPoolingLayer(), # 10, 4, 4
        ReluLayer(),

        ReshapeLayer(), # 160
        AffineLayer(10*4*4, output_dim, weights_init, biases_init)
    ])

    error = CrossEntropySoftmaxError()
    
    learning_rule = GradientDescentLearningRule(hyper["learning_rate"])

    train_data, valid_data, test_data = load_data(rng, batch_size=hyper["batch_size"])

    train_and_save_results(
        exp_name + "_cnn_2",
        model,
        error,
        learning_rule,
        hyper,
        train_data,
        valid_data,
        test_data,
        stats_interval
        )

def generator(X, y, batch_size):
    total_input = len(X)
    
    while True:
        features, targets = [], []
        i = 0
        while len(features) < batch_size:
            index = randrange(0, total_input)
            feats = X[index]
            labels = y[index]
           
            features.append(feats)
            targets.append(labels)
            
        yield (np.array(features), np.array(targets))

def getFeaturesTargets(X, y):
    feats = []
    targets = []

    for feat, label in zip(X, y):
        feats.append(feat)
        targets.append(label)

    return np.array(feats), np.array(targets)

def save_plot_metrics(log_file_name, history):
    keys = history.history.keys()

    f, ax = plt.subplots(len(keys), 1, figsize=(5, 22))

    for idx, k in enumerate(keys):
        ax[idx].plot(history.history[k])
        ax[idx].set_title("model " + k)
        ax[idx].set_ylabel(k)
        ax[idx].set_xlabel('epoch')
    
    f.savefig("{:s}.png".format(log_file_name), dpi=90)

def save_log_metrics(log_file_name, hyper, history):
    header = ""

    for key in hyper:
        header = header + ", " + key + ": " + str(hyper[key])

    header = header[2:]

    with open(log_file_name + ".txt", "w+") as log_file:
        log_file.write(header+"\n")
        
        keys = history.history.keys()
        head = ""
        
        c = 0
        for k in keys:
            if c == 0:
                l = len(history.history[k]) # number of epochs
                h = np.zeros(l)
            head = head + k + ","
            h = np.vstack((h, history.history[k]))
            c = c + 1

        head = head[:-1]
        head = head + "\n"
        log_file.write(head)

        h = h[1:,:]
        h = h.T

        for row in h:
            new_line = ""
            for value in row:
                new_line = new_line + "{:.8f},".format(value)
            new_line = new_line[:-1]
            new_line = new_line + "\n"
            log_file.write(new_line)

    log_file.close()

def train_model(model, hyper_params, log_file_name):
    learning_rate = hyper_params["learning_rate"]
    training_size = hyper_params["training_size"]
    batch_size = hyper_params["batch_size"]
    num_epochs = hyper_params["num_epochs"]

    rng = np.random.RandomState(seed)

    train_data, valid_data, test_data = load_data(rng, batch_size=hyper_params["batch_size"])

    # Reshape train and valid targets
    n_elem, n_feat = train_data.inputs.shape
    n_feat = int(n_feat**0.5)
    train_data.inputs = train_data.inputs.reshape((n_elem, n_feat, n_feat, 1))
    train_data.targets = to_categorical(train_data.targets)

    n_elem, n_feat = valid_data.inputs.shape
    n_feat = int(n_feat**0.5)
    valid_data.inputs = valid_data.inputs.reshape((n_elem, n_feat, n_feat, 1))
    valid_data.targets = to_categorical(valid_data.targets)  

    n_elem, n_feat = test_data.inputs.shape
    n_feat = int(n_feat**0.5)
    test_data.inputs = test_data.inputs.reshape((n_elem, n_feat, n_feat, 1))
    test_data.targets = to_categorical(test_data.targets)  

    history = model.fit_generator(
        generator(train_data.inputs, train_data.targets, batch_size),
        samples_per_epoch = training_size,
        validation_data = getFeaturesTargets(valid_data.inputs, valid_data.targets),
        nb_epoch = num_epochs
        )

    eval_ = model.evaluate(train_data.inputs, train_data.targets)
    for val, key in zip(eval_, model.metrics_names):
        hyper_params[key] = val

    save_log_metrics(log_file_name, hyper_params, history)
    save_plot_metrics(log_file_name, history)

def train_networks(exp_name, model_type, learning_rate, training_size, batch_size, num_epochs):
    model = None
    log_file_name = None
    if model_type == 1:
        model = get_model_1(learning_rate)
        log_file_name = "{:s}_cnn1_log".format(exp_name)
    elif model_type == 2:
        model = get_model_2(learning_rate)
        log_file_name = "{:s}_cnn2_log".format(exp_name)
    else:
        raise NotImplementedError

    hyper = OrderedDict()
    hyper["learning_rate"] = learning_rate
    hyper["training_size"] = training_size
    hyper["batch_size"] = batch_size
    hyper["num_epochs"] = num_epochs
    
    train_model(model, hyper, log_file_name)

#########################################################################################

parser = argparse.ArgumentParser(description="CNN systems for coursework 2")
parser.add_argument('exp_name', type=str, help="Name of experiment")
parser.add_argument('model_type', type=int, help="Type of classifier")
parser.add_argument('-n', dest='num_epochs', type=int, default=100)
parser.add_argument('-l', dest='learning_rate', type=float, default=0.001)
parser.add_argument('-t', dest='training_size', type=int, default=2000)
parser.add_argument('-b', dest='batch_size', type=int, default=50)

args = parser.parse_args()
exp_name = args.exp_name
model_type = args.model_type
num_epochs = args.num_epochs
learning_rate = args.learning_rate
training_size = args.training_size
batch_size = args.batch_size

train_networks(exp_name, model_type, learning_rate, training_size, batch_size, num_epochs)