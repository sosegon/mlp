import matplotlib.pyplot as plt
#%matplotlib inline
plt.style.use('ggplot')

import numpy as np
import logging
from mlp.data_providers import MNISTDataProvider

from mlp.layers import AffineLayer, SoftmaxLayer, SigmoidLayer, ReluLayer
from mlp.errors import CrossEntropySoftmaxError
from mlp.models import MultipleLayerModel
from mlp.initialisers import ConstantInit, GlorotUniformInit, SELUInit, NormalInit, UniformInit
from mlp.learning_rules import MomentumLearningRule
from mlp.optimisers import Optimiser
from time import time

import argparse

class L1Penalty(object):
    """L1 parameter penalty.
    
    Term to add to the objective function penalising parameters
    based on their L1 norm.
    """
    
    def __init__(self, coefficient):
        """Create a new L1 penalty object.
        
        Args:
            coefficient: Positive constant to scale penalty term by.
        """
        assert coefficient > 0., 'Penalty coefficient must be positive.'
        self.coefficient = coefficient
        
    def __call__(self, parameter):
        """Calculate L1 penalty value for a parameter.
        
        Args:
            parameter: Array corresponding to a model parameter.
            
        Returns:
            Value of penalty term.
        """
        return np.abs(parameter.ravel()).sum() * self.coefficient
        
    def grad(self, parameter):
        """Calculate the penalty gradient with respect to the parameter.
        
        Args:
            parameter: Array corresponding to a model parameter.
            
        Returns:
            Value of penalty gradient with respect to parameter. This
            should be an array of the same shape as the parameter.
        """
        output = np.copy(parameter)
        output[output < 0] = -1
        output[output > 0] = 1
        return self.coefficient * output
    
    def __repr__(self):
        return 'L1Penalty({0})'.format(self.coefficient)
        

class L2Penalty(object):
    """L1 parameter penalty.
    
    Term to add to the objective function penalising parameters
    based on their L2 norm.
    """

    def __init__(self, coefficient):
        """Create a new L2 penalty object.
        
        Args:
            coefficient: Positive constant to scale penalty term by.
        """
        assert coefficient > 0., 'Penalty coefficient must be positive.'
        self.coefficient = coefficient
        
    def __call__(self, parameter):
        """Calculate L2 penalty value for a parameter.
        
        Args:
            parameter: Array corresponding to a model parameter.
            
        Returns:
            Value of penalty term.
        """
        return (parameter.ravel()**2).sum() * self.coefficient / 2
        
    def grad(self, parameter):
        """Calculate the penalty gradient with respect to the parameter.
        
        Args:
            parameter: Array corresponding to a model parameter.
            
        Returns:
            Value of penalty gradient with respect to parameter. This
        raise NotImplementedError()
            should be an array of the same shape as the parameter.
        """
        return self.coefficient * parameter
    
    def __repr__(self):
        return 'L2Penalty({0})'.format(self.coefficient)

def write_to_log_file(log_file_name, header, keys, stats):
    with open(log_file_name, "w+") as log_file:
        log_file.write(header+"\n")
        head = ""
        for k in keys:
            head = head + k + ","
        
        head = head[:-1]
        head = head + "\n"
        log_file.write(head)
        
        for log in stats:
            new_line = ""
            for value in log:
                new_line = new_line + "{:.8f},".format(value)                
            new_line = new_line[:-1]
            new_line = new_line + "\n"
            log_file.write(new_line)
    
    log_file.close()
    
def train_model_and_plot_stats(
        model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True):
    
    # As well as monitoring the error over training also monitor classification
    # accuracy i.e. proportion of most-probable predicted classes being equal to targets
    data_monitors={'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}

    # Use the created objects to initialise a new Optimiser instance.
    optimiser = Optimiser(
        model, error, learning_rule, train_data, valid_data, data_monitors, notebook=notebook)

    # Run the optimiser for 5 epochs (full passes through the training set)
    # printing statistics every epoch.
    stats, keys, run_time = optimiser.train(num_epochs=num_epochs, stats_interval=stats_interval)    
    
    # Plot the change in the validation and training set error over training.
    fig_1 = plt.figure(figsize=(8, 4))
    ax_1 = fig_1.add_subplot(111)
    for k in ['error(train)', 'error(valid)']:
        ax_1.plot(np.arange(1, stats.shape[0]) * stats_interval, 
                  stats[1:, keys[k]], label=k)
    ax_1.legend(loc=0)
    ax_1.set_xlabel('Epoch number')

    # Plot the change in the validation and training set accuracy over training.
    fig_2 = plt.figure(figsize=(8, 4))
    ax_2 = fig_2.add_subplot(111)
    for k in ['acc(train)', 'acc(valid)']:
        ax_2.plot(np.arange(1, stats.shape[0]) * stats_interval, 
                  stats[1:, keys[k]], label=k)
    ax_2.legend(loc=0)
    ax_2.set_xlabel('Epoch number')
    
    return stats, keys, run_time, fig_1, ax_1, fig_2, ax_2

def normalize_data(data):
    mean = np.mean(data)
    std = np.std(data)

    return (data - mean) / std

def network(train_data, valid_data, hyper, rng, weights_penalty=None, biases_penalty=None, file_name="exp"):
    # Hyperparameters
    batch_size = hyper[0]  # number of data points in a batch
    learning_rate = hyper[1]
    num_epochs = hyper[2]
    
    # Plot variables
    stats_interval = 1
    
    # Reset random number generator and data provider states on each run
    # to ensure reproducibility of results
    train_data.reset()
    valid_data.reset()

    # Alter data-provider batch size
    train_data.batch_size = batch_size 
    valid_data.batch_size = batch_size

    # Network parameters
    weights_init = GlorotUniformInit(0.5, rng)
    biases_init = ConstantInit(0.)
    input_dim, output_dim, hidden_dim = 784, 10, 100
    
    # Network architecture
    model = MultipleLayerModel([
        AffineLayer(input_dim, hidden_dim, weights_init, 
                    biases_init, weights_penalty=weights_penalty,
                    biases_penalty=biases_penalty),
        ReluLayer(),
        AffineLayer(hidden_dim, hidden_dim, weights_init, 
                    biases_init, weights_penalty=weights_penalty,
                    biases_penalty=biases_penalty),
        ReluLayer(),
        AffineLayer(hidden_dim, output_dim, weights_init, 
                    biases_init, weights_penalty=weights_penalty,
                    biases_penalty=biases_penalty)
    ])
    
    error = CrossEntropySoftmaxError()
    
    learning_rule = MomentumLearningRule(learning_rate=learning_rate, mom_coeff=0.9)

    #Remember to use notebook=False when you write a script to be run in a terminal
    _ = train_model_and_plot_stats(
        model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=False)
    print("Total training time: {:.2f}s".format(_[2]))
    
    if weights_penalty is not None:
        wpc = weights_penalty.coefficient
    else:
        wpc = 0
        
    if biases_penalty is not None:
        bpc = biases_penalty.coefficient
    else:
        bpc = 0
    
    header = "batch_size: {:d}, learning_rate: {:.2f}, number_epochs: {:d}, weights_penalty_coeff: {:.6f},  biases_penalty_coeff: {:.6f},time: {:.2f}".format(
        batch_size,
        learning_rate,
        num_epochs,
        wpc,
        bpc,
        _[2]
        )
    
    log_file_name = "{:s}.txt".format(file_name)
    weights_file_name = "{:s}.ws".format(file_name)

    write_to_log_file(log_file_name, header, _[1], _[0])
    np.savetxt(weights_file_name, model.layers[0].weights)

def train_networks(hyper):
    # The below code will set up the data providers, random number
    # generator and logger objects needed for training runs. As
    # loading the data from file take a little while you generally
    # will probably not want to reload the data providers on
    # every training run. If you wish to reset their state you
    # should instead use the .reset() method of the data providers.

    # Seed a random number generator
    seed = 22102017
    rng = np.random.RandomState(seed)
    batch_size = 80
    # Set up a logger object to print info about the training run to stdout
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = [logging.StreamHandler()]

    # Create data provider objects for the MNIST data set
    train_data = MNISTDataProvider('train', batch_size=batch_size, rng=rng)
    valid_data = MNISTDataProvider('valid', batch_size=batch_size, rng=rng)

    train_data.inputs = normalize_data(train_data.inputs)
    valid_data.inputs = normalize_data(valid_data.inputs)

    network(train_data, valid_data, hyper, rng, None, None, "lab5_exp0")
    network(train_data, valid_data, hyper, rng, L1Penalty(1e-5), L1Penalty(1e-5), "lab5_exp1_l1")
    network(train_data, valid_data, hyper, rng, L1Penalty(1e-3), L1Penalty(1e-3), "lab5_exp2_l1")
    network(train_data, valid_data, hyper, rng, L2Penalty(1e-4), L2Penalty(1e-4), "lab5_exp3_l2")
    network(train_data, valid_data, hyper, rng, L2Penalty(1e-2), L2Penalty(1e-2), "lab5_exp4_l2")

parser = argparse.ArgumentParser(description='Train NN with different activations')
parser.add_argument('-b', dest='batch_size', type=int, default=50)
parser.add_argument('-l', dest='learning_rate', type=float, default=0.01)
parser.add_argument('-n', dest='num_epochs', type=int, default=100)

args = parser.parse_args()
batch_size = args.batch_size
learning_rate = args.learning_rate
num_epochs = args.num_epochs

train_networks([batch_size, learning_rate, num_epochs])
