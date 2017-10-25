import matplotlib.pyplot as plt
#%matplotlib inline
plt.style.use('ggplot')

def write_to_log_file(log_file_name, keys, stats):
    with open(log_file_name, "w+") as log_file:
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


# The below code will set up the data providers, random number
# generator and logger objects needed for training runs. As
# loading the data from file take a little while you generally
# will probably not want to reload the data providers on
# every training run. If you wish to reset their state you
# should instead use the .reset() method of the data providers.
import numpy as np
import logging
from mlp.data_providers import MNISTDataProvider

# Seed a random number generator
seed = 10102016 
rng = np.random.RandomState(seed)
batch_size = 100
# Set up a logger object to print info about the training run to stdout
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = [logging.StreamHandler()]

# Create data provider objects for the MNIST data set
train_data = MNISTDataProvider('train', batch_size=batch_size, rng=rng)
valid_data = MNISTDataProvider('valid', batch_size=batch_size, rng=rng)

# The model set up code below is provided as a starting point.
# You will probably want to add further code cells for the
# different experiments you run.

from mlp.layers import AffineLayer, SoftmaxLayer, SigmoidLayer, ReluLayer, LeakyReluLayer, ELULayer, SELULayer
from mlp.errors import CrossEntropySoftmaxError
from mlp.models import MultipleLayerModel
from mlp.initialisers import ConstantInit, GlorotUniformInit, SELUInit
from mlp.learning_rules import GradientDescentLearningRule
from mlp.optimisers import Optimiser
from time import time

def activation_network(train_data, valid_data, rng, hyper, activation1, activation2, log_file_name):
    #setup hyperparameters
    batch_size = hyper[0]  # number of data points in a batch
    learning_rate = hyper[1]
    num_epochs = hyper[2]

    stats_interval = 1
    input_dim, output_dim, hidden_dim = 784, 10, 100

    # Reset random number generator and data provider states on each run
    # to ensure reproducibility of results
    rng.seed(seed)
    train_data.reset()
    valid_data.reset()

    # Alter data-provider batch size
    train_data.batch_size = batch_size 
    valid_data.batch_size = batch_size

    weights_init = GlorotUniformInit(rng=rng)
    biases_init = ConstantInit(0.)

    model = MultipleLayerModel([
        AffineLayer(input_dim, hidden_dim, weights_init, biases_init), 
        activation1,
        AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
        activation2,
        AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
    ])

    error = CrossEntropySoftmaxError()
    # Use a basic gradient descent learning rule
    learning_rule = GradientDescentLearningRule(learning_rate=learning_rate)

    start = time()
    #Remember to use notebook=False when you write a script to be run in a terminal
    _ = train_model_and_plot_stats(
        model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=False)
    end = time()
    print("Total training time: {:.2f}s".format(end - start))

    write_to_log_file(log_file_name, _[1], _[0])


activation_network(train_data, valid_data, rng, [100, 0.05, 100], SigmoidLayer(), SigmoidLayer(), "sig_0_05_guniinit.txt")
activation_network(train_data, valid_data, rng, [100, 0.05, 100], ReluLayer(), ReluLayer(), "relu_0_05_guniinit.txt")
activation_network(train_data, valid_data, rng, [100, 0.05, 100], LeakyReluLayer(), LeakyReluLayer(), "lrelu_0_05_guniinit.txt")
activation_network(train_data, valid_data, rng, [100, 0.05, 100], ELULayer(), ELULayer(), "elu_0_05_guniinit.txt")
activation_network(train_data, valid_data, rng, [100, 0.05, 100], SELULayer(), SELULayer(), "selu_0_05_guniinit.txt")