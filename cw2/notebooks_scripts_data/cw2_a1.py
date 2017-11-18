import argparse

from mlp.layers import AffineLayer, ReluLayer,DropoutLayer
from mlp.errors import CrossEntropySoftmaxError
from mlp.models import MultipleLayerModel
from mlp.initialisers import ConstantInit, GlorotUniformInit
from mlp.learning_rules import GradientDescentLearningRule
from collections import OrderedDict
import numpy as np

from common import load_data, load_data_hog, train_and_save_results, L2Penalty

#########################################################################################
stats_interval = 1
seed=10102016

def network_dropout(num_epochs):
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
        DropoutLayer(rng, hyper["dropout_prob"]),
        AffineLayer(input_dim, hidden_dim, weights_init, biases_init), 
        ReluLayer(),
        DropoutLayer(rng, hyper["dropout_prob"]),
        AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init), 
        ReluLayer(),
        DropoutLayer(rng, hyper["dropout_prob"]),
        AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
    ])

    error = CrossEntropySoftmaxError()
    
    learning_rule = GradientDescentLearningRule(hyper["learning_rate"])

    train_data, valid_data, test_data = load_data(rng, batch_size=hyper["batch_size"])

    train_and_save_results(
        "network_dropout",
        model,
        error,
        learning_rule,
        hyper,
        train_data,
        valid_data,
        test_data,
        stats_interval
        )

def network_regularization(num_epochs):
    hyper = OrderedDict()
    hyper["learning_rate"] = 0.001
    hyper["l2_coeff"] = 1e-2
    hyper["batch_size"] = 50
    hyper["num_epochs"] = num_epochs

    input_dim, output_dim, hidden_dim = 784, 47, 100

    rng = np.random.RandomState(seed)

    weights_init = GlorotUniformInit(rng=rng, gain=2.**0.5)
    biases_init = ConstantInit(0.)

    weights_penalty = L2Penalty(hyper["l2_coeff"])
    biases_penalty = L2Penalty(hyper["l2_coeff"])

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
    
    learning_rule = GradientDescentLearningRule(hyper["learning_rate"])

    train_data, valid_data, test_data = load_data(rng, batch_size=hyper["batch_size"])

    train_and_save_results(
        "network_regularization",
        model,
        error,
        learning_rule,
        hyper,
        train_data,
        valid_data,
        test_data,
        stats_interval
        )

def network_hog(num_epochs):
    hyper = OrderedDict()
    hyper["learning_rate"] = 0.001
    hyper["batch_size"] = 50
    hyper["num_epochs"] = num_epochs

    input_dim, output_dim, hidden_dim = 36, 47, 100

    rng = np.random.RandomState(seed)

    weights_init = GlorotUniformInit(rng=rng, gain=2.**0.5)
    biases_init = ConstantInit(0.)

    model = MultipleLayerModel([
        AffineLayer(input_dim, hidden_dim, weights_init, biases_init), 
        ReluLayer(),
        AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init), 
        ReluLayer(),
        AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
    ])

    error = CrossEntropySoftmaxError()
    
    learning_rule = GradientDescentLearningRule(hyper["learning_rate"])

    train_data, valid_data, test_data = load_data_hog(rng, batch_size=hyper["batch_size"])

    train_and_save_results(
        "network_hog",
        model,
        error,
        learning_rule,
        hyper,
        train_data,
        valid_data,
        test_data,
        stats_interval
        )

def train_networks(model_type, num_epochs):
    if model_type == 0:
        network_dropout(num_epochs)
    elif model_type == 1:
        network_regularization(num_epochs)
    elif model_type == 2:
        network_hog(num_epochs)
    else:
        print("No valid model")

#########################################################################################

parser = argparse.ArgumentParser(description="Baseline systems for coursework 2")
parser.add_argument('model_type', type=int, help="Type of classifier")
parser.add_argument('-n', dest='num_epochs', type=int, default=100)

args = parser.parse_args()
model_type = args.model_type
num_epochs = args.num_epochs

train_networks(model_type, num_epochs)