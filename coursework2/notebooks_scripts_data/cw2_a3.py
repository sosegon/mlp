import argparse

from mlp.layers import AffineLayer, ReluLayer,DropoutLayer
from mlp.errors import CrossEntropySoftmaxError
from mlp.models import MultipleLayerModel
from mlp.initialisers import ConstantInit, GlorotUniformInit
from mlp.learning_rules import GradientDescentLearningRule, AdamLearningRule, RMSPropRule
from collections import OrderedDict
import numpy as np

from common import load_data, load_data_hog, train_and_save_results, L2Penalty

#########################################################################################
stats_interval = 1
seed=10102016

def network_dropout(exp_name, learning_rule, hyper):
    hyper["dropout_prob"] = 0.5

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
    
    train_data, valid_data, test_data = load_data(rng, batch_size=hyper["batch_size"])

    train_and_save_results(
        exp_name + "_network_dropout_" + type(learning_rule).__name__,
        model,
        error,
        learning_rule,
        hyper,
        train_data,
        valid_data,
        test_data,
        stats_interval
        )

def network_regularization(exp_name, learning_rule, hyper):
    hyper["l2_coeff"] = 1e-2

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
    
    train_data, valid_data, test_data = load_data(rng, batch_size=hyper["batch_size"])

    train_and_save_results(
        exp_name + "_network_regularization_" + type(learning_rule).__name__,
        model,
        error,
        learning_rule,
        hyper,
        train_data,
        valid_data,
        test_data,
        stats_interval
        )

def network_hog(exp_name, learning_rule, hyper):
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
    
    train_data, valid_data, test_data = load_data_hog(rng, batch_size=hyper["batch_size"])

    train_and_save_results(
        exp_name + "_network_hog_" + type(learning_rule).__name__,
        model,
        error,
        learning_rule,
        hyper,
        train_data,
        valid_data,
        test_data,
        stats_interval
        )

def train_networks(exp_name, model_type, learning_rule_type, learning_rate, batch_size, num_epochs):
    hyper = OrderedDict()
    hyper["learning_rate"] = learning_rate
    hyper["batch_size"] = batch_size
    hyper["num_epochs"] = num_epochs
    hyper["epsilon"] = 1e-8

    if learning_rule_type == 0:
        hyper["alpha"] = 0.9
        hyper["beta"] = 0.999
        learning_rule = AdamLearningRule(hyper["learning_rate"], hyper["alpha"], hyper["beta"], hyper["epsilon"])
    elif learning_rule_type == 1:
        hyper["beta"] = 0.9
        learning_rule = RMSPropRule(hyper["learning_rate"], hyper["beta"], hyper["epsilon"])
    else:
        print("No valid learning rule")
        raise NotImplemented

    if model_type == 0:
        network_dropout(exp_name, learning_rule, hyper)
    elif model_type == 1:
        network_regularization(exp_name, learning_rule, hyper)
    elif model_type == 2:
        network_hog(exp_name, learning_rule, hyper)
    else:
        print("No valid model")

#########################################################################################

parser = argparse.ArgumentParser(description="Baseline systems for coursework 2")
parser.add_argument('exp_name', type=str, help="Name of experiment")
parser.add_argument('model_type', type=int, help="Type of classifier")
parser.add_argument('learning_rule_type', type=int, help="Learning rule")
parser.add_argument('-n', dest='num_epochs', type=int, default=100)
parser.add_argument('-l', dest='learning_rate', type=float, default=0.001)
parser.add_argument('-b', dest='batch_size', type=int, default=50)


args = parser.parse_args()
exp_name = args.exp_name
model_type = args.model_type
learning_rule_type = args.learning_rule_type
num_epochs = args.num_epochs
learning_rate = args.learning_rate
batch_size = args.batch_size

train_networks(exp_name, model_type, learning_rule_type, learning_rate, batch_size, num_epochs)