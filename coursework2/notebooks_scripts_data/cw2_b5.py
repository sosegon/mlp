import argparse

from mlp.layers import AffineLayer, ReluLayer,DropoutLayer, ReshapeLayer, MaxPoolingLayer, ConvolutionalLayer
from mlp.errors import CrossEntropySoftmaxError
from mlp.models import MultipleLayerModel
from mlp.initialisers import ConstantInit, GlorotUniformInit
from mlp.learning_rules import GradientDescentLearningRule, AdamLearningRule
from collections import OrderedDict
import numpy as np

from common import load_data, load_data_hog, train_and_save_results, L2Penalty

#########################################################################################
stats_interval = 1
seed=10102016

def cnn_1(exp_name, hyper):
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
    
    #learning_rule = GradientDescentLearningRule(hyper["learning_rate"])
    learning_rule = AdamLearningRule(hyper["learning_rate"], hyper["alpha"], hyper["beta"], hyper["epsilon"])

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

def cnn_2(exp_name, hyper):
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
    
    #learning_rule = GradientDescentLearningRule(hyper["learning_rate"])
    learning_rule = AdamLearningRule(hyper["learning_rate"], hyper["alpha"], hyper["beta"], hyper["epsilon"])

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

def train_networks(exp_name, model_type, learning_rate, batch_size, num_epochs):
    hyper = OrderedDict()
    hyper["learning_rate"] = learning_rate
    hyper["batch_size"] = batch_size
    hyper["num_epochs"] = num_epochs
    hyper["epsilon"] = 1e-8
    hyper["alpha"] = 0.9
    hyper["beta"] = 0.999

    if model_type == 1:
        cnn_1(exp_name, hyper)
    elif model_type == 2:
        cnn_2(exp_name, hyper)
    else:
        print("No valid model")

#########################################################################################

parser = argparse.ArgumentParser(description="Baseline systems for coursework 2")
parser.add_argument('exp_name', type=str, help="Name of experiment")
parser.add_argument('model_type', type=int, help="Type of classifier")
parser.add_argument('-n', dest='num_epochs', type=int, default=100)
parser.add_argument('-l', dest='learning_rate', type=float, default=0.001)
parser.add_argument('-b', dest='batch_size', type=int, default=50)

args = parser.parse_args()
exp_name = args.exp_name
model_type = args.model_type
num_epochs = args.num_epochs
learning_rate = args.learning_rate
batch_size = args.batch_size

train_networks(exp_name, model_type, learning_rate, batch_size, num_epochs)