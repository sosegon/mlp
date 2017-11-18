import argparse

from mlp.layers import AffineLayer, ReluLayer,DropoutLayer
from mlp.errors import CrossEntropySoftmaxError
from mlp.models import MultipleLayerModel
from mlp.initialisers import ConstantInit, GlorotUniformInit
from mlp.learning_rules import GradientDescentLearningRule, MomentumLearningRule
import numpy as np

from common import load_data, load_data_hog, train_model_and_plot_stats, write_to_log_file, L2Penalty

#########################################################################################
stats_interval = 1
seed=10102016

def network_drop_out(num_epochs):
	learning_rate = 0.02
	mom_coeff = 0.09
	incl_prob = 0.5
	batch_size = 50

	input_dim, output_dim, hidden_dim = 784, 47, 100

	rng = np.random.RandomState(seed)

	weights_init = GlorotUniformInit(rng=rng, gain=2.**0.5)
	biases_init = ConstantInit(0.)

	model = MultipleLayerModel([
		DropoutLayer(rng, incl_prob),
	    AffineLayer(input_dim, hidden_dim, weights_init, biases_init), 
	    ReluLayer(),
	    DropoutLayer(rng, incl_prob),
	    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init), 
	    ReluLayer(),
	    DropoutLayer(rng, incl_prob),
	    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
	])

	error = CrossEntropySoftmaxError()
	
	learning_rule = MomentumLearningRule(learning_rate, mom_coeff)

	train_data, valid_data = load_data(rng, batch_size=batch_size)

	#Remember to use notebook=False when you write a script to be run in a terminal
	_ = train_model_and_plot_stats(
	    model, 
	    error, 
	    learning_rule, 
	    train_data, 
	    valid_data, 
	    num_epochs, 
	    stats_interval, 
	    notebook=False)
	
	header = "batch_size: {:d}, learning_rate: {:.2f}, mom_coeff: {:.2f}, number_epochs: {:d}, time: {:.2f}".format(
        batch_size,
        learning_rate,
        mom_coeff,
        num_epochs,
        _[2]
	)

	log_file_name = "network_dropout_log.txt"

	write_to_log_file(log_file_name, header, _[1], _[0])

	_[3].savefig("network_dropout_error.png", dpi=90)
	_[5].savefig("network_dropout_accuracy.png", dpi=90)

def network_regularization(num_epochs):
	learning_rate = 0.02
	mom_coeff = 0.09
	batch_size = 50
	penalty_coeff = 1e-2

	input_dim, output_dim, hidden_dim = 784, 47, 100

	rng = np.random.RandomState(seed)

	weights_init = GlorotUniformInit(rng=rng, gain=2.**0.5)
	biases_init = ConstantInit(0.)

	weights_penalty = L2Penalty(penalty_coeff)
	biases_penalty = L2Penalty(penalty_coeff)

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
	
	learning_rule = MomentumLearningRule(learning_rate, mom_coeff)

	train_data, valid_data = load_data(rng, batch_size=batch_size)

	#Remember to use notebook=False when you write a script to be run in a terminal
	_ = train_model_and_plot_stats(
	    model, 
	    error, 
	    learning_rule, 
	    train_data, 
	    valid_data, 
	    num_epochs, 
	    stats_interval, 
	    notebook=False)
	
	header = "batch_size: {:d}, learning_rate: {:.2f}, mom_coeff: {:.2f}, l2_penalty_coeff: {:.5f}, number_epochs: {:d}, time: {:.2f}".format(
        batch_size,
        learning_rate,
        mom_coeff,
        penalty_coeff,
        num_epochs,
        _[2]
	)

	log_file_name = "network_regularization_log.txt"

	write_to_log_file(log_file_name, header, _[1], _[0])

	_[3].savefig("network_regularization_error.png", dpi=90)
	_[5].savefig("network_regularization_accuracy.png", dpi=90)

def network_hog(num_epochs):
	learning_rate = 0.02
	mom_coeff = 0.09
	batch_size = 50

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
	
	learning_rule = MomentumLearningRule(learning_rate, mom_coeff)

	train_data, valid_data = load_data_hog(rng, batch_size=batch_size)

	#Remember to use notebook=False when you write a script to be run in a terminal
	_ = train_model_and_plot_stats(
	    model, 
	    error, 
	    learning_rule, 
	    train_data, 
	    valid_data, 
	    num_epochs, 
	    stats_interval, 
	    notebook=False)
	
	header = "batch_size: {:d}, learning_rate: {:.2f}, mom_coeff: {:.2f}, number_epochs: {:d}, time: {:.2f}".format(
        batch_size,
        learning_rate,
        mom_coeff,
        num_epochs,
        _[2]
	)

	log_file_name = "network_hog_log.txt"

	write_to_log_file(log_file_name, header, _[1], _[0])

	_[3].savefig("network_hog_error.png", dpi=90)
	_[5].savefig("network_hog_accuracy.png", dpi=90)

def train_networks(model_type, num_epochs):
	if model_type == 0:
		network_drop_out(num_epochs)
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