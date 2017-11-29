import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog
import logging
from mlp.data_providers import MNISTDataProvider, EMNISTDataProvider
from mlp.optimisers import Optimiser
from collections import OrderedDict

plt.style.use('ggplot')

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

def train_and_save_results(
    file_name, model, error, learning_rule, hyper, train_data, valid_data, test_data, stats_interval):
    
    _ = train_model_and_plot_stats(
        model, 
        error, 
        learning_rule, 
        train_data, 
        valid_data,
        test_data, 
        hyper['num_epochs'], 
        stats_interval, 
        notebook=False)

    header = ""

    for key in hyper:
        header = header + ", " + key + ": " + str(hyper[key])

    for key in _[3]:
        header = header + ", " + key + ": " + str(_[3][key])

    header = header + ", time: " + str(_[2])
    header = header[2:]

    write_to_log_file("{:s}_log.txt".format(file_name), header, _[1], _[0])
    _[4].savefig("{:s}_error.png".format(file_name), dpi=90)
    _[6].savefig("{:s}_accuracy.png".format(file_name), dpi=90)

def train_model_and_plot_stats(
        model, error, learning_rule, train_data, valid_data, test_data, num_epochs, stats_interval, notebook=True):
    # As well as monitoring the error over training also monitor classification
    # accuracy i.e. proportion of most-probable predicted classes being equal to targets
    data_monitors={'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}

    # Use the created objects to initialise a new Optimiser instance.
    optimiser = Optimiser(
        model, error, learning_rule, train_data, valid_data, data_monitors, notebook=notebook)

    # Run the optimiser for 5 epochs (full passes through the training set)
    # printing statistics every epoch.
    stats, keys, run_time = optimiser.train(num_epochs=num_epochs, stats_interval=stats_interval)

    # Test after training
    tester = Tester(model, error, test_data, data_monitors)
    test_metrics = tester.test()

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
    
    return stats, keys, run_time, test_metrics, fig_1, ax_1, fig_2, ax_2

def normalize_data(data):
    mean = np.mean(data)
    std = np.std(data)

    return (data - mean) / std

def load_data(rng, batch_size=100):
    # The below code will set up the data providers, random number
    # generator and logger objects needed for training runs. As
    # loading the data from file take a little while you generally
    # will probably not want to reload the data providers on
    # every training run. If you wish to reset their state you
    # should instead use the .reset() method of the data providers.
    
    # Set up a logger object to print info about the training run to stdout
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = [logging.StreamHandler()]

    # Create data provider objects for the MNIST data set
    train_data = EMNISTDataProvider('train', batch_size=batch_size, rng=rng)
    valid_data = EMNISTDataProvider('valid', batch_size=batch_size, rng=rng)
    test_data = EMNISTDataProvider('valid', batch_size=batch_size, rng=rng)

    # Normalize data
    train_data.inputs = normalize_data(train_data.inputs)
    valid_data.inputs = normalize_data(valid_data.inputs)
    test_data.inputs = normalize_data(test_data.inputs)

    return train_data, valid_data, test_data

def extract_hog(data_set):
    num_samples, size_sample = data_set.shape
    size_digit = int(np.sqrt(size_sample))
    
    reshaped_data = data_set.reshape((num_samples, size_digit, size_digit))

    hog_data = []

    for elem in reshaped_data:
        hog_data.append(extract_features(elem))

    return np.array(hog_data)

def load_data_hog(rng, batch_size=100):
    train_data, valid_data, test_data = load_data(rng, batch_size)

    train_data.inputs = extract_hog(train_data.inputs)
    valid_data.inputs = extract_hog(valid_data.inputs)
    test_data.inputs = extract_hog(test_data.inputs)

    return train_data, valid_data, test_data

# from https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/2b62a1c3-e151-4a0e-b6b6-e424fa46ceab/lessons/fd66c083-4ccb-4fe3-bda1-c29db76f50a0/concepts/d479f43a-7bbb-4de7-9452-f6b991ece599
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                                  visualise=True, feature_vector=False)
        return features, hog_image
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                       visualise=False, feature_vector=feature_vec)
        return features

def extract_features(image):
    # Define HOG parameters
    orient = 9
    pix_per_cell = 14
    cell_per_block = 1
    
    hog_feats = get_hog_features(
            image, 
            orient, 
            pix_per_cell, 
            cell_per_block, 
            vis=False, 
            feature_vec=True
        )

    return hog_feats

class Tester(object):
    """Basic model tester"""

    def __init__(self, model, error, test_dataset, data_monitors):
        self.model = model
        self.error = error
        self.test_dataset = test_dataset
        self.data_monitors = OrderedDict([('error', error)])
        self.data_monitors.update(data_monitors)

    def eval_monitors(self, dataset, label):
        """Evaluates the monitors for the given dataset.

        Args:
            dataset: Dataset to perform evaluation with.
            label: Tag to add to end of monitor keys to identify dataset.

        Returns:
            OrderedDict of monitor values evaluated on dataset.
        """
        data_mon_vals = OrderedDict([(key + label, 0.) for key
                                     in self.data_monitors.keys()])
        for inputs_batch, targets_batch in dataset:
            activations = self.model.fprop(inputs_batch, evaluation=True)
            for key, data_monitor in self.data_monitors.items():
                data_mon_vals[key + label] += data_monitor(
                    activations[-1], targets_batch)
        for key, data_monitor in self.data_monitors.items():
            data_mon_vals[key + label] /= dataset.num_batches
        return data_mon_vals

    def test(self):

        return self.eval_monitors(self.test_dataset, '(test)')


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
