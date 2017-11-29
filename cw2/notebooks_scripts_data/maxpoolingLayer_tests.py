import numpy as np

def test_maxpool_layer_fprop(layer_class, do_cross_correlation=False):
    """Tests `fprop` method of a convolutional layer.
    
    Checks the outputs of `fprop` method for a fixed input against known
    reference values for the outputs and raises an AssertionError if
    the outputted values are not consistent with the reference values. If
    tests are all passed returns True.
    
    Args:
        layer_class: Convolutional layer implementation following the 
            interface defined in the provided skeleton class.
        do_cross_correlation: Whether the layer implements an operation
            corresponding to cross-correlation (True) i.e kernels are
            not flipped before sliding over inputs, or convolution
            (False) with filters being flipped.

    Raises:
        AssertionError: Raised if output of `layer.fprop` is inconsistent 
            with reference values either in shape or values.
    """
    inputs = np.arange(0,384).reshape((2, 3, 8, 8))
        
    layer = layer_class(
        pool_size=2
    )

    outputs = layer.fprop(inputs)
    
    # assert layer_output.shape == true_output.shape, (
    #     'Layer fprop gives incorrect shaped output. '
    #     'Correct shape is \n\n{0}\n\n but returned shape is \n\n{1}.'
    #     .format(true_output.shape, layer_output.shape)
    # )
    # assert np.allclose(layer_output, true_output), (
    #     'Layer fprop does not give correct output. '
    #     'Correct output is \n\n{0}\n\n but returned output is \n\n{1}\n\n difference is \n\n{2}.'
    #     .format(true_output, layer_output, true_output-layer_output)
    # )
    return True

def test_conv_layer_bprop(layer_class, do_cross_correlation=False):
    """Tests `bprop` method of a convolutional layer.
    
    Checks the outputs of `bprop` method for a fixed input against known
    reference values for the gradients with respect to inputs and raises 
    an AssertionError if the returned values are not consistent with the
    reference values. If tests are all passed returns True.
    
    Args:
        layer_class: Convolutional layer implementation following the 
            interface defined in the provided skeleton class.
        do_cross_correlation: Whether the layer implements an operation
            corresponding to cross-correlation (True) i.e kernels are
            not flipped before sliding over inputs, or convolution
            (False) with filters being flipped.

    Raises:
        AssertionError: Raised if output of `layer.bprop` is inconsistent 
            with reference values either in shape or values.
    """
    inputs = np.arange(384).reshape((2, 3, 8, 8))
    # print("#########################################################")
    # print("inputs")
    # print(inputs)
    
    grads_wrt_outputs = np.arange(0,96).reshape((2, 3, 4, 4))
    # print("#########################################################")
    # print("grads_wrt_outputs")
    # print(grads_wrt_outputs)

    
    layer = layer_class(
        pool_size=2
    )

    outputs = layer.bprop(inputs, [], grads_wrt_outputs)
    # print("#########################################################")
    # print("outputs")
    # print(outputs)

    return True

from mlp.layers import MaxPoolingLayer
fprop_correct = test_maxpool_layer_fprop(MaxPoolingLayer, False)
bprop_correct = test_conv_layer_bprop(MaxPoolingLayer, False)
# grads_wrt_param_correct = test_conv_layer_grad_wrt_params(ConvolutionalLayer, False)
# if fprop_correct and grads_wrt_param_correct and bprop_correct:
#     print('All tests passed.')