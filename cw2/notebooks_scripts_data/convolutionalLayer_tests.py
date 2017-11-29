import numpy as np

def test_conv_layer_fprop(layer_class, do_cross_correlation=False):
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
    inputs = np.arange(0,96).reshape((2, 3, 4, 4))
    kernels = np.arange(-12, 12).reshape((2, 3, 2, 2))
    if do_cross_correlation:
        kernels = kernels[:, :, ::-1, ::-1]
    biases = np.arange(2)
    true_output = np.array(
        [[[[ -958., -1036., -1114.],
           [-1270., -1348., -1426.],
           [-1582., -1660., -1738.]],
          [[ 1707.,  1773.,  1839.],
           [ 1971.,  2037.,  2103.],
           [ 2235.,  2301.,  2367.]]],
         [[[-4702., -4780., -4858.],
           [-5014., -5092., -5170.],
           [-5326., -5404., -5482.]],
          [[ 4875.,  4941.,  5007.],
           [ 5139.,  5205.,  5271.],
           [ 5403.,  5469.,  5535.]]]]
    )
    
    layer = layer_class(
        num_input_channels=kernels.shape[1], 
        num_output_channels=kernels.shape[0], 
        input_dim_1=inputs.shape[2], 
        input_dim_2=inputs.shape[3],
        kernel_dim_1=kernels.shape[2],
        kernel_dim_2=kernels.shape[3]
    )
    layer.params = [kernels, biases]
    layer_output = layer.fprop(inputs)
    
    assert layer_output.shape == true_output.shape, (
        'Layer fprop gives incorrect shaped output. '
        'Correct shape is \n\n{0}\n\n but returned shape is \n\n{1}.'
        .format(true_output.shape, layer_output.shape)
    )
    assert np.allclose(layer_output, true_output), (
        'Layer fprop does not give correct output. '
        'Correct output is \n\n{0}\n\n but returned output is \n\n{1}\n\n difference is \n\n{2}.'
        .format(true_output, layer_output, true_output-layer_output)
    )
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
    inputs = np.arange(96).reshape((2, 3, 4, 4))
    kernels = np.arange(-12, 12).reshape((2, 3, 2, 2))
    if do_cross_correlation:
        kernels = kernels[:, :, ::-1, ::-1]
    biases = np.arange(2)
    grads_wrt_outputs = np.arange(-20, 16).reshape((2, 2, 3, 3))
    outputs = np.array(
        [[[[ -958., -1036., -1114.],
           [-1270., -1348., -1426.],
           [-1582., -1660., -1738.]],
          [[ 1707.,  1773.,  1839.],
           [ 1971.,  2037.,  2103.],
           [ 2235.,  2301.,  2367.]]],
         [[[-4702., -4780., -4858.],
           [-5014., -5092., -5170.],
           [-5326., -5404., -5482.]],
          [[ 4875.,  4941.,  5007.],
           [ 5139.,  5205.,  5271.],
           [ 5403.,  5469.,  5535.]]]]
    )
    true_grads_wrt_inputs = np.array(
      [[[[ 147.,  319.,  305.,  162.],
         [ 338.,  716.,  680.,  354.],
         [ 290.,  608.,  572.,  294.],
         [ 149.,  307.,  285.,  144.]],
        [[  23.,   79.,   81.,   54.],
         [ 114.,  284.,  280.,  162.],
         [ 114.,  272.,  268.,  150.],
         [  73.,  163.,  157.,   84.]],
        [[-101., -161., -143.,  -54.],
         [-110., -148., -120.,  -30.],
         [ -62.,  -64.,  -36.,    6.],
         [  -3.,   19.,   29.,   24.]]],
       [[[  39.,   67.,   53.,   18.],
         [  50.,   68.,   32.,   -6.],
         [   2.,  -40.,  -76.,  -66.],
         [ -31.,  -89., -111.,  -72.]],
        [[  59.,  115.,  117.,   54.],
         [ 114.,  212.,  208.,   90.],
         [ 114.,  200.,  196.,   78.],
         [  37.,   55.,   49.,   12.]],
        [[  79.,  163.,  181.,   90.],
         [ 178.,  356.,  384.,  186.],
         [ 226.,  440.,  468.,  222.],
         [ 105.,  199.,  209.,   96.]]]])
    layer = layer_class(
        num_input_channels=kernels.shape[1], 
        num_output_channels=kernels.shape[0], 
        input_dim_1=inputs.shape[2], 
        input_dim_2=inputs.shape[3],
        kernel_dim_1=kernels.shape[2],
        kernel_dim_2=kernels.shape[3]
    )
    layer.params = [kernels, biases]
    layer_grads_wrt_inputs = layer.bprop(inputs, outputs, grads_wrt_outputs)
    assert layer_grads_wrt_inputs.shape == true_grads_wrt_inputs.shape, (
        'Layer bprop returns incorrect shaped array. '
        'Correct shape is \n\n{0}\n\n but returned shape is \n\n{1}.'
        .format(true_grads_wrt_inputs.shape, layer_grads_wrt_inputs.shape)
    )
    assert np.allclose(layer_grads_wrt_inputs, true_grads_wrt_inputs), (
        'Layer bprop does not return correct values. '
        'Correct output is \n\n{0}\n\n but returned output is \n\n{1}\n\n difference is \n\n{2}'
        .format(true_grads_wrt_inputs, layer_grads_wrt_inputs, layer_grads_wrt_inputs-true_grads_wrt_inputs)
    )
    return True

def test_conv_layer_grad_wrt_params(
        layer_class, do_cross_correlation=False):
    """Tests `grad_wrt_params` method of a convolutional layer.
    
    Checks the outputs of `grad_wrt_params` method for fixed inputs 
    against known reference values for the gradients with respect to 
    kernels and biases, and raises an AssertionError if the returned
    values are not consistent with the reference values. If tests
    are all passed returns True.
    
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
    inputs = np.arange(96).reshape((2, 3, 4, 4))
    kernels = np.arange(-12, 12).reshape((2, 3, 2, 2))
    biases = np.arange(2)
    grads_wrt_outputs = np.arange(-20, 16).reshape((2, 2, 3, 3))
    true_kernel_grads = np.array(
        [[[[ -240.,  -114.],
         [  264.,   390.]],
        [[-2256., -2130.],
         [-1752., -1626.]],
        [[-4272., -4146.],
         [-3768., -3642.]]],
       [[[ 5268.,  5232.],
         [ 5124.,  5088.]],
        [[ 5844.,  5808.],
         [ 5700.,  5664.]],
        [[ 6420.,  6384.],
         [ 6276.,  6240.]]]])
    if do_cross_correlation:
        kernels = kernels[:, :, ::-1, ::-1]
        true_kernel_grads = true_kernel_grads[:, :, ::-1, ::-1]
    true_bias_grads = np.array([-126.,   36.])
    layer = layer_class(
        num_input_channels=kernels.shape[1], 
        num_output_channels=kernels.shape[0], 
        input_dim_1=inputs.shape[2], 
        input_dim_2=inputs.shape[3],
        kernel_dim_1=kernels.shape[2],
        kernel_dim_2=kernels.shape[3]
    )
    layer.params = [kernels, biases]
    layer_kernel_grads, layer_bias_grads = (
        layer.grads_wrt_params(inputs, grads_wrt_outputs))
    assert layer_kernel_grads.shape == true_kernel_grads.shape, (
        'grads_wrt_params gives incorrect shaped kernel gradients output. '
        'Correct shape is \n\n{0}\n\n but returned shape is \n\n{1}.'
        .format(true_kernel_grads.shape, layer_kernel_grads.shape)
    )
    assert np.allclose(layer_kernel_grads, true_kernel_grads), (
        'grads_wrt_params does not give correct kernel gradients output. '
        'Correct output is \n\n{0}\n\n but returned output is \n\n{1}.'
        .format(true_kernel_grads, layer_kernel_grads)
    )
    assert layer_bias_grads.shape == true_bias_grads.shape, (
        'grads_wrt_params gives incorrect shaped bias gradients output. '
        'Correct shape is \n\n{0}\n\n but returned shape is \n\n{1}.'
        .format(true_bias_grads.shape, layer_bias_grads.shape)
    )
    assert np.allclose(layer_bias_grads, true_bias_grads), (
        'grads_wrt_params does not give correct bias gradients output. '
        'Correct output is \n\n{0}\n\n but returned output is \n\n{1}.'
        .format(true_bias_grads, layer_bias_grads)
    )
    return True

from mlp.layers import ConvolutionalLayer
fprop_correct = test_conv_layer_fprop(ConvolutionalLayer, False)
bprop_correct = test_conv_layer_bprop(ConvolutionalLayer, False)
grads_wrt_param_correct = test_conv_layer_grad_wrt_params(ConvolutionalLayer, False)
if fprop_correct and grads_wrt_param_correct and bprop_correct:
    print('All tests passed.')