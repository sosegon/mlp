import numpy as np
from mlp.layers import BatchNormalizationLayer
test_inputs = np.array([[-1.38066782, -0.94725498, -3.05585424,  2.28644454,  0.85520889,
         0.10575624,  0.23618609,  0.84723205,  1.06569909, -2.21704034],
       [ 0.11060968, -0.0747448 ,  0.56809029,  2.45926149, -2.28677816,
        -0.9964566 ,  2.7356007 ,  1.98002308, -0.39032315,  1.46515481]])
test_grads_wrt_outputs = np.array([[-0.43857052,  1.00380109, -1.18425494,  0.00486091,  0.21470207,
        -0.12179054, -0.11508482,  0.738482  , -1.17249238,  0.69188295],
       [ 1.07802015,  0.69901145,  0.81603688, -1.76743026, -1.24418692,
        -0.65729963, -0.50834305, -0.49016145,  1.63749743, -0.71123104]])

#produce BatchNorm fprop and bprop
activation_layer = BatchNormalizationLayer(input_dim=10)

beta = np.array(10*[0.3])
gamma = np.array(10*[0.5])

activation_layer.params = [gamma, beta]
BN_fprop = activation_layer.fprop(test_inputs)
BN_bprop = activation_layer.bprop(
    test_inputs, BN_fprop, test_grads_wrt_outputs)
BN_grads_wrt_params = activation_layer.grads_wrt_params(
    test_inputs, test_grads_wrt_outputs)

###############################################################################
###############################################################################
###############################################################################

true_fprop_outputs = np.array([[-0.1999955 , -0.19998686, -0.19999924, -0.1996655 ,  0.79999899,
         0.79999177, -0.1999984 , -0.19999221,  0.79999528, -0.19999926],
       [ 0.7999955 ,  0.79998686,  0.79999924,  0.7996655 , -0.19999899,
        -0.19999177,  0.7999984 ,  0.79999221, -0.19999528,  0.79999926]])
assert BN_fprop.shape == true_fprop_outputs.shape, (
    'Layer bprop returns incorrect shaped array. '
    'Correct shape is \n\n{0}\n\n but returned shape is \n\n{1}.'
    .format(true_fprop_outputs.shape, BN_fprop.shape)
)
assert np.allclose(np.round(BN_fprop, decimals=2), np.round(true_fprop_outputs, decimals=2)), (
'Layer bprop does not return correct values. '
'Correct output is \n\n{0}\n\n but returned output is \n\n{1}\n\n difference is \n\n{2}'
.format(true_fprop_outputs, BN_fprop, BN_fprop-true_fprop_outputs)
)

print("Batch Normalization F-prop test passed")

###############################################################################
###############################################################################
###############################################################################

true_bprop_outputs = np.array([[ -9.14558020e-06,   9.17665617e-06,  -8.40575535e-07,
          6.85384297e-03,   9.40668131e-07,   7.99795574e-06,
          5.03719464e-07,   1.69038704e-05,  -1.82061629e-05,
          5.62083224e-07],
       [  9.14558020e-06,  -9.17665617e-06,   8.40575535e-07,
         -6.85384297e-03,  -9.40668131e-07,  -7.99795574e-06,
         -5.03719464e-07,  -1.69038704e-05,   1.82061629e-05,
         -5.62083224e-07]])
assert BN_bprop.shape == true_bprop_outputs.shape, (
    'Layer bprop returns incorrect shaped array. '
    'Correct shape is \n\n{0}\n\n but returned shape is \n\n{1}.'
    .format(true_bprop_outputs.shape, BN_bprop.shape)
)
assert np.allclose(np.round(BN_bprop, decimals=2), np.round(true_bprop_outputs, decimals=2)), (
'Layer bprop does not return correct values. '
'Correct output is \n\n{0}\n\n but returned output is \n\n{1}\n\n difference is \n\n{2}'
.format(true_bprop_outputs, BN_bprop, BN_bprop-true_bprop_outputs)
)

print("Batch Normalization B-prop test passed")

###############################################################################
###############################################################################
###############################################################################

grads_wrt_gamma, grads_wrt_beta = BN_grads_wrt_params
true_grads_wrt_gamma = np.array(([ 1.51657703, -0.30478163,  2.00028878, -1.77110552,  1.45888603,
        0.53550028, -0.39325697, -1.2286243 , -2.8099633 , -1.40311192]))
true_grads_wrt_beta = np.array([ 0.63944963,  1.70281254, -0.36821806, -1.76256935, -1.02948485,
       -0.77909018, -0.62342786,  0.24832055,  0.46500505, -0.01934809])

assert grads_wrt_gamma.shape == true_grads_wrt_gamma.shape, (
    'Layer bprop returns incorrect shaped array. '
    'Correct shape is \n\n{0}\n\n but returned shape is \n\n{1}.'
    .format(true_grads_wrt_gamma.shape, grads_wrt_gamma.shape)
)
assert np.allclose(np.round(grads_wrt_gamma, decimals=2), np.round(true_grads_wrt_gamma, decimals=2)), (
'Layer bprop does not return correct values. '
'Correct output is \n\n{0}\n\n but returned output is \n\n{1}\n\n difference is \n\n{2}'
.format(true_grads_wrt_gamma, grads_wrt_gamma, grads_wrt_gamma-true_grads_wrt_gamma)
)

assert grads_wrt_beta.shape == true_grads_wrt_beta.shape, (
    'Layer bprop returns incorrect shaped array. '
    'Correct shape is \n\n{0}\n\n but returned shape is \n\n{1}.'
    .format(true_grads_wrt_beta.shape, grads_wrt_beta.shape)
)
assert np.allclose(np.round(grads_wrt_beta, decimals=2), np.round(true_grads_wrt_beta, decimals=2)), (
'Layer bprop does not return correct values. '
'Correct output is \n\n{0}\n\n but returned output is \n\n{1}\n\n difference is \n\n{2}'
.format(true_grads_wrt_beta, grads_wrt_beta, grads_wrt_beta-true_grads_wrt_beta)
)

print("Batch Normalization grads wrt to params test passed")
