import tensorflow as tf
from tensorflow.python.ops import math_ops, init_ops
from tensorflow.python.ops.rnn_cell_impl import RNNCell
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import _linear
from tensorflow.python.layers import base as base_layer
from tensorflow.contrib.rnn import LSTMStateTuple
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.eager import context
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.keras.layers import InputSpec as input_spec
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.util import nest

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

class LSTM_old(tf.nn.rnn_cell.LSTMCell):
  def __init__(self, *args, **kwargs):
    kwargs['state_is_tuple'] = False
    returns = super(LSTM_old, self).__init__(*args, **kwargs)
    self._output_size = self._state_size
    return returns

  def __call__(self, inputs, state):
    output, next_state = super(LSTM_old, self).__call__(inputs, state)
    return next_state, next_state

class LSTM(tf.nn.rnn_cell.LSTMCell):
  """def __init__(self, *args, **kwargs):
    kwargs['state_is_tuple'] = False
    returns = super(LSTM, self).__init__(*args, **kwargs)
    self._output_size = self._state_size
    return returns

  def __call__(self, inputs, state):
    output, next_state= super(LSTM, self).__call__(inputs, state)
    return next_state, next_state"""

  """Basic LSTM recurrent network cell.

  The implementation is based on: http://arxiv.org/abs/1409.2329.

  We add forget_bias (default: 1) to the biases of the forget gate in order to
  reduce the scale of forgetting in the beginning of the training.

  It does not allow cell clipping, a projection layer, and does not
  use peep-hole connections: it is the basic baseline.

  For advanced models, please use the full @{tf.nn.rnn_cell.LSTMCell}
  that follows.
  """

  def __init__(self, num_units, forget_bias=1.0,
               state_is_tuple=False, activation=None, reuse=None, name=None):
    """Initialize the basic LSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell.
      forget_bias: float, The bias added to forget gates (see above).
        Must set to `0.0` manually when restoring from CudnnLSTM-trained
        checkpoints.
      state_is_tuple: If True, accepted and returned states are 2-tuples of
        the `c_state` and `m_state`.  If False, they are concatenated
        along the column axis.  The latter behavior will soon be deprecated.
      activation: Activation function of the inner states.  Default: `tanh`.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
      name: String, the name of the layer. Layers with the same name will
        share weights, but to avoid mistakes we require reuse=True in such
        cases.

      When restoring from CudnnLSTM-trained checkpoints, must use
      `CudnnCompatibleLSTMCell` instead.
    """
    super(LSTM, self).__init__(num_units=num_units, reuse=reuse, name=name)

    # Inputs must be 2-dimensional.
    self.input_spec = base_layer.InputSpec(ndim=2)

    self._num_units = num_units
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    self._activation = activation or math_ops.tanh

  @property
  def state_size(self):
    return (LSTMStateTuple(self._num_units, self._num_units)
            if self._state_is_tuple else 2 * self._num_units)

  @property
  def output_size(self):
    return self._num_units

  def build(self, inputs_shape):
    if inputs_shape[1].value is None:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                       % inputs_shape)

    input_depth = inputs_shape[1].value
    h_depth = self._num_units
    self._kernel = self.add_variable(
        _WEIGHTS_VARIABLE_NAME,
        shape=[input_depth + h_depth, 4 * self._num_units])
    self._bias = self.add_variable(
        _BIAS_VARIABLE_NAME,
        shape=[4 * self._num_units],
        initializer=init_ops.zeros_initializer(dtype=self.dtype))

    self.built = True

  def call(self, inputs, state):
    """Long short-term memory cell (LSTM).

    Args:
      inputs: `2-D` tensor with shape `[batch_size, input_size]`.
      state: An `LSTMStateTuple` of state tensors, each shaped
        `[batch_size, self.state_size]`, if `state_is_tuple` has been set to
        `True`.  Otherwise, a `Tensor` shaped
        `[batch_size, 2 * self.state_size]`.

    Returns:
      A pair containing the new hidden state, and the new state (either a
        `LSTMStateTuple` or a concatenated state, depending on
        `state_is_tuple`).
    """
    sigmoid = math_ops.sigmoid
    one = constant_op.constant(1, dtype=dtypes.int32)
    # Parameters of gates are concatenated into one multiply for efficiency.
    if self._state_is_tuple:
      c, h = state
    else:
      c, h = array_ops.split(value=state, num_or_size_splits=2, axis=one)

    gate_inputs = math_ops.matmul(
        array_ops.concat([inputs, h], 1), self._kernel)
    gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)

    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    i, j, f, o = array_ops.split(
        value=gate_inputs, num_or_size_splits=4, axis=one)

    forget_bias_tensor = constant_op.constant(self._forget_bias, dtype=f.dtype)
    # Note that using `add` and `multiply` instead of `+` and `*` gives a
    # performance improvement. So using those at the cost of readability.
    add = math_ops.add
    multiply = math_ops.multiply
    new_c = add(multiply(c, sigmoid(add(f, forget_bias_tensor))),
                multiply(sigmoid(i), self._activation(j)))
    new_h = multiply(self._activation(new_c), sigmoid(o))

    if self._state_is_tuple:
      new_state = LSTMStateTuple(new_c, new_h)
    else:
      new_state = array_ops.concat([new_c, new_h], 1)

    return new_h, new_state



class MinimalRNN(RNNCell):
    """Minimal RNN where Phi is a multi-layer perceptron.
       This implementation is based on:
       Minmin Chen (2017)
       "MinimalRNN: Toward More Interpretable and Trainable Recurrent Neural Networks"
       https://arxiv.org/abs/1711.06788.pdf
    """

    def __init__(self,
                 num_units,
                 activation=None,
                 kernel_initializer=None,
                 bias_initializer=None):
      """Initialize the parameters for a cell.
        Args:
          num_units: list of int, layer sizes for Phi
          kernel_initializer: (optional) The initializer to use for the weight and
            projection matrices.
          bias_initializer: (optional) The initializer to use for the bias matrices.
            Default: vectors of ones.
      """
      super(MinimalRNN, self).__init__(_reuse=True)

      self._activation = activation or math_ops.tanh
      self._num_units = num_units
      self._kernel_initializer = kernel_initializer
      self._bias_initializer = bias_initializer

    @property
    def state_size(self):
      return self._num_units[-1]

    @property
    def output_size(self):
      return self._num_units[-1]

    def __call__(self, inputs, state, scope=None):
        """Run one step of minimal RNN.
          Args:
            inputs: input Tensor, 2D, batch x num_units.
            state: a state Tensor, `2-D, batch x state_size`.
          Returns:
            A tuple containing:
            - A `2-D, [batch x num_units]`, Tensor representing the output of the
              cell after reading `inputs` when previous state was `state`.
            - A `2-D, [batch x num_units]`, Tensor representing the new state of cell after reading `inputs` when
              the previous state was `state`.  Same type and shape(s) as `state`.
          Raises:
            ValueError:
            - If input size cannot be inferred from inputs via
              static shape inference.
            - If state is not `2D`.
        """
        # Phi projection to a latent space / candidate
        #z = inputs
        z = self._activation(inputs)

        """for i, layer_size in enumerate(self._num_units):
          with tf.variable_scope("phi_" + str(i)):
            z = self._activation(_linear(
                z,
                layer_size,
                True,
                bias_initializer=self._bias_initializer,
                kernel_initializer=self._kernel_initializer))"""

        # Update gate
        bias_ones = self._bias_initializer
        if self._bias_initializer is None:
          bias_ones = init_ops.constant_initializer(1.0, dtype=inputs.dtype)
        with tf.variable_scope("update_gate"):
          arg = _linear(
              [state, z],
              self._num_units[-1],
              True,
              bias_initializer=bias_ones,
              kernel_initializer=self._kernel_initializer)
          u = math_ops.sigmoid(arg)

        # Activation step
        new_h = u * state + (1 - u) * z

        return new_h, new_h

class MGUCell(RNNCell):
    """Minimal Gated Unit (MGU) for recurrent neural networks.

    This implementation is based on:
         http://link.springer.com/article/10.1007/s11633-016-1006-2
    """

    def __init__(self, num_units, input_size=None):
        self._num_units = num_units
        self._input_size = num_units if input_size is None else input_size

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """MGU with nunits cells."""
        with tf.variable_scope(scope or type(self).__name__):  # "MGUCell"
            with tf.variable_scope("forget_gate"):
                arg = _linear(
                    [state, inputs],
                    self._num_units,
                    True)
                f = math_ops.sigmoid(arg)

            print(f)

            with tf.variable_scope("candidate"):
                h_tilde = tf.tanh(_linear([inputs, f * state], self._num_units, True))

            h = (1 - f) * state + f * h_tilde

        return h, h
