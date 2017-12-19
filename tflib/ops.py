import lasagne
import numpy as np
import tensorflow as tf
import tflib as lib


_WEIGHTNORM = False


def calculate_gain(nonlinearity):
    if nonlinearity == 'sigmoid' or nonlinearity == 'linear':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return np.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        negative_slope = 0.01
        return np.sqrt(2.0 / (1 + negative_slope ** 2))
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))


def weight_initializer(
    name, shape, val=0, gain='linear',
    std=0.01, mean=0.0, range=0.01, alpha=0.01
):
    gain = calculate_gain(gain)

    if name == 'Constant':
        W = lasagne.init.Constant(val).sample(shape)
    elif name == 'Normal':
        W = lasagne.init.Normal(std, mean).sample(shape)
    elif name == 'Uniform':
        W = lasagne.init.Uniform(
            range=range, std=std, mean=mean).sample(shape)
    elif name == 'GlorotNormal':
        W = lasagne.init.GlorotNormal(gain=gain).sample(shape)
    elif name == 'HeNormal':
        W = lasagne.init.HeNormal(gain=gain).sample(shape)
    elif name == 'HeUniform':
        W = lasagne.init.HeUniform(gain=gain).sample(shape)
    elif name == 'Orthogonal':
        W = lasagne.init.Orthogonal(gain=gain).sample(shape)
    else:
        W = lasagne.init.GlorotUniform(gain=gain).sample(shape)

    return W.astype('float32')


def Embedding(name, n_symbols, emb_dim, indices):
    with tf.name_scope(name):
        emb = lib.param(
            name,
            weight_initializer('Normal', [n_symbols, emb_dim], std=1.0/np.sqrt(n_symbols))
            )

        return tf.nn.embedding_lookup(emb, indices)


def Linear(
    name, inputs,
    input_dim, output_dim,
    **kwargs
):
    with tf.name_scope(name):
        weight_values = weight_initializer(
            kwargs.get('init', 'HeNormal'),
            (input_dim, output_dim), gain=kwargs.get('activation', 'linear')
            )

        weight = lib.param(
            name + '.W',
            weight_values
        )

        if _WEIGHTNORM:
            norm_values = np.sqrt(np.sum(np.square(weight_values), axis=0))
            # nort.m_values = np.linalg.norm(weight_values, axis=0)

            target_norms = lib.param(
                name + '.g',
                norm_values
            )

            with tf.name_scope('weightnorm'):
                norms = tf.sqrt(tf.reduce_sum(tf.square(weight), reduction_indices=[0]))
                weight = weight * (target_norms / norms)

        if inputs.get_shape().ndims == 2:
            result = tf.matmul(inputs, weight)
        else:
            reshaped_inputs = tf.reshape(inputs, [-1, input_dim])
            result = tf.matmul(reshaped_inputs, weight)
            result = tf.reshape(result, tf.stack(tf.unstack(tf.shape(inputs))[:-1] + [output_dim]))

        if kwargs.get('bias', True):
            b = lib.param(
                name + '.b',
                weight_initializer('Constant', output_dim, val=0.)
            )

            result = tf.nn.bias_add(result, b)

        if kwargs.get('batchnorm', False):
            result = tf.layers.batch_normalization(
                inputs=result, axis=-1, training=kwargs.get('training_mode', True)
            )

        return result


def Conv2D(
    name, input,
    depth, n_filters, kernel, stride,
    **kwargs
):
    with tf.name_scope(name) as scope:
        filter_values = weight_initializer(
            kwargs.get('init', 'GlorotUniform'),
            (kernel, kernel, depth, n_filters), gain=kwargs.get('activation', 'relu')
            )

        filters = lib.param(
            name+'.W',
            filter_values
        )

        if _WEIGHTNORM:
            norm_values = np.sqrt(np.sum(np.square(filter_values), axis=(0, 1, 2)))
            target_norms = lib.param(
                name + '.g',
                norm_values
            )
            with tf.name_scope('weightnorm'):
                norms = tf.sqrt(tf.reduce_sum(tf.square(filters), reduction_indices=[0, 1, 2]))
                filters = filters * (target_norms / norms)

        out = tf.nn.conv2d(
            input, filters, strides=[1, 1, stride, stride],
            padding=kwargs.get('padding', 'SAME'),
            data_format='NCHW'
            )

        if kwargs.get('bias', True):
            b = lib.param(
                name+'.b',
                weight_initializer('Constant', n_filters, val=0.)
            )

            out = tf.nn.bias_add(out, b, data_format='NCHW')

        if kwargs.get('batchnorm', False):
            # Note: when training, the moving_mean and moving_variance need to be updated.
            # By default the update ops are placed in tf.GraphKeys.UPDATE_OPS,
            # so they need to be added as a dependency to the train_op. For example:
            # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            # with tf.control_dependencies(update_ops):
            #    train_op = optimizer.minimize(loss)
            out = tf.layers.batch_normalization(
                inputs=out, axis=1, training=kwargs.get('training_mode', True)
                )

        return out


def GRUStep(
    name, input_dim, hidden_dim,
    x_t, state_tm1
):
    gates = tf.nn.sigmoid(
        Linear(
            name+'.Gates',
            tf.concat([x_t, state_tm1], 1),
            input_dim + hidden_dim,
            2 * hidden_dim,
            activation='sigmoid'
        )
    )

    update, reset = tf.split(gates, 2, axis=1)
    scaled_state = reset * state_tm1

    candidate = tf.tanh(
        Linear(
            name+'.Candidate',
            tf.concat(axis=1, values=[x_t, scaled_state]),
            input_dim + hidden_dim,
            hidden_dim,
            activation='tanh'
        )
    )

    state_t = (update * candidate) + ((1 - update) * state_tm1)

    return state_t


def LSTMStep(
    name, input_dim, hidden_dim,
    x_t, state_tm1
):
    h_tm1, c_tm1 = tf.split(state_tm1, 2, axis=1)
    gates = Linear(
            name+'.Gates',
            tf.concat(axis=1, values=[x_t, h_tm1]),
            input_dim + hidden_dim,
            4 * hidden_dim,
            activation='sigmoid'
            )

    i_t, f_t, o_t, g_t = tf.split(gates, 4, axis=1)

    # Using forget_gate bias = 1., input_gate bias = -1 and default to 0
    c_t = tf.nn.sigmoid(f_t+1.)*c_tm1 + tf.nn.sigmoid(i_t-1.)*tf.tanh(g_t)
    h_t = tf.nn.sigmoid(o_t)*tf.tanh(c_t)

    state_t = tf.concat(axis=1, values=[h_t, c_t])

    return state_t


def StackedRNNStep(
    type, name, input_dim, hidden_dim,
    x_t, states_tm1,
    n_layers=1
):
    if type == 'LSTM':
        RNNFunc = LSTMStep
    elif type == 'GRU':
        RNNFunc = GRUStep
    else:
        raise NotImplementedError

    output_arr = []
    input_to_rnn = x_t
    states_tm1 = tf.split(states_tm1, n_layers, axis=1)

    for i in xrange(n_layers):
        state_t = RNNFunc(
            name+'.Layer%d' % (i+1),
            input_dim if i == 0 else hidden_dim, hidden_dim,
            input_to_rnn, states_tm1[i]
            )

        output_arr += [state_t]
        # If LSTM need to take only h_t not c_t
        h_t = state_t[:, :hidden_dim]
        input_to_rnn = h_t

    return tf.concat(output_arr, axis=1)


class RNNCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, type, name, input_dim, hidden_dim, n_layers=1):
        self._type = type
        self._name = name
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._n_layers = n_layers
        self._size = 2*hidden_dim if type == 'LSTM' else hidden_dim

    @property
    def state_size(self):
        return self._size*self._n_layers

    @property
    def output_size(self):
        return self._size*self._n_layers

    def __call__(self, x_t, state_tm1, scope=None):
        states = StackedRNNStep(
            self._type, self._name, self._input_dim, self._hidden_dim,
            x_t, state_tm1, n_layers=self._n_layers)

        return states, states


def RNN(
    type, name,
    inputs,
    input_dim, hidden_dim,
    h0=None, mask=None,
    n_layers=1,
    bidirectional=False,
    return_cell_state=False
):
    '''
    inputs:  (BATCH_SIZE, N_STEPS, INPUT_DIM)
    h0:      (N_DIRECTIONS, N_LAYERS * HIDDEN_DIM)
    outputs: (BATCH_SIZE, N_STEPS, N_DIRECTIONS, N_LAYERS, HIDDEN_DIM)
    '''
    size = 2*hidden_dim if type == 'LSTM' else hidden_dim
    batch_size, seq_len, _ = tf.unstack(tf.shape(inputs))

    n_dir = 2 if bidirectional else 1

    if h0 is None:
        h0 = tf.tile(lib.param(
            name+'.h0',
            weight_initializer('Constant', (1, n_dir, n_layers*size), val=0.)
            ), [batch_size, 1, 1])

    if mask is None:
        sequence_length = tf.tile(tf.expand_dims(seq_len, 0), [batch_size])
    else:
        sequence_length = tf.reduce_sum(mask, axis=-1)

    if bidirectional:
        states, _ = tf.nn.bidirectional_dynamic_rnn(
            RNNCell(type, name+'.Forward', input_dim, hidden_dim, n_layers),
            RNNCell(type, name+'.Backward', input_dim, hidden_dim, n_layers),
            inputs,
            sequence_length=sequence_length,
            initial_state_fw=h0[:, 0],
            initial_state_bw=h0[:, 1]
        )
        states = tf.stack(states, axis=2)
    else:
        states, _ = tf.nn.dynamic_rnn(
            RNNCell(type, name, input_dim, hidden_dim, n_layers),
            inputs,
            sequence_length=sequence_length,
            initial_state=h0[:, 0]
        )
        states = tf.expand_dims(states, axis=2)

    states = tf.stack(tf.split(states, n_layers, axis=-1), axis=3)

    if return_cell_state:
        return states
    else:
        return states[:, :, :, :, :hidden_dim]


class AttentionRNNCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self,
                 type, name, ctx, ctx_mask,
                 input_dim, hidden_dim, ctx_dim,
                 n_layers=1, position_gap=1.,
                 closed_loop=False):
        self._type = type
        self._name = name
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._ctx_dim = ctx_dim
        self._n_layers = n_layers
        self._size = 2*hidden_dim if type == 'LSTM' else hidden_dim
        self._ctx = ctx
        self._ctx_mask = tf.to_float(ctx_mask)
        self._closed_loop = closed_loop
        self._position_gap = position_gap

    @property
    def state_size(self):
        if self._closed_loop:
            return (self._size*self._n_layers, 1, self._ctx_dim, 34)
        else:
            return (self._size*self._n_layers, 1, self._ctx_dim)

    @property
    def output_size(self):
        if self._closed_loop:
            return (self._size*self._n_layers, 1, self._ctx_dim, 34)
        else:
            return (self._size*self._n_layers, 1, self._ctx_dim)

    def __call__(self, x_t, state_tm1, scope=None):
        print self._closed_loop
        if self._closed_loop:
            h_tm1, kappa_tm1, w_tm1, x_t = state_tm1
        else:
            h_tm1, kappa_tm1, w_tm1 = state_tm1

        window_params = tf.exp(tf.clip_by_value(Linear(
                self._name+'.AttentionParameters',
                tf.concat([x_t, h_tm1, w_tm1], 1),
                34+self._hidden_dim + self._ctx_dim,
                2
            ), 0., 8.))
        kappa_del, beta_t = tf.split(window_params, 2, axis=1)
        kappa_t = kappa_tm1 + self._position_gap*kappa_del

        u_t = tf.to_float(tf.range(tf.shape(self._ctx)[1]))[None]
        phi_t = -beta_t[:, 0, None]*tf.pow(
            kappa_t[:, 0, None]-u_t, 2)*self._ctx_mask - 1000.*(1. - self._ctx_mask)
        phi_t = tf.nn.softmax(2*phi_t)*self._ctx_mask

        w_t = tf.reduce_sum(phi_t[:, :, None]*self._ctx, axis=1)

        states = StackedRNNStep(
            self._type, self._name, self._input_dim+self._ctx_dim, self._hidden_dim,
            tf.concat([x_t, w_t], 1), h_tm1, n_layers=self._n_layers)

        if self._closed_loop:
            x_tp1 = lib.ops.Linear(
                    'Decoder.Output',
                    tf.concat([states, w_t], 1),
                    self._hidden_dim + self._ctx_dim,
                    34
                )
            state_t = (states, kappa_t, w_t, x_tp1)
        else:
            state_t = (states, kappa_t, w_t)

        return state_t, state_t


def AttentionRNN(
    type, name,
    ctx, ctx_mask,
    input_dim, hidden_dim, ctx_dim,
    inputs=None, state0=None, mask=None,
    n_layers=1, position_gap=1.,
    return_cell_state=False,
    closed_loop=False,
    seq_len=1000
):
    size = 2*hidden_dim if type == 'LSTM' else hidden_dim
    if closed_loop:
        batch_size = tf.shape(ctx)[0]
    else:
        batch_size, seq_len, _ = tf.unstack(tf.shape(inputs))

    if state0 is None:
        h0 = tf.tile(lib.param(
            name+'.h0',
            weight_initializer('Constant', (1, n_layers*size), val=0.)
            ), [batch_size, 1])
        w0 = tf.tile(lib.param(
            name+'.w0',
            weight_initializer('Constant', (1, ctx_dim), val=0.)
            ), [batch_size, 1])
        k0 = tf.tile(lib.param(
            name+'.k0',
            weight_initializer('Constant', (1, 1), val=0.)
            ), [batch_size, 1])
        if closed_loop:
            x0 = tf.zeros((batch_size, 34), dtype=tf.float32)
            state0 = (h0, k0, w0, x0)
            inputs = tf.zeros((batch_size, seq_len, 20), dtype=tf.float32)
        else:
            state0 = (h0, k0, w0)

    if mask is None:
        sequence_length = tf.tile(tf.expand_dims(seq_len, 0), [batch_size])
    else:
        sequence_length = tf.reduce_sum(mask, axis=-1)

    states, _ = tf.nn.dynamic_rnn(
        AttentionRNNCell(type, name, ctx, ctx_mask,
                         input_dim, hidden_dim, ctx_dim, n_layers, position_gap, closed_loop),
        inputs,
        sequence_length=sequence_length,
        initial_state=state0
    )

    return states
