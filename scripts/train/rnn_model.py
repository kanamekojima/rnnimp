import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow.compat.v1 as tf
if tf.__version__.startswith('2'):
    tf.disable_v2_behavior()


class Config:

    def __init__(
            self,
            input_dim,
            num_inputs,
            num_classes,
            output_points_fw,
            output_points_bw,
            rnn_cell_type,
            num_units,
            num_layers,
            feature_size,
            scope,
            ):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_inputs = num_inputs
        self.output_points_fw = output_points_fw
        self.output_points_bw = output_points_bw
        self.num_outputs = self.get_num_outputs()
        self.rnn_cell_type = rnn_cell_type
        self.num_units = num_units
        self.num_layers = num_layers
        self.feature_size = feature_size
        self.scope = scope

    def get_num_outputs(self):
        if self.output_points_fw is None:
            return 0
        return len(self.output_points_fw)

    def write(self, json_file):
        co_varnames = Config.__init__.__code__.co_varnames
        co_argcount = Config.__init__.__code__.co_argcount
        arg_set = set(co_varnames[:co_argcount])
        param_dict = {
            key: value
            for key, value in self.__dict__.items() if key in arg_set
        }
        with open(json_file, 'w') as fout:
            json.dump(param_dict, fout, indent=2)

    @staticmethod
    def load(json_file):
        with open(json_file, 'r') as fp:
            param_dict = json.load(fp)
        return Config(**param_dict)


def setup_model(inputs, features, config, reuse=False):
    with tf.variable_scope(config.scope):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False
        input_list = tf.unstack(inputs, config.num_inputs, 1)
        if features is None:
            features = tf.get_variable(
                'features',
                initializer=tf.zeros(
                    [config.num_inputs, 2, config.feature_size]),
                trainable=False)
        else:
            features = tf.get_variable(
                'features', initializer=features, trainable=False)
        rnn_inputs = []
        for i in range(config.num_inputs):
            if config.feature_size == 0:
                rnn_inputs.append(input_list[i])
            else:
                rnn_input = tf.matmul(input_list[i], features[i])
                rnn_inputs.append(rnn_input)

        rnn_fw_cells = [
            make_rnn_cell(config, i) for i in range(config.num_layers)
        ]
        rnn_bw_cells = [
            make_rnn_cell(config, i) for i in range(config.num_layers)
        ]

        fw_end = config.output_points_fw[-1]
        bw_start = config.output_points_bw[0]

        outputs_fw = [None] * config.num_inputs
        outputs_bw = [None] * config.num_inputs

        if fw_end is not None:
            inputs_fw = rnn_inputs[: fw_end + 1]
            outputs, _ = make_rnn(rnn_fw_cells, inputs_fw, 'rnn_fw')
            for t in range(fw_end + 1):
                outputs_fw[t] = outputs[t]

        if bw_start is not None:
            inputs_bw = [
                rnn_inputs[i]
                for i in range(config.num_inputs - 1, bw_start - 1, -1)
            ]
            outputs, _ = make_rnn(rnn_bw_cells, inputs_bw, 'rnn_bw')
            for i, t in enumerate(
                    range(config.num_inputs - 1, bw_start - 1, -1)):
                outputs_bw[t] = outputs[i]

        logits = []
        for i, (t_fw, t_bw) in enumerate(
                zip(config.output_points_fw, config.output_points_bw)):
            if t_fw is None:
                rnn_output = outputs_bw[t_bw]
                output_dim = config.num_units
            elif t_bw is None:
                rnn_output = outputs_fw[t_fw]
                output_dim = config.num_units
            else:
                rnn_output = tf.concat(
                    [outputs_fw[t_fw], outputs_bw[t_bw]], axis=1)
                output_dim = 2 * config.num_units

            num_classes = config.num_classes
            W = tf.get_variable(
                'weight_{:d}'.format(i + 1),
                initializer=tf.truncated_normal(
                    [output_dim, num_classes], stddev=1.0))
            b = tf.get_variable(
                'bias_{:d}'.format(i + 1),
                initializer=tf.truncated_normal([num_classes], stddev=0.1))

            logit = tf.matmul(rnn_output, W) + b
            logits.append(logit)
        return logits


def make_rnn_cell(config, layer_number):
    if config.rnn_cell_type == 'GRU':
        cell = tf.nn.rnn_cell.GRUCell(
            config.num_units, name='GRU_{:d}'.format(layer_number))
    elif config.rnn_cell_type == 'LSTM':
        cell = tf.nn.rnn_cell.LSTMCell(
            config.num_units, forget_bias=1.0,
            state_is_tuple=True, name='LSTM_{:d}'.format(layer_number))
    else:
        cell = tf.nn.rnn_cell.GRUCell(
            config.num_units, name='GRU_{:d}'.format(layer_number))
    return cell


def make_rnn(rnn_cells, input_tensors, scope):
    batch_size = tf.shape(input_tensors[0])[0]
    outputs = [None] * len(input_tensors)
    rnn_cell_inputs = [None] * len(input_tensors)
    states = []
    for i, rnn_cell in enumerate(rnn_cells):
        for t, input_tensor in enumerate(input_tensors):
            rnn_cell_inputs[t] = input_tensor
        state = rnn_cell.zero_state(batch_size, tf.float32)
        with tf.variable_scope(scope):
            for t, input_tensor in enumerate(input_tensors):
                if t > 0:
                    tf.get_variable_scope().reuse_variables()
                output, state = rnn_cell(rnn_cell_inputs[t], state)
                if i > 0:
                    output = output + input_tensor
                outputs[t] = output
            states.append(state)
        input_tensors = outputs
    return outputs, states


def setup_hybrid_model(inputs, config1, config2, scope, reuse=False):
    assert scope != config1.scope
    assert scope != config2.scope
    logits1 = setup_model(inputs, None, config1, reuse)
    logits2 = setup_model(inputs, None, config2, reuse)
    input_list = [
        tf.stop_gradient(tf.concat([logit1, logit2], 1))
        for logit1, logit2 in zip(logits1, logits2)
    ]
    logits = []
    with tf.variable_scope(scope):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False
        num_classes = config1.num_classes
        for i, input in enumerate(input_list, start=1):
            W = tf.get_variable(
                'weight_{:d}'.format(i), initializer=tf.truncated_normal(
                    [num_classes * 2, num_classes], stddev=1.0))
            b = tf.get_variable(
                'bias_{:d}'.format(i), initializer=tf.truncated_normal(
                    [num_classes], stddev=0.1))
            logits.append(tf.matmul(input, W) + b)
    return logits
