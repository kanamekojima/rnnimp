import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import tensorflow.compat.v1 as tf
if tf.__version__.startswith('2'):
    tf.disable_v2_behavior()

from common_utils import mkdir
from rnn_model import setup_model, setup_hybrid_model


class TrainConfig:

    def __init__(
            self,
            output_prefix,
            batch_size,
            max_iteration_count,
            use_rsquare,
            num_threads,
            ):
        self.batch_size = batch_size
        self.max_iteration_count = max_iteration_count
        self.use_rsquare = use_rsquare
        self.weights_for_cross_entropy = None
        self.init_learning_rate = 1e-2
        self.min_learning_rate = 1e-6
        self.init_lr_update_check_step = 5000
        self.min_lr_update_check_step = 100
        self.init_clip_norm = 10.0
        self.clip_norm_update_step = 50
        self.init_checkpoint_file = None
        self.output_prefix = output_prefix
        self.num_threads = num_threads
        self.display_step = 10
        self.train_data_initialization_interval = 5


class TrainDataInitializer:

    def __init__(self, initializer, data_feeder, inputs_ph, outputs_ph):
        self.initializer = initializer
        self.data_feeder = data_feeder
        self.inputs_ph = inputs_ph
        self.outputs_ph = outputs_ph

    def initialize(self, sess):
        inputs, outputs = self.data_feeder.get_next()
        feed_dict = {self.inputs_ph: inputs, self.outputs_ph: outputs}
        sess.run(self.initializer, feed_dict=feed_dict)


class GlobalNormHandler:

    def __init__(self, buffer_size, init_value):
        self._buffer_size = buffer_size
        self._global_norms = np.full(buffer_size, init_value)
        self._pointer = 0

    def add(self, global_norm):
        if np.isnan(global_norm) or np.isinf(global_norm):
            return
        self._global_norms[self._pointer] = global_norm
        self._pointer += 1
        if self._pointer >= self._buffer_size:
            self._pointer = 0

    def get_median(self):
        median_value = np.median(self._global_norms)
        return median_value


class RSquareCalc:

    def __init__(self, Y):
        assert Y.ndim == 3
        Y = Y.transpose(1, 0, 2)
        self._allele_counts = Y[:, :, 1].reshape(
            Y.shape[0], -1, 2).sum(2)
        self._stds = np.std(self._allele_counts, axis=1)
        self._allele_dosages = np.zeros(
            self._allele_counts.shape[1], dtype=np.float64)

    def _get_rsquare(self, allele_counts, std, allele_dosages):
        if std == 0:
            return None
        if np.var(allele_dosages) == 0:
            return 0
        r = np.corrcoef(allele_dosages, allele_counts)[0, 1]
        return r * r

    def get_mean_rsquare(self, predictions_list):

        def get_allele_dosage(h0, h1):
            return h0[1] * h1[0] + h0[0] * h1[1] + 2.0 * h0[1] * h1[1]

        count = 0
        mean_rsquare = 0
        for i, predictions in enumerate(predictions_list):
            for j in range(len(self._allele_dosages)):
                h0, h1 = predictions[2 * j], predictions[2 * j + 1]
                self._allele_dosages[j] = get_allele_dosage(h0, h1)
            rsquare = self._get_rsquare(
                self._allele_counts[i], self._stds[i], self._allele_dosages)
            if rsquare is not None:
                mean_rsquare += rsquare
                count += 1
        if count == 0:
            return None
        return mean_rsquare / float(count)

    def get_site_counts(self):
        site_count = len(self._stds)
        active_site_count = np.count_nonzero(self._stds)
        return site_count, active_site_count


def train(
        train_data_feeder,
        validation_inputs,
        validation_outputs,
        features,
        config,
        train_config,
        ):
    rsquare_calc = RSquareCalc(validation_outputs)
    with tf.Graph().as_default():
        training_batch_size = train_config.batch_size
        tf_dataset, inputs_ph, outputs_ph = get_tf_train_dataset(
            train_data_feeder, training_batch_size)
        iterator = tf.data.make_initializable_iterator(tf_dataset)
        next_inputs, next_outputs = iterator.get_next()
        next_inputs = tf.ensure_shape(
            next_inputs,
            [training_batch_size, *train_data_feeder.input_shape])
        next_outputs = tf.ensure_shape(
            next_outputs,
            [training_batch_size, *train_data_feeder.output_shape])
        train_data_initializer = TrainDataInitializer(
            iterator.initializer, train_data_feeder, inputs_ph, outputs_ph)
        logits = setup_model(next_inputs, features, config)

        tf_dataset = get_tf_validation_dataset(
            validation_inputs, validation_outputs)
        iterator = tf.data.make_one_shot_iterator(tf_dataset)
        next_val_inputs, next_val_outputs = iterator.get_next()
        next_val_inputs = tf.ensure_shape(
            next_val_inputs, validation_inputs.shape)
        next_val_outputs = tf.ensure_shape(
            next_val_outputs, validation_outputs.shape)

        val_logits = setup_model(next_val_inputs, features, config, reuse=True)

        variables_to_save = get_variables_to_save(config.scope)
        variables_to_train = []
        for variable in tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope=config.scope):
            if variable.name.startswith(config.scope + '/features'):
                continue
            variables_to_train.append(variable)
        train_op_dict = get_train_op_dict(
            next_outputs, logits, config, train_config)
        validation_op_dict = get_validation_op_dict(
            next_val_outputs, val_logits, config, train_config)
        tf_config = tf.ConfigProto(
            device_count={'GPU': 0},
            intra_op_parallelism_threads=train_config.num_threads,
            inter_op_parallelism_threads=train_config.num_threads,
        )
        with tf.Session(config=tf_config) as sess:
            sess.run(tf.global_variables_initializer())
            min_v_loss, max_v_rsquare = train_parameters(
                sess, train_op_dict, validation_op_dict, variables_to_train,
                variables_to_save, rsquare_calc, train_config,
                train_data_initializer)
            write_stats(
                min_v_loss, max_v_rsquare, rsquare_calc,
                train_config.output_prefix + '.stats')


def train_hybrid_model(
        train_data_feeder,
        validation_inputs,
        validation_outputs,
        config1,
        config2,
        train_config,
        scope='hybrid',
        ):
    rsquare_calc = RSquareCalc(validation_outputs)
    with tf.Graph().as_default():
        training_batch_size = train_config.batch_size
        tf_dataset, inputs_ph, outputs_ph = get_tf_train_dataset(
            train_data_feeder, training_batch_size)
        iterator = tf.data.make_initializable_iterator(tf_dataset)
        next_inputs, next_outputs = iterator.get_next()
        next_inputs = tf.ensure_shape(
            next_inputs,
            [training_batch_size, *train_data_feeder.input_shape])
        next_outputs = tf.ensure_shape(
            next_outputs,
            [training_batch_size, *train_data_feeder.output_shape])
        train_data_initializer = TrainDataInitializer(
            iterator.initializer, train_data_feeder, inputs_ph, outputs_ph)
        logits = setup_hybrid_model(next_inputs, config1, config2, scope)

        tf_dataset = get_tf_validation_dataset(
            validation_inputs, validation_outputs)
        iterator = tf.data.make_one_shot_iterator(tf_dataset)
        next_val_inputs, next_val_outputs = iterator.get_next()
        next_val_inputs = tf.ensure_shape(
            next_val_inputs, validation_inputs.shape)
        next_val_outputs = tf.ensure_shape(
            next_val_outputs, validation_outputs.shape)
        val_logits = setup_hybrid_model(
            next_val_inputs, config1, config2, scope, reuse=True)

        variables_to_restore1 = get_variables_to_save(config1.scope)
        variables_to_restore2 = get_variables_to_save(config2.scope)
        variables_to_train = tf.trainable_variables(scope)
        variables_to_save = (
            variables_to_restore1 + variables_to_restore2 + variables_to_train)

        train_op_dict = get_train_op_dict(
            next_outputs, logits, config1, train_config,
            variables_to_train)
        validation_op_dict = get_validation_op_dict(
            next_val_outputs, val_logits, config1, train_config)
        tf_config = tf.ConfigProto(
            device_count={'GPU': 0},
            intra_op_parallelism_threads=train_config.num_threads,
            inter_op_parallelism_threads=train_config.num_threads,
        )
        with tf.Session(config=tf_config) as sess:
            sess.run(tf.global_variables_initializer())
            tf.train.Saver(variables_to_restore1).restore(
                sess, train_config.init_checkpoint_file1)
            tf.train.Saver(variables_to_restore2).restore(
                sess, train_config.init_checkpoint_file2)
            min_v_loss, max_v_rsquare = train_parameters(
                sess, train_op_dict, validation_op_dict, variables_to_train,
                variables_to_save, rsquare_calc, train_config,
                train_data_initializer)
            write_stats(
                min_v_loss, max_v_rsquare, rsquare_calc,
                train_config.output_prefix + '.stats')

    with tf.Graph().as_default() as graph:
        inputs_ph = tf.placeholder(
            tf.float32, shape=[None, config1.num_inputs, config1.input_dim],
            name='inputs')
        logits = setup_hybrid_model(inputs_ph, config1, config2, scope)
        predictions = tf.reshape(
            tf.nn.softmax(tf.concat(logits, 0)),
            shape=[config1.num_outputs, -1, config1.num_classes],
            name='predictions')
        variables_to_save = []
        variables_to_save.extend(get_variables_to_save(config1.scope))
        variables_to_save.extend(get_variables_to_save(config2.scope))
        variables_to_save.extend(tf.trainable_variables(scope))
        tf_config = tf.ConfigProto(
            device_count={'GPU': 0},
            intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1,
        )
        saver = tf.train.Saver(variables_to_save, max_to_keep=1)
        with tf.Session(config=tf_config) as sess:
            saver.restore(sess, train_config.output_prefix)
            saver.save(
                sess, train_config.output_prefix, write_meta_graph=True,
                write_state=False)


def get_tf_train_dataset(data_feeder, batch_size=None):
    inputs_ph = tf.placeholder(
        tf.float32, shape=[None, *data_feeder.input_shape],
        name='inputs_ph')
    outputs_ph = tf.placeholder(
        tf.float32, shape=[None, *data_feeder.output_shape],
        name='outputs_ph')
    tf_dataset = tf.data.Dataset.zip((
        tf.data.Dataset.from_tensor_slices(inputs_ph),
        tf.data.Dataset.from_tensor_slices(outputs_ph)))
    tf_dataset = tf_dataset.shuffle(data_feeder.sample_size)
    tf_dataset = tf_dataset.repeat()
    if batch_size is None:
        batch_size = data_feeder.sample_size
    tf_dataset = tf_dataset.batch(batch_size)
    return tf_dataset, inputs_ph, outputs_ph


def get_tf_validation_dataset(inputs, outputs, batch_size=None):
    tf_dataset = tf.data.Dataset.zip((
        tf.data.Dataset.from_tensor_slices(inputs),
        tf.data.Dataset.from_tensor_slices(outputs)))
    tf_dataset = tf_dataset.repeat()
    tf_dataset = tf_dataset.batch(len(inputs))
    return tf_dataset


def get_variables_to_save(scope):
    variable_set_to_save = set(tf.trainable_variables(scope=scope))
    for variable in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        for subscope_prefix in ['features', 'rnn_fw/bn_', 'rnn_bw/bn_']:
            if variable.name.startswith(scope + '/' + subscope_prefix):
                variable_set_to_save.add(variable)
                break
    return list(variable_set_to_save)


def get_train_op_dict(
        outputs,
        logits,
        config,
        train_config,
        variables_to_train=None):
    outputs = tf.unstack(outputs, config.num_outputs, 1)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits, labels=outputs)
    if train_config.weights_for_cross_entropy is not None:
        weights = tf.expand_dims(
            train_config.weights_for_cross_entropy, axis=1)
        cross_entropy = tf.multiply(cross_entropy, weights)
    loss = tf.reduce_mean(cross_entropy)

    lr = tf.get_variable('lr', initializer=1.0, trainable=False)
    new_lr = tf.placeholder(tf.float32, shape=[], name='new_lr')
    update_lr = tf.assign(lr, new_lr)

    clip_norm = tf.get_variable('clip_norm', initializer=1.0, trainable=False)
    new_clip_norm = tf.placeholder(tf.float32, shape=[], name='new_clip_norm')
    update_clip_norm = tf.assign(clip_norm, new_clip_norm)

    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    gradients, variables = zip(*optimizer.compute_gradients(
        loss, var_list=variables_to_train))
    global_norm = tf.global_norm(gradients)
    train_op = optimizer.apply_gradients(zip(gradients, variables))
    train_op = tf.group([train_op, tf.get_collection(tf.GraphKeys.UPDATE_OPS)])
    gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=clip_norm)
    is_invalid_gradient = tf.logical_or(
        tf.is_nan(global_norm), tf.is_inf(global_norm))
    train_op = tf.cond(
        pred=is_invalid_gradient,
        true_fn=lambda: False, false_fn=lambda: train_op)

    predictions = tf.nn.softmax(tf.concat(logits, 0))
    correct_flags = tf.equal(
        tf.argmax(predictions, 1), tf.argmax(tf.concat(outputs, 0), 1))
    accuracy = tf.reduce_mean(tf.cast(correct_flags, tf.float32))
    return {
        'loss': loss,
        'accuracy': accuracy,
        'train': train_op,
        'global norm': global_norm,
        'update lr': update_lr,
        'new lr': new_lr,
        'update clip norm': update_clip_norm,
        'new clip norm': new_clip_norm,
        'global norm': global_norm,
    }


def get_validation_op_dict(outputs, logits, config, train_config):
    outputs = tf.unstack(outputs, config.num_outputs, 1)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits, labels=outputs)
    if train_config.weights_for_cross_entropy is not None:
        weights = tf.expand_dims(
            train_config.weights_for_cross_entropy, axis=1)
        cross_entropy = tf.multiply(cross_entropy, weights)
    loss = tf.reduce_mean(cross_entropy)
    predictions = tf.nn.softmax(tf.concat(logits, 0))
    correct_flags = tf.equal(
        tf.argmax(predictions, 1), tf.argmax(tf.concat(outputs, 0), 1))
    accuracy = tf.reduce_mean(tf.cast(correct_flags, tf.float32))
    predictions = tf.reshape(
        predictions, shape=[config.num_outputs, -1, config.num_classes])
    return {
        'loss': loss,
        'accuracy': accuracy,
        'predictions': predictions,
    }


def train_parameters(
        sess,
        train_op_dict,
        validation_op_dict,
        variables_to_train,
        variables_to_save,
        rsquare_calc,
        train_config,
        train_data_initializer,
        ):
    min_loss = float('inf')
    min_v_loss = float('inf')
    max_v_rsquare = 0
    last_v_update_step = 0

    global_norm_handler = GlobalNormHandler(
        200, train_config.init_clip_norm / 5)

    mkdir(os.path.dirname(train_config.output_prefix))
    saver = tf.train.Saver(variables_to_train, max_to_keep=1)
    if train_config.init_checkpoint_file is not None:
        saver.restore(sess, train_config.init_checkpoint_file)

    current_lr = train_config.init_learning_rate
    sess.run(
        train_op_dict['update lr'],
        feed_dict={train_op_dict['new lr']: current_lr})
    current_clip_norm = train_config.init_clip_norm
    sess.run(
        train_op_dict['update clip norm'],
        feed_dict={train_op_dict['new clip norm']: current_clip_norm})

    lr_update_check_step = train_config.init_lr_update_check_step
    validation_op_list = [
        validation_op_dict['loss'],
        validation_op_dict['accuracy'],
        validation_op_dict['predictions'],
    ]
    train_data_initializer.initialize(sess)
    for step in range(1, train_config.max_iteration_count + 1):
        if step % train_config.display_step == 0 or step == 1:
            _, global_norm, loss, accuracy = sess.run([
                train_op_dict['train'],
                train_op_dict['global norm'],
                train_op_dict['loss'],
                train_op_dict['accuracy'],
            ])
        else:
            _, global_norm = sess.run([
                train_op_dict['train'],
                train_op_dict['global norm'],
            ])
        if step % train_config.clip_norm_update_step == 0 or step < 1000:
            global_norm_handler.add(global_norm)
            current_clip_norm = global_norm_handler.get_median() * 5
            sess.run(
                train_op_dict['update clip norm'],
                feed_dict={train_op_dict['new clip norm']: current_clip_norm})
        if step % train_config.display_step == 0 or step == 1:
            print_log_line(step, loss, accuracy, current_lr)
            min_loss = min(min_loss, loss)
            v_loss, v_accuracy, v_predictions = sess.run(validation_op_list)
            v_update_flag = False
            v_rsquare = rsquare_calc.get_mean_rsquare(v_predictions)
            if train_config.use_rsquare and v_rsquare is not None:
                if v_rsquare > max_v_rsquare:
                    max_v_rsquare = v_rsquare
                    v_update_flag = True
            else:
                if v_loss < min_v_loss:
                    v_update_flag = True
            min_v_loss = min(min_v_loss, v_loss)
            if v_update_flag:
                last_v_update_step = step
                print_validation_log_line(step, v_loss, v_accuracy, v_rsquare)
                saver.save(
                    sess, train_config.output_prefix,
                    write_meta_graph=False, write_state=False)
                save_point_loss = loss
            if step - last_v_update_step > lr_update_check_step:
                lr_update_check_step = max(
                    lr_update_check_step // 2,
                    train_config.min_lr_update_check_step)
                current_lr *= 0.5
                if current_lr < train_config.min_learning_rate:
                    break
                saver.restore(sess, train_config.output_prefix)
                sess.run(
                    train_op_dict['update lr'],
                    feed_dict={train_op_dict['new lr']: current_lr})
                min_loss = save_point_loss
                last_v_update_step = step
        if step % train_config.train_data_initialization_interval == 0:
            train_data_initializer.initialize(sess)
    saver.restore(sess, train_config.output_prefix)
    tf.train.Saver(variables_to_save, max_to_keep=1).save(
        sess, train_config.output_prefix, write_meta_graph=False,
        write_state=False)
    print('Training finished', flush=True)
    return min_v_loss, max_v_rsquare


def print_log_line(step, loss, accuracy, learning_rate):
    print(' '.join([
        'Step {:06d}:'.format(step),
        'Minibatch Loss={:.5f},'.format(loss),
        'Training Accuracy={:.5f},'.format(accuracy),
        'Learning Rate={:.2E}'.format(learning_rate),
    ]), flush=True)


def print_validation_log_line(step, loss, accuracy, rsquare):
    print(' '.join([
        'Step {:06d}:'.format(step),
        'Validation Loss={:.5f},'.format(loss),
        'Validation Accuracy={:.5f},'.format(accuracy),
        'R2={:.5f}'.format(rsquare) if rsquare is not None else 'R2=NA',
    ]), flush=True)


def write_stats(min_v_loss, max_v_rsquare, rsquare_calc, output_file):
    site_count, active_site_count = rsquare_calc.get_site_counts()
    mkdir(os.path.dirname(output_file))
    with open(output_file, 'wt') as fout:
        fout.write('min validation loss: {:f}\n'.format(
            min_v_loss * site_count))
        fout.write('max validation R2: {:f}\n'.format(
            max_v_rsquare * active_site_count))
        fout.write('site count: {:d}\n'.format(site_count))
        fout.write('active site count: {:d}\n'.format(active_site_count))
