from argparse import ArgumentParser
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import sys

import tensorflow.compat.v1 as tf
if tf.__version__.startswith('2'):
    tf.disable_v2_behavior()
from tensorflow.python.tools import optimize_for_inference_lib

from common_utils import mkdir


def main():
    description = 'convert meta graph to pb'
    parser = ArgumentParser(description=description, add_help=False)
    parser.add_argument('--input-node-names', type=str, required=True,
                        dest='input_node_names', help='input node names')
    parser.add_argument('--input-node-dtypes', type=str, required=True,
                        dest='input_node_dtypes', help='input node dtypes')
    parser.add_argument('--output-node-names', type=str, required=True,
                        dest='output_node_names', help='output node names')
    parser.add_argument('--checkpoint', type=str, required=True,
                        dest='checkpoint_file', help='checkpoint file')
    parser.add_argument('--output-file', type=str, required=True,
                        dest='output_file', help='output file')
    args = parser.parse_args()

    input_node_names = args.input_node_names.split(',')
    output_node_names = args.output_node_names.split(',')
    input_node_dtypes = []
    for input_node_dtype in args.input_node_dtypes.split(','):
        if input_node_dtype == 'float64':
            input_node_dtypes.append(tf.float64.as_datatype_enum)
        elif input_node_dtype == 'float32':
            input_node_dtypes.append(tf.float32.as_datatype_enum)
        elif input_node_dtype == 'int32':
            input_node_dtypes.append(tf.int32.as_datatype_enum)
        elif input_node_dtype == 'int64':
            input_node_dtypes.append(tf.int64.as_datatype_enum)
        else:
            print('Unsupported dtype: ' + dtype, file=sys.stderr)
            sys.exit(-1)
    tf_config = tf.ConfigProto(
        device_count={'GPU': 0},
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1,
    )
    with tf.Session(config=tf_config) as sess:
        tf.train.import_meta_graph(args.checkpoint_file + '.meta').restore(
            sess, args.checkpoint_file)
        input_node_names = ['inputs']
        output_node_names = ['predictions']
        graph_def = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, output_node_names)
    graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def=graph_def,
        input_node_names=input_node_names,
        output_node_names=output_node_names,
        placeholder_type_enum=input_node_dtypes)
    if tf.__version__.startswith('1'):
        transforms = [
            'remove_attribute(attribute_name=_class)',
            'remove_nodes(op=Identity, op=CheckNumerics)',
            'strip_unused_nodes',
            'fold_constants(ignore_errors=true)',
            'fold_batch_norms',
            'fold_old_batch_norms',
            'merge_duplicate_nodes'
        ]
        from tensorflow.tools.graph_transforms import TransformGraph
        graph_def = TransformGraph(
            graph_def, input_node_names, output_node_names, transforms)
    mkdir(os.path.dirname(args.output_file))
    tf.train.write_graph(
        graph_def, os.path.dirname(args.output_file),
        os.path.basename(args.output_file), as_text=False)


if __name__ == '__main__':
    main()
