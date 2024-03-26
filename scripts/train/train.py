from argparse import ArgumentParser
import os
import sys

from common_utils import system
import slurm


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    description = 'train'
    parser = ArgumentParser(description=description, add_help=False)
    parser.add_argument('--data-list', type=str, required=True,
                        dest='data_file_prefix_list_file',
                        help='list file of data file prefix')
    parser.add_argument('--output-prefix', type=str, required=True,
                        dest='output_prefix', help='output prefix')
    parser.add_argument('--rnn-cell-type', type=str, default='GRU',
                        dest='rnn_cell_type', help='RNN cell type')
    parser.add_argument('--num-units', type=int, default=40,
                        dest='num_units', help='number of units')
    parser.add_argument('--num-layers-higher', type=int, default=5,
                        dest='num_layers_higher',
                        help='number of layers for higher MAF')
    parser.add_argument('--num-layers-lower', type=int, default=5,
                        dest='num_layers_lower',
                        help='number of layers for lower MAF')
    parser.add_argument('--feature-size', type=int, default=10,
                        dest='feature_size', help='feature size')
    parser.add_argument('--gamma1', type=float, default=0.75,
                        dest='gamma1', help='gamma1')
    parser.add_argument('--gamma2', type=float, default=-0.75,
                        dest='gamma2', help='gamma2')
    parser.add_argument('--batch-size', type=int, default=500,
                        dest='batch_size', help='batch size')
    parser.add_argument('--max-iteration-count', type=int, default=100000,
                        dest='max_iteration_count', help='max iteration count')
    parser.add_argument('--validation-sample-size', type=int, default=100,
                        dest='validation_sample_size',
                        help='validation sample size')
    parser.add_argument('--num-threads', type=int, default=2,
                        dest='num_threads', help='num threads')
    parser.add_argument('--slurm', action='store_true', default=False,
                        dest='slurm_flag', help='use slurm')
    parser.add_argument('--job-name-prefix', type=str, default='train',
                        dest='job_name_prefix',
                        help='prefix for slurm job name')
    parser.add_argument('--memory-size', type=str, default='20GB',
                        dest='memory_size', help='memory size per thread')
    parser.add_argument('--python3-bin', type=str, default='python3',
                        dest='python3_bin', help='path to Python3 binary')
    args = parser.parse_args()

    data_file_prefix_list = []
    with open(args.data_file_prefix_list_file, 'rt') as fin:
        for line in fin:
            data_file_prefix = os.path.join(
                os.path.dirname(args.data_file_prefix_list_file),
                os.path.basename(line.strip()))
            data_file_prefix_list.append(data_file_prefix)

    for i, data_file_prefix in enumerate(data_file_prefix_list, start=1):
        previous_job_id_list = []
        output_dir = os.path.dirname(args.output_prefix)
        output_basename = '{:s}_{:d}'.format(
            os.path.basename(args.output_prefix), i)
        output_prefix = os.path.join(
            output_dir, output_basename, output_basename + '_higher_MAF')

        common_1st_step_options = ' '.join([
            ' --rnn-cell-type ' + args.rnn_cell_type,
            ' --num-units {:d}'.format(args.num_units),
            ' --feature-size {:d}'.format(args.feature_size),
        ])
        common_options = ' '.join([
            '--hap {:s}.hap.gz'.format(data_file_prefix),
            '--legend {:s}.legend.gz'.format(data_file_prefix),
            '--max-iteration-count {:d}'.format(args.max_iteration_count),
            '--batch-size {:d}'.format(args.batch_size),
            '--validation-sample-size {:d}'.format(
                args.validation_sample_size),
            '--num-threads {:d}'.format(args.num_threads),
        ])
        command = args.python3_bin
        command += ' ' + os.path.join(SCRIPT_DIR, 'train_1st_step.py')
        command += ' --scope higher_MAF'
        command += ' --gamma {:f}'.format(args.gamma1)
        command += ' --num-layers {:d}'.format(args.num_layers_higher)
        command += ' --output-prefix ' + output_prefix
        command += ' ' + common_1st_step_options
        command += ' ' + common_options
        if args.slurm_flag:
            job_name = '{:s}_{:d}_higher_MAF'.format(args.job_name_prefix, i)
            job_id = slurm.sbatch(
                command,
                os.path.join(output_dir, 'slurm', 'higher_MAF'),
                job_name,
                slurm.parse_memory_size(args.memory_size),
                args.num_threads)
            previous_job_id_list.append(job_id)
        else:
            system(command)

        output_prefix = os.path.join(
            output_dir, output_basename, output_basename + '_lower_MAF')
        command = args.python3_bin
        command += ' ' + os.path.join(SCRIPT_DIR, 'train_1st_step.py')
        command += ' --scope lower_MAF'
        command += ' --gamma {:f}'.format(args.gamma2)
        command += ' --num-layers {:d}'.format(args.num_layers_higher)
        command += ' --rsquare'
        command += ' --output-prefix ' + output_prefix
        command += ' ' + common_1st_step_options
        command += ' ' + common_options
        if args.slurm_flag:
            job_name = '{:s}_{:d}_lower_MAF'.format(args.job_name_prefix, i)
            job_id = slurm.sbatch(
                command,
                os.path.join(output_dir, 'slurm', 'lower_MAF'),
                job_name,
                slurm.parse_memory_size(args.memory_size),
                args.num_threads)
            previous_job_id_list.append(job_id)
        else:
            system(command)

        output_prefix = os.path.join(
            output_dir, output_basename, output_basename + '_hybrid')
        config_file1 = os.path.join(
            output_dir, output_basename,
            output_basename + '_higher_MAF_config.json')
        config_file2 = os.path.join(
            output_dir, output_basename,
            output_basename + '_lower_MAF_config.json')
        init_checkpoint_file1 = os.path.join(
            output_dir, output_basename, output_basename + '_higher_MAF')
        init_checkpoint_file2 = os.path.join(
            output_dir, output_basename, output_basename + '_lower_MAF')
        command = args.python3_bin
        command += ' ' + os.path.join(SCRIPT_DIR, 'train_2nd_step.py')
        command += ' --config1 ' + config_file1
        command += ' --config2 ' + config_file2
        command += ' --init-checkpoint1 ' + init_checkpoint_file1
        command += ' --init-checkpoint2 ' + init_checkpoint_file2
        command += ' --rsquare'
        command += ' --output-prefix ' + output_prefix
        command += ' ' + common_options

        if args.slurm_flag:
            job_name = '{:s}_{:d}_hybrid'.format(args.job_name_prefix, i)
            job_id = slurm.sbatch(
                command,
                os.path.join(output_dir, 'slurm', 'hybrid'),
                job_name,
                slurm.parse_memory_size(args.memory_size),
                args.num_threads,
                previous_job_id_list=previous_job_id_list)
            previous_job_id_list = [job_id]
        else:
            system(command)

        checkpoint_prefix = os.path.join(
            output_dir, output_basename, output_basename + '_hybrid')
        output_prefix = os.path.join(
            output_dir, 'models', '{:s}_{:d}').format(
                os.path.basename(args.output_prefix), i)

        command_list = []
        command = args.python3_bin
        command += ' ' + os.path.join(
            SCRIPT_DIR, 'convert_meta_graph_to_pb.py')
        command += ' --input-node-names inputs'
        command += ' --input-node-dtypes float32'
        command += ' --output-node-names predictions'
        command += ' --checkpoint ' + checkpoint_prefix
        command += ' --output-file {:s}.pb'.format(checkpoint_prefix)
        command_list.append(command)

        command = args.python3_bin
        command += ' -m tf2onnx.convert'
        command += ' --input {:s}.pb'.format(checkpoint_prefix)
        command += ' --inputs inputs:0'
        command += ' --output {:s}.onnx'.format(output_prefix)
        command += ' --outputs predictions:0'
        command_list.append(command)

        command = args.python3_bin
        command += ' -m onnxruntime.tools.convert_onnx_models_to_ort'
        command += ' {:s}.onnx'.format(output_prefix)
        command_list.append(command)

        command = 'cp {:s}.legend.gz {:s}.legend.gz'.format(
            checkpoint_prefix, output_prefix)
        command_list.append(command)

        if args.slurm_flag:
            job_name = '{:s}_{:d}_convert'.format(args.job_name_prefix, i)
            slurm.sbatch(
                '; '.join(command_list) + ';',
                os.path.join(output_dir, 'slurm', 'convert'),
                job_name,
                slurm.parse_memory_size(args.memory_size),
                args.num_threads,
                previous_job_id_list=previous_job_id_list)
        else:
            for command in command_list:
                system(command)


if __name__ == '__main__':
    main()
