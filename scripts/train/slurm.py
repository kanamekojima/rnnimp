import datetime
import os
import pathlib
import subprocess
import time


def system(command):
    subprocess.call(command, shell=True)


def system_with_output_lines(command):
    p = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True)
    stdout, stderr = p.communicate()
    stdout_lines = [
        line.rstrip() for line in stdout.decode('utf-8').splitlines()]
    stderr_lines = [
        line.rstrip() for line in stderr.decode('utf-8').splitlines()]
    return stdout_lines, stderr_lines


def mkdir(dirname):
    os.makedirs(pathlib.Path(dirname).resolve(), exist_ok=True)


def parse_memory_size(memory_size):
    if memory_size.endswith('GB'):
        return memory_size[:-1]
    if memory_size.endswith('TB'):
        return memory_size[:-1]
    return '1G'


def sbatch(
        slurm_command,
        slurm_dir,
        job_name,
        memory_usage,
        num_threads,
        previous_job_id_list=None,
        options=None):
    script_dir = os.path.join(slurm_dir, 'scripts')
    log_dir = os.path.join(slurm_dir, 'log')
    mkdir(script_dir)
    mkdir(log_dir)
    current_time = time.time()
    milliseconds = str(current_time).split('.')[1][:3]
    current_time = datetime.datetime.fromtimestamp(current_time)
    current_time = current_time.strftime('%Y%m%d_%H%M%S')
    current_time += '_' + milliseconds

    script_file = os.path.join(
        script_dir, 'sbatch_{:s}_{:s}.sh'.format(job_name, current_time))

    with open(script_file, 'wt') as fout:
        fout.write('#!/bin/bash\n')
        fout.write('#$ -S /bin/bash\n\n')
        fout.write(slurm_command)
        fout.write('\n')
    system('chmod 700 ' + script_file)

    if options is None:
        options = []
    log_file_prefix = os.path.join(
        log_dir, '{:s}_{:s}'.format(job_name, current_time))
    options.extend([
        '--export=ALL',
        '--chdir=' + os.getcwd(),
        '-o {:s}.o'.format(log_file_prefix),
        '-e {:s}.e'.format(log_file_prefix),
        '--job-name ' + job_name,
        '--mem=' + memory_usage,
    ])
    if previous_job_id_list is None:
        previous_job_id_list = []
    if len(previous_job_id_list) > 0:
        options.append('--depend=' + ','.join([
            'afterany:' + job_id for job_id in previous_job_id_list]))
    if num_threads > 1:
        options.append('--cpus-per-task={:d}'.format(num_threads))
    command = 'sbatch {:s} {:s}'.format(' '.join(options), ' ' + script_file)
    print('command: ' + command)
    stdout_lines, stderr_lines = system_with_output_lines(command)
    for line in stdout_lines:
        print(line)
    for line in stderr_lines:
        print(line)
    if len(stdout_lines) != 1 or len(stderr_lines) != 0:
        return None
    job_id = stdout_lines[0].split(' ')[-1]
    return job_id
