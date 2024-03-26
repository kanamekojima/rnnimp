from contextlib import contextmanager
import gzip
import os
import pathlib
import subprocess


def system(command):
    subprocess.call(command, shell=True)


def mkdir(dirname):
    os.makedirs(pathlib.Path(dirname).resolve(), exist_ok=True)


@contextmanager
def reading(filename):
    root, ext = os.path.splitext(filename)
    fp = gzip.open(filename, 'rt') if ext == '.gz' else open(filename, 'r')
    try:
        yield fp
    finally:
        fp.close()


@contextmanager
def writing(filename):
    root, ext = os.path.splitext(filename)
    fp = gzip.open(filename, 'wt') if ext == '.gz' else open(filename, 'w')
    try:
        yield fp
    finally:
        fp.close()
