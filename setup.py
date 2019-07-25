# -*- coding: utf-8 -*-

from ast import parse
from distutils.sysconfig import get_python_lib
from functools import partial
from os import path, listdir
from platform import python_version_tuple

from setuptools import setup, find_packages

if python_version_tuple()[0] == '3':
    imap = map
    ifilter = filter
else:
    from itertools import imap, ifilter

if __name__ == '__main__':
    package_name = 'mnet_deep_cdr'

    with open(path.join(package_name, '__init__.py')) as f:
        __author__, __version__ = imap(
            lambda buf: next(imap(lambda e: e.value.s, parse(buf).body)),
            ifilter(lambda line: line.startswith('__version__') or line.startswith('__author__'), f)
        )

    to_funcs = lambda *paths: (partial(path.join, path.dirname(__file__), package_name, *paths),
                               partial(path.join, get_python_lib(prefix=''), package_name, *paths))
    _data_join, _data_install_dir = to_funcs('_data')
    ipynb_checkpoints_join, ipynb_checkpoints_install_dir = to_funcs('.ipynb_checkpoints')
    deep_model_join, deep_model_install_dir = to_funcs('deep_model')
    # mat_scr_join, mat_scr_install_dir = to_funcs('mat_scr')
    REFUGE_result_join, REFUGE_result_install_dir = to_funcs('REFUGE_result')
    result_join, result_install_dir = to_funcs('result')
    test_img_join, test_img_install_dir = to_funcs('test_img')

    setup(
        name=package_name,
        author=__author__,
        version=__version__,
        install_requires=['pyyaml'],
        test_suite=package_name + '.tests',
        packages=find_packages(),
        package_dir={package_name: package_name},
        data_files=[
            (_data_install_dir(), list(imap(_data_join, listdir(_data_join())))),
            (ipynb_checkpoints_install_dir(), list(imap(ipynb_checkpoints_join, listdir(ipynb_checkpoints_join())))),
            (deep_model_install_dir(), list(imap(deep_model_join, listdir(deep_model_join())))),
            # (mat_scr_install_dir(), list(imap(mat_scr_join, listdir(mat_scr_join())))),
            (REFUGE_result_install_dir(), list(imap(REFUGE_result_join, listdir(REFUGE_result_join())))),
            (result_install_dir(), list(imap(result_join, listdir(result_join())))),
            (test_img_install_dir(), list(imap(test_img_join, listdir(test_img_join()))))
        ]
    )
