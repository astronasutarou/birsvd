#!/usr/bin/env python
# -*- coding: utf-8 -*-
from glob import glob
from setuptools import setup
from distutils.dist import Distribution
import os,sys,re


with open('README.md', 'r') as fd:
  version = '0.1'
  author = 'Ryou Ohsawa'
  email = 'ohsawa@ioa.s.u-tokyo.ac.jp'
  description = ''
  long_description = fd.read()
  license = 'MIT'

classifiers = [
  'Development Status :: 3 - Alpha',
  'Environment :: Console',
  'Intended Audience :: Science/Research',
  'License :: OSI Approved :: MIT License',
  'Operating System :: POSIX :: Linux',
  'Programming Language :: Python :: 3',
  'Programming Language :: Python :: 3.7',
  'Topic :: Scientific/Engineering :: Astronomy']


if __name__ == '__main__':
  try:
    import numpy
  except ImportError:
    raise SystemExit('NumPy is not available.')

  setup(
    name='birsvd',
    version=version,
    author=author,
    author_email=email,
    maintainer=author,
    maintainer_email=email,
    description=description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://bitbucket.org/ryou_ohsawa/birsvd/src/master/',
    license=license,
    classifiers=classifiers,
    install_requires=['numpy','scipy','scikit-learn'])
