#!/usr/bin/python

from setuptools import setup

with open('README.rst') as readme_file:
    readme = readme_file.read()

setup(name='coconut_learning',
      version='1.0',
      description='Examples for learning Coconut',
      author='Paul Jones',
      url='https://github.com/pwjones89/coconut_learning',
      license='MIT',
      test_suite='tests',
      )
