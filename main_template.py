''' general import stuff '''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function   # print function in python 3.x
import xlrd # read spreadsheet
import numpy as np
import sys
import os
import argparse
import csv
from random import randint
import pickle


''' variable declaration '''
FLAGS = None



''' main function declaration '''
def main(arg):  # arg should be variable (var), data shape/dimension (shape), other object (obj), and etc...

    pass


''' main loop '''
if __name__ == '__main__':

    ''' example '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                      default=False,
                      help='If true, uses fake data for unit testing.')
    parser.add_argument('--max_steps', type=int, default=1000,
                      help='Number of steps to run trainer.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.9,
                      help='Keep probability for training dropout.')
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
    parser.add_argument('--log_dir', type=str, default='/tmp/tensorflow/mnist/logs/mnist_with_summaries',
                      help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()

    # tensorflow run
    #tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)