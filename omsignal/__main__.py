#!/usr/bin/env python3
import argparse

from omsignal.train import get_train_parser
from omsignal.test import get_test_parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='omsignal')
    subparsers = parser.add_subparsers(title="commands", dest="command")
    get_train_parser(subparsers)
    get_test_parser(subparsers)

    args = parser.parse_args()
