#!/usr/bin/env python3
import argparse
import sys

from omsignal.test import get_test_parser, test
from omsignal.train import get_train_parser, train

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='omsignal')
    subparsers = parser.add_subparsers(title="commands", dest="command")
    get_train_parser(subparsers)
    get_test_parser(subparsers)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help(sys.stderr)
        sys.exit(1)
    if args.command == 'train':
        train(args)
    elif args.command == 'test':
        test(args)
