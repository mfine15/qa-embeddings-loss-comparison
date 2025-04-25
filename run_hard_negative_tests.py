#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to run all tests related to the hard negative issue.
"""

import os
import sys
import pytest
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run tests for hard negative issue.")
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--debug", action="store_true", help="Run debugging-focused tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    # If no test type is specified, run all tests
    if not (args.unit or args.integration or args.debug or args.all):
        args.all = True
    
    return args


def main():
    args = parse_args()
    
    # Base pytest options
    pytest_args = []
    if args.verbose:
        pytest_args.append("-v")
    
    # Collect test files to run
    test_files = []
    
    if args.all or args.unit:
        test_files.append("tests/unit/test_hard_negative_evaluation.py")
    
    if args.all or args.integration:
        test_files.append("tests/integration/test_hard_negative_pipeline.py")
    
    if args.all or args.debug:
        test_files.append("tests/unit/test_hard_negative_debugging.py")
    
    # Run tests
    print(f"Running tests: {', '.join(test_files)}")
    
    if not test_files:
        print("No tests selected to run.")
        return
    
    pytest_args.extend(test_files)
    
    return pytest.main(pytest_args)


if __name__ == "__main__":
    sys.exit(main() or 0)