import test_basics
import test_he_operations
import test_matmul
import test_conv2d
import unittest
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbosity", type=int, default=1)
    args = parser.parse_args()

    runner = unittest.TextTestRunner(verbosity=args.verbosity)

    print("Basics")
    runner.run(test_basics.get_suite())

    print("HeOperations")
    runner.run(test_he_operations.get_suite())

    print("Matmul")
    runner.run(test_matmul.get_suite())

    print("Conv2d")
    runner.run(test_conv2d.get_suite())


