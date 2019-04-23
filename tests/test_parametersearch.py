import unittest
import socket
import random
import time
from multiprocessing import Process
from threading import Thread
from parametersearch import *


class TestServer(unittest.TestCase):
    def setUp(self):
        self.parameter_search = ParameterSearch()
        for learning_rate in [1e-2, 1e-3]:
            self.parameter_search.add_parameter_setting({"learning_rate": learning_rate})

    def test_iteration(self):
        n = 0
        for _, _ in self.parameter_search:
            n += 1
        self.assertEqual(n, 2)


if __name__ == '__main__':
    unittest.main()
