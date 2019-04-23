#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import unittest
import socket
import random
import time
from multiprocessing import Process
from threading import Thread
from parametersearch import *

class TestServer(unittest.TestCase):
    def setUp(self):
        self._host = "localhost"
        self._port = 8532 + random.randint(0, 1000)
        self.parameter_search = ParameterSearch(host=self._host, port=self._port)
        self.servermanager = ParameterSearch()
        for learning_rate in [1e-2, 1e-3]:
            self.servermanager.add_parameter_setting({"learning_rate": learning_rate})

        def _run_server(m, host='localhost', port=5732):
            m.start_server(host, port)

        self._serverprocess = Thread(target=_run_server, args=(self.servermanager, self._host, self._port))
        self._serverprocess.start()
        self._serverprocess.join(1)  # wait for process to come up

    def tearDown(self):
        self.servermanager.is_serving = False
        self._serverprocess.join()  # wait for process to die

    def test_read_remote_job(self):
        job_id, hparams = self.parameter_search.get_next_setting()
        self.assertIsNotNone(job_id)
        self.assertIsNotNone(hparams)
        self.assertIn("learning_rate", hparams)

    def test_request_too_many_jobs(self):
        for _ in range(2):
            job_id, hparams = self.parameter_search.get_next_setting()
            self.assertIsNotNone(job_id)
            self.assertIsNotNone(hparams)
            self.assertIn("learning_rate", hparams)

        job_id, hparams = self.parameter_search.get_next_setting()
        self.assertIsNone(job_id)
        self.assertIsNone(hparams)

    def test_complete_job(self):
        job_id, hparams = self.parameter_search.get_next_setting()
        self.parameter_search.submit_result(job_id, "42")

    def test_complete_job_utf8result(self):
        job_id, hparams = self.parameter_search.get_next_setting()
        self.parameter_search.submit_result(job_id, "resäult")

    def test_complete_job_invalid_id(self):
        job_id, hparams = self.parameter_search.get_next_setting()
        with self.assertRaises(RuntimeError):
            self.parameter_search.submit_result(job_id + 43, "0")

    def test_auto_shutdown(self):
        for job_id, hparams in self.parameter_search:
            self.parameter_search.submit_result(job_id, "42")
        time.sleep(5)
        self.assertFalse(self._serverprocess.is_alive())


if __name__ == '__main__':
    unittest.main()
