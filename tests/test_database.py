
import csv
import json

import unittest
from parametersearch import Database


class TestDatabase(unittest.TestCase):

    def setUp(self):
        self.output_file = "/tmp/test.csv"
        self.database = Database(self.output_file)

    def test_add_job_and_complete(self):
        job = {"learning_rate": 0.1}
        job = self.database.add_job(job)
        self.database.complete_job(job.id, 42)

    def test_add_job_twice(self):
        job = {"learning_rate": 0.1}
        j1 = self.database.add_job(job)
        j2 = self.database.add_job(job)
        self.assertNotEqual(j1.id, j2.id, "two jobs obtained the same ID")

    def test_complete_job_twice(self):
        data = {"learning_rate": 0.1}
        j = self.database.add_job(data)
        self.database.complete_job(j.id, 42)
        with self.assertRaises(Exception):
             self.database.complete_job(j.id, 42)

    def test_unknown_job(self):
        id = 0
        with self.assertRaises(Exception):
            self.database.complete_job(id, 42)

    def test_njobs(self):
        self.assertEqual(self.database.n_jobs, 0)
        job = {"learning_rate": 0.1}
        id = self.database.add_job(job)
        self.assertEqual(self.database.n_jobs, 1)
        job = {"learning_rate": 0.1}
        id = self.database.add_job(job)
        self.assertEqual(self.database.n_jobs, 2)

    def test_get_jobs(self):
        job_data = {"learning_rate": 0.1}
        j1 = self.database.add_job(job_data)
        j2 = self.database.get_job(j1.id)
        self.assertEqual(j1, j2)
        with self.assertRaises(Exception):
            self.database.get_job(j1.id+1)

    def test_write_data(self):
        jobs = [
            {"learning_rate": 0.1, "kernel": "rbf"},
            {"learning_rate": 0.3, "kernel": "polynomial"},
            {"learning_rate": 0.2, "kernel": "rbf"}
        ]
        for j in jobs:
            self.database.add_job(j)
        self.database._save_results()

        results = []
        with open(self.output_file) as f:
            f.readline() # skip over header line
            reader = csv.reader(f)
            for r in reader:
                results.append(r)

        # we need to sort the job lists (CSV is not guaranteed to be in the same order as they've been added)
        a = sorted([tuple(j.items()) for j in jobs])
        b = sorted([tuple(json.loads(j[1]).items()) for j in results])
        self.assertEqual(a, b)




if __name__ == '__main__':
    unittest.main()