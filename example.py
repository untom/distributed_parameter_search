"""
Simple example of how to use distributed_parameter_search.py.

This performs a hyperparameter search of machine learning algorithms on MNIST.
To do this, simply start the script:

    python3 example.py

you can add additional processes to help in the search by starting additional processes
and passing them the --client flag:

    python3 example.py --client
"""

import pandas as pd
from sklearn import datasets, ensemble, metrics, model_selection
from parametersearch import  ParameterSearch, define_search_grid

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--host", type=str, help='host address', default="localhost")
parser.add_argument("--port", type=int, help='host port', default="7532")
parser.add_argument("--client", help="run as client process", action="store_true")
args = parser.parse_args()

# load data & split into train, validation & test
#mnist = datasets.fetch_openml('mnist_784', version=1, cache=True, data_home="/tmp")  # this is slower than fetc_mldata
mnist = datasets.fetch_mldata('MNIST original', data_home="/tmp")
data = mnist.data.reshape(len(mnist.data), -1)
labels = mnist.target.reshape(-1)
x_tr, x_te, y_tr, y_te = model_selection.train_test_split(data, labels, random_state=42, test_size=0.4)
x_tr, x_va, y_tr, y_va = model_selection.train_test_split(x_tr, y_tr, random_state=42, test_size=0.4)


if not args.client:  # we are the host
    param_grid = [
        {'n_estimators': [10, 50], 'max_features': [14, 28, 56]},
    ]
    parameter_search = define_search_grid(param_grid, "results.csv")
    parameter_search.start_server(args.host, args.port, as_thread=True)
else:
    parameter_search = ParameterSearch(host=args.host, port=args.port)


for (job_id, hyperparams) in parameter_search:
    print("Working on job %d: %s" % (job_id, hyperparams), flush=True)
    #model = svm.SVC(**hyperparams)
    model = ensemble.RandomForestClassifier(**hyperparams)
    model.fit(x_tr, y_tr)
    p_va = model.predict(x_va)
    acc_va = metrics.accuracy_score(y_va, p_va)
    parameter_search.submit_result(job_id, acc_va)

# after we're done, the master process prints the result list:
if not args.client:
    r = pd.DataFrame(parameter_search.get_results(), columns=["ID", "params", "result"])
    print(r.sort_values("result", ascending=False))
