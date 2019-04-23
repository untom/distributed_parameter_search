Distributed Parameter Search
============================

This library distributes parameter searches over many clients.
Its designed for simplicity and hackability first.

Simple Usage
------------

This is a single-file implementation, so just copy ```parametersearch.py``` to your source directory.
From there, do `from parametersearch import ParameterSearch` to use it.
`ParameterSearch` can be used to define all the different hyperparameter settings you want to try out.
As example, this piece of code defines two settings of different learning rates:

    ps = ParameterSearch(output_file="results.csv")  # results will be stored in results.csv
    ps.add_parameter_setting({"learning_rate": 1e-2})
    ps.add_parameter_setting({"learning_rate": 1e-3})

or you can use ```define_search_grid``` to set up a grid search:

    param_grid = [{
        'n_estimators': [20, 50],
        'max_features': [14, 28]
        }]
    ps = define_search_grid(param_grid, output_file="results.csv")


Then, you can iterate over the created ParameterSearch instance to process the different settings, and
use the ```submit_result``` method to report the results back to the ParameterSearch object:

    for (job_id, hyperparams) in ps:
        print("Working on job %d: %s" % (job_id, hyperparams), flush=True)
        model = sklearn.ensemble.RandomForestClassifier(**hyperparams)
        model.fit(x_tr, y_tr)
        p_va = model.predict(x_va)
        accuracy_va = metrics.accuracy_score(y_va, p_va)
        ps.submit_result(job_id, accuracy_va)


Distributed Usage
-----------------
You can distribute your hyperparameter search over several machines. To do this, set up your ParameterSearch
as usual in your server process, then call ```ParameterSearch.start_server(...)``` to make your
hyperparameter search available to the outside world.

Next start up any client processes: these create ParameterSearch instances that connect to the server process:

    ps = ParameterSearch(host="my.server.com", port=5732)

And then use the ParameterSearch as usual. It will connect to the server and receive parameter settings defined
there. See ```example_gridsearch.py``` for a simple example.


License
-------
Distributed Parameter Search is copyrighted (c) 2019 by Thomas Unterthiner and licensed under the
`General Public License (GPL) Version 2 or higher <http://www.gnu.org/licenses/gpl-2.0.html>`_.
See ``LICENSE.md`` for the full details.
