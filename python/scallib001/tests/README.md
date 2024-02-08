## Collection of functionality and cpu performance tests for scallib001

The tests make use of the standard python test library [pytest](https://docs.pytest.org/en/7.1.x/getting-started.html). 

When necessary, pytest can be installed using pip: pip install pytest.

If in addition you want to do cpu performance benchmarking, you need to
install the pytest plugin [pytest-benchmark](https://pytest-benchmark.readthedocs.io/en/latest/).

The pytest-benchmark plugin for pytest can be installed using pip: pip install pytest-benchmark.

To run the tests, go to the directory that contains scallib001 and do

python -m pytest scallib001

