.PHONY: flake8 test coverage

flake8:
	flake8 --max-line-length=100 --count --statistics --exit-zero tyssue/

test:
	py.test

coverage:
	py.test --cov=tyssue --cov-config=.coveragerc tests/
