.PHONY: flake8 test coverage

flake8:
	flake8 --max-line-length=100 --count --statistics --exit-zero tyssue/

test:
	py.test tyssue/tests/

coverage:
	py.test --cov=tyssue --cov-config=.coveragerc tyssue/tests/
