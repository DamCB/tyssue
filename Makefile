.PHONY: flake8 test coverage

flake8:
	flake8 --max-line-length=100 --count --statistics --exit-zero tyssue/

test:
	pytest tyssue/tests/

coverage:
	pytest --cov=tyssue --cov-config=.coveragerc tyssue/tests/
