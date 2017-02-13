.PHONY: flake8 test coverage

flake8:
	flake8 --max-line-length=100 --count --statistics --exit-zero tyssue/

test:
	cd tyssue/ && py.test && cd ..

coverage:
	cd tyssue/ && py.test --cov=tyssue --cov-config ../.coveragerc --cov-report=term && cd ..
