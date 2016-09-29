.PHONY: flake8 test coverage

flake8:
	flake8 --max-line-length=100 --count --statistics --exit-zero tyssue/

test:
	py.test

coverage:
	#nosetests tyssue --with-coverage --cover-package=tyssue -v
	py.test --cov-config coveragerc

full-test:
	bash .full_test.sh
