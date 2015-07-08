.PHONY: flake8 test coverage

flake8:
	flake8 --max-line-length=100 --count --statistics --exit-zero src/py-tyssue/tyssue/

test:
	nosetests tyssue -v

coverage:
	nosetests tyssue --with-coverage --cover-package=tyssue -v

full-test:
	bash .full_test.sh
