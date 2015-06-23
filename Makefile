.PHONY: flake8 test coverage

flake8:
	flake8 --max-line-length=80 --count --statistics --exit-zero src/tyssue

test:
	nosetests tyssue --all-modules -v

coverage:
	nosetests tyssue --with-coverage --cover-package=tyssue --all-modules -v
