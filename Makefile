
help:
	@echo "build - compile all coco files"
	@echo "test - run tests with default Python"

build:
	coconut -f ./coconut_learning/
	coconut -f ./tests/

test: build
	python setup.py test
