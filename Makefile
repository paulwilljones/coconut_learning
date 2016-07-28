
help:
	@echo "build - compile all coco files"
	@echo "clean - remove all Python artifacts"
	@echo "test - run tests with default Python"

build:
	coconut -f ./coconut_learning/
	coconut -f ./tests/

clean:
	find . -path './coconut_learning/__init__.py' -exec truncate -s 0 {} + -o -path '*/coconut_learning/*.py*' -delete
	find . -path './tests/__init__.py' -exec truncate -s 0 {} + -o -path '*/tests/*.py*' -delete

test:
	python setup.py test
