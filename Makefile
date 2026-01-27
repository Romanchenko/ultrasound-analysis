.PHONY: test test-verbose test-fetal-head-circ test-fetal-planes help

# Activate virtual environment
VENV = venv
PYTHON = $(VENV)/bin/python
PYTEST = $(VENV)/bin/pytest

# Default target
help:
	@echo "Available targets:"
	@echo "  make test                    - Run all tests with pytest"
	@echo "  make test-verbose            - Run all tests with verbose output"
	@echo "  make test-fetal-head-circ    - Run tests for fetal_head_circ dataset"
	@echo "  make test-fetal-planes       - Run tests for fetal_planes_db dataset"
	@echo "  make test-coverage           - Run tests with coverage report"

# Run all tests
test:
	$(PYTEST) datasets/ -v

# Run all tests with verbose output
test-verbose:
	$(PYTEST) datasets/ -vv

# Run tests for fetal_head_circ dataset
test-fetal-head-circ:
	$(PYTEST) datasets/fetal_head_circ/basic_test/ -v

# Run tests for fetal_planes_db dataset
test-fetal-planes:
	$(PYTEST) datasets/fetal_planes_db/basic_test/ -v

# Run tests for cyclegan
test-cycle-gan:
	$(PYTEST) domain_adaptation/cyclegan/ -v