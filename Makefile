.PHONY: setup setup-dev kaggle-preview kaggle-apply install-editable test test-infra lint clean sync-drive pull-drive help

help:
	@echo "FQ-SSIUU Project Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make setup-dev         Install generic local development environment (pep621)"
	@echo "  make kaggle-preview    Run Kaggle overlay dry-run report validation (non-mutating)"
	@echo "  make kaggle-apply      Apply Kaggle overlay installation (requires CUDA_VISIBLE_DEVICES=0)"
	@echo "  make install-editable  Install package in editable mode with dev dependencies"
	@echo "  make test-infra        Run infrastructure unit tests (unittest)"
	@echo "  make test              Run unit tests with pytest"
	@echo "  make lint              Run ruff linter on all source files"
	@echo "  make clean             Remove __pycache__, .ruff_cache, .pytest_cache"

setup:
	@echo "=========================================================================="
	@echo "ERROR: 'make setup' is DEPRECATED to prevent silent environment mutation."
	@echo "For generic local development, use:  make setup-dev"
	@echo "For Kaggle environment preview, use: make kaggle-preview"
	@echo "For Kaggle environment apply, use:   make kaggle-apply"
	@echo "=========================================================================="
	@exit 1

setup-dev:
	pip install -e ".[quantisation,attribution,evaluation,dev]"

kaggle-preview:
	python3 scripts/kaggle_setup.py --preview

kaggle-apply:
	CUDA_VISIBLE_DEVICES=0 python3 scripts/kaggle_setup.py --apply

install-editable:
	pip install -e ".[dev]"
	@echo "Package installed in editable mode."

test-infra:
	python3 -m unittest -v tests/test_kaggle_dependency_policy.py

test:
	@echo "Running unit tests..."
	pytest tests/ -v --tb=short

lint:
	@echo "Running ruff linter..."
	ruff check src/ tests/ experiments/
	ruff format --check src/ tests/ experiments/

lint-fix:
	ruff check --fix src/ tests/ experiments/
	ruff format src/ tests/ experiments/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "Clean complete."

sync-drive:
	bash scripts/sync_to_drive.sh

pull-drive:
	bash scripts/pull_from_drive.sh
