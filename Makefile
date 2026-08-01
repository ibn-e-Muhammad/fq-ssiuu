.PHONY: setup setup-dev test lint clean sync-drive pull-drive install-editable help

help:
	@echo "FQ-SSIUU Project Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make setup          Run the full environment setup (calls scripts/setup_env.sh)"
	@echo "  make install-editable  Install the fq_ssiuu package in editable mode"
	@echo "  make test           Run all unit tests with pytest"
	@echo "  make lint           Run ruff linter on all source files"
	@echo "  make clean          Remove __pycache__, .ruff_cache, .pytest_cache"
	@echo "  make sync-drive     Sync results and logs to Google Drive"
	@echo "  make pull-drive     Pull checkpoints from Google Drive"
	@echo "  make gate0          Run all GATE-0 environment verification checks"

setup:
	@echo "Running environment setup..."
	bash scripts/setup_env.sh

install-editable:
	pip install -e ".[dev]"
	@echo "Package installed in editable mode. Import as: from fq_ssiuu import ..."

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

gate0:
	python3.11 experiments/gate0_env_check.py
