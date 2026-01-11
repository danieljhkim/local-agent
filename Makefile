.PHONY: help install install-dev test lint format type-check clean run server db-init dev all

# Default target
help:
	@echo "Local Agent - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install package in editable mode"
	@echo "  make install-dev    Install with dev dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make dev            Install dev deps and run all checks"
	@echo "  make run            Run the agent CLI"
	@echo "  make server         Start the FastAPI server"
	@echo ""
	@echo "Code Quality:"
	@echo "  make format         Auto-format code with ruff"
	@echo "  make lint           Run linting checks"
	@echo "  make type-check     Run mypy type checking"
	@echo "  make test           Run tests with pytest"
	@echo ""
	@echo "Database:"
	@echo "  make db-init        Initialize database and storage"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          Remove build artifacts and cache"
	@echo "  make clean-all      Deep clean (including storage/db)"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# Development workflow
dev: install-dev format lint type-check test
	@echo "✓ All checks passed!"

# Run commands
run:
	agent

server:
	uvicorn local_agent.web.app:app --reload --host 0.0.0.0 --port 8000

# Database
db-init:
	@mkdir -p storage
	@echo "Storage directory created"

# Code quality
format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

lint:
	ruff check src/ tests/

type-check:
	mypy src/

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=local_agent --cov-report=html --cov-report=term

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ htmlcov/ .coverage

clean-all: clean
	rm -rf storage/
	@echo "⚠️  Storage and database removed"

# Quick aliases
all: dev
