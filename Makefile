.PHONY: setup test lint watchdog paper analyst options crypto benchmark

setup:
	pip install -r requirements-dev.txt

test:
	python -m pytest tests/ -q

lint:
	ruff check agents/ scripts/ data_pipelines/ tests/

watchdog:
	python scripts/political_watchdog.py --dry-run

paper:
	python scripts/paper_trader.py --dry-run

analyst:
	python scripts/market_analyst.py --dry-run --ticker NVDA

options:
	python scripts/options_flow_monitor.py --dry-run --ticker NVDA

crypto:
	python scripts/crypto_trader.py --dry-run

benchmark:
	python scripts/model_benchmark.py
