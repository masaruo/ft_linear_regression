UV := uv
DATA ?= data.csv
EPOCHS ?= 50000
LR ?= 1e-2
KM ?=
SEED ?= 42
TEST_RATIO ?= 0.2
PLOT ?= plot.png

RUN_ARGS := --data $(DATA) --epochs $(EPOCHS) --lr $(LR)
KM_ARG := $(if $(KM),--km $(KM),)
PLOT_ARG := $(if $(PLOT),--plot $(PLOT),)

.PHONY: help setup run eval plot lint format test clean

help:
	@echo "Targets:"
	@echo "  make setup  - install/update dependencies via uv"
	@echo "  make run    - run linear regression training"
	@echo "               ex: make run KM=100000"
	@echo "  make eval   - run train/test evaluation metrics"
	@echo "               ex: make eval TEST_RATIO=0.25 SEED=7"
	@echo "  make plot   - save fit plot image (default: plot.png)"
	@echo "               ex: make plot PLOT=fit.png"
	@echo "  make lint   - run ruff checks"
	@echo "  make format - run ruff formatter"
	@echo "  make test   - run pytest"
	@echo "  make clean  - remove Python cache files"

setup:
	$(UV) sync

run:
	$(UV) run python -m src.ft_lr $(RUN_ARGS) $(KM_ARG)

eval:
	$(UV) run python -m src.ft_lr $(RUN_ARGS) --eval --test-ratio $(TEST_RATIO) --seed $(SEED) $(KM_ARG)

plot:
	$(UV) run python -m src.ft_lr $(RUN_ARGS) --eval --test-ratio $(TEST_RATIO) --seed $(SEED) $(PLOT_ARG)

lint:
	$(UV) run ruff check src

format:
	$(UV) run ruff format src

test:
	$(UV) run pytest

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf ft_liner_regression.egg-info
