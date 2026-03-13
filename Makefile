UV := uv
KM ?= 0
KM_ARG := $(if $(KM),--km $(KM),)


.PHONY: help setup predict train clean lint format

help:
	@echo "Targets:"
	@echo "  make setup   - install/update dependencies via uv"
	@echo "  make predict - run linear regression training"
	@echo " 			ex: make predict KM=4000"
	@echo "  make train   - run training"

setup:
	$(UV) sync

predict:
	$(UV) run python -m src.ft_lr $(KM_ARG)

train:
	$(UV) run python -m src.ft_lr

lint:
	-$(UV) run ruff check src
	$(UV) run mypy src

format:
	$(UV) run ruff format src

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf ft_liner_regression.egg-info
	rm -rf params.json plot.png
	rm -rf .mypy_cache .ruff_cache
