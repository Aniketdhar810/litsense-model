# Developer tasks. CI runs `make install-ci lint test`. Recipes assume a POSIX shell (Linux,
# Colab, macOS, Git Bash on Windows); README.md lists the plain commands behind each target.

PYTHON ?= python
# Packages from requirements-ml.txt that the fast (not slow) tests need. CI installs only these on
# top of requirements-core.txt; extend the list when a phase's fast tests import more.
# scikit-learn: ranking.train fits lightgbm's LGBMRanker (lightgbm.sklearn) in the bundle tests (D81).
CI_ML_PACKAGES := pandas|pyarrow|requests|pytest|ruff|matplotlib|scipy|implicit|onnx|scikit-learn

.PHONY: install-core install-ml install-ci install-colab install-service lint format test test-all \
	tiny stages service service-test service-openapi service-image

install-core:
	$(PYTHON) -m pip install -r requirements-core.txt

# The HTTP model service (service/README.md): core + FastAPI, uvicorn, settings, metrics.
install-service:
	$(PYTHON) -m pip install -r requirements-core.txt -r requirements-service.txt

install-ml:
	$(PYTHON) -m pip install -r requirements-core.txt -r requirements-ml.txt

# Colab keeps its preinstalled torch (docs/decisions.md D21): everything else from the pins.
install-colab:
	grep -v '^torch==' requirements-ml.txt > .colab-requirements.txt
	$(PYTHON) -m pip install -q -r requirements-core.txt -r .colab-requirements.txt
	rm -f .colab-requirements.txt

# CPU-only torch from PyTorch's index for the LightGCN tests (the PyPI wheel bundles CUDA, D43).
install-ci:
	grep -E '^($(CI_ML_PACKAGES))==' requirements-ml.txt > .ci-requirements.txt
	$(PYTHON) -m pip install -r requirements-core.txt -r requirements-service.txt \
		-r .ci-requirements.txt
	$(PYTHON) -m pip install --index-url https://download.pytorch.org/whl/cpu \
		$$(grep '^torch==' requirements-ml.txt)
	rm -f .ci-requirements.txt

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .

test:
	$(PYTHON) -m pytest -m "not slow"

test-all:
	$(PYTHON) -m pytest

tiny:
	$(PYTHON) -m tests.fixtures.tiny.make_tiny --out local/tiny

stages:
	$(PYTHON) -m ml.stages list

# Run the model service (needs BUNDLE_DIR with an extracted bundle or BUNDLE_URL, and
# SERVICE_API_KEY or ALLOW_NO_AUTH=true; see service/README.md).
service:
	$(PYTHON) -m service

service-test:
	$(PYTHON) -m pytest tests/service -m "not slow"

service-openapi:
	$(PYTHON) -m service.openapi

service-image:
	docker build -f service/Dockerfile -t litsense-service .
