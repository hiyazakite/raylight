# ===========================================================================
# Makefile — Developer workflow for raylight
#
# Common commands:
#   make build          Build all native extensions
#   make test           Run test suite
#   make lint           Lint + format check
#   make clean          Remove build artifacts
#   make install        Editable install with dev dependencies
# ===========================================================================

PYTHON   ?= python3.13
SRC_DIR  := src
CSRC_DIR := csrc/quantization/gguf

# Allocator paths
ALLOC_SRC := csrc/alloc/raylight_alloc.c
ALLOC_DST := $(SRC_DIR)/raylight/lib/raylight_alloc.so

# CUDA include path (override with: make build-alloc CUDA_HOME=/opt/cuda)
CUDA_HOME ?= /usr/local/cuda

# Environment for tests
export PYTHONPATH := $(SRC_DIR):$(PYTHONPATH)
export CUDA_VISIBLE_DEVICES ?= 0

.PHONY: help build build-cuda build-alloc clean test lint format install check

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

build: build-cuda build-alloc ## Build all native extensions

build-cuda: ## Build GGUF fused CUDA extension (requires torch + nvcc)
	$(PYTHON) setup.py build_ext --inplace

build-alloc: $(ALLOC_DST) ## Build allocator interceptor (requires gcc + libcuda)

$(ALLOC_DST): $(ALLOC_SRC)
	@mkdir -p $(dir $@)
	gcc -shared -fPIC -o $@ $< -I$(CUDA_HOME)/include -lcuda -ldl
	@echo "Built: $@"

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------

install: ## Editable install with dev dependencies
	$(PYTHON) -m pip install -e ".[dev]"

# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

test: ## Run CPU-safe test suite
	$(PYTHON) -m pytest tests/ -v -m "not gpu and not triton"

test-gpu: ## Run GPU tests (requires CUDA)
	$(PYTHON) -m pytest tests/ -v -m "gpu or triton"

test-all: ## Run all tests
	$(PYTHON) -m pytest tests/ -v

test-cuda: ## Run GGUF CUDA kernel tests only
	$(PYTHON) -m pytest tests/expansion/test_gguf_fused_kernels.py tests/expansion/test_gguf_lora_residual.py -v

# ---------------------------------------------------------------------------
# Lint / format
# ---------------------------------------------------------------------------

lint: ## Run linter (ruff check)
	ruff check $(SRC_DIR)/ tests/

format: ## Auto-format code (ruff format)
	ruff format $(SRC_DIR)/ tests/

check: lint ## Alias for lint

# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	rm -f build_error.log build_gguf_error.log build_raylight_alloc_error.log
	find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true

clean-so: ## Remove built .so files (forces rebuild)
	rm -f $(SRC_DIR)/raylight/expansion/comfyui_gguf/_C_gguf*.so
	rm -f $(ALLOC_DST)
	@echo "Removed .so files — run 'make build' to rebuild"
