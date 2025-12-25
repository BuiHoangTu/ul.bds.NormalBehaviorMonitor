.PHONY: setup preprocess train

setup:
	@if command -v micromamba &> /dev/null; then \
		micromamba create -n nbm -f env.yml -y; \
	else \
		echo "micromamba not found, using pip"; \
		pip install -e .; \
	fi

preprocess:
	python src/nbm/preprocess/__init__.py --config-path train-config.yaml

train:
	python src/nbm/train/__init__.py --config-path train-config.yaml --model-type $(MODEL_TYPE)