PYTHON ?= python

DATA_ROOT ?= data/recodai
SPLIT ?= test

# --------
# Install
# --------

.PHONY: install
install:
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

# -----
# Lint
# -----

.PHONY: lint
lint:
	$(PYTHON) -m ruff check .

# -----
# Test
# -----

.PHONY: test
test:
	$(PYTHON) -m pytest -q

# ------------------------
# Training / Inference CLI
# ------------------------

TRAIN_CONFIG ?= configs/dino_v3_518_r69.json
TRAIN_OUT ?= outputs/models/r69.pth
TRAIN_DEVICE ?= cuda

.PHONY: train
train:
	$(PYTHON) scripts/train_dino_decoder.py --config $(TRAIN_CONFIG) --data-root $(DATA_ROOT) --out $(TRAIN_OUT) --device $(TRAIN_DEVICE)

FFT_CONFIG ?= configs/fft_classifier_logmag_256.json
FFT_OUT ?= outputs/models/fft_cls.pth

.PHONY: train_fft
train_fft:
	$(PYTHON) scripts/train_fft_classifier.py --config $(FFT_CONFIG) --data-root $(DATA_ROOT) --out $(FFT_OUT) --device $(TRAIN_DEVICE)

PRED_CONFIG ?= configs/dino_v3_518_r69_fft_gate.json
PRED_OUT ?= outputs/submission.csv
PRED_DEVICE ?= cuda

.PHONY: predict
predict:
	$(PYTHON) scripts/predict_submission.py --config $(PRED_CONFIG) --data-root $(DATA_ROOT) --split $(SPLIT) --out $(PRED_OUT) --device $(PRED_DEVICE)

.PHONY: eval
eval:
	$(PYTHON) scripts/sanity_submissions.py --data-root $(DATA_ROOT) --out-dir outputs/sanity --split train

.PHONY: package
package:
	$(PYTHON) scripts/package_kaggle_dataset.py --out-dir kaggle_bundle --include-models

.PHONY: sync_notebook
sync_notebook:
	$(PYTHON) scripts/sync_ipynb_from_py.py --py notebooks/fase_00_submissao_kaggle.py
