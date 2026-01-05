# Tutoriais (workflow rápido)

Este documento mostra “receitas” para os 3 loops mais comuns na competição:

1) treinar um modelo
2) calibrar pós-processamento localmente
3) gerar e ensemblar submissões

## 1) Treinar do zero (Trainer)

O `Trainer` está em `src/forgeryseg/training/trainer.py` e é usado pelo wrapper:

- `scripts/train_dino_decoder.py`
 - No Kaggle: `notebooks/fase_01_treino_kaggle.ipynb` (template completo)

Exemplo (treino simples):

- `python scripts/train_dino_decoder.py --config configs/dino_v3_518_r69.json --data-root data/recodai --out outputs/models/r69.pth --epochs 5 --device cuda`

Exemplo (k-fold estratificado + scheduler + early stopping em `val_of1`):

- `python scripts/train_dino_decoder.py --config configs/dino_v3_518_r69.json --data-root data/recodai --out outputs/models/r69.pth --folds 5 --scheduler cosine --patience 3`

## 2) Calibrar thresholds de pós-processamento (local)

Fluxo recomendado:

1) gerar um `submission.csv` no `split=train` (ou `supplemental`)
2) medir o score local (oF1) com `scripts/evaluate_submission.py`
3) ajustar `inference.postprocess.*` (via config ou `--set`) e repetir

### 2.1 Avaliar um CSV existente

- `python scripts/evaluate_submission.py --data-root data/recodai --split train --csv outputs/submission.csv --show-worst 20`

### 2.2 Fazer “sweep” de `prob_threshold`

Use o script abaixo para gerar submissões no `train/supplemental` e escolher o melhor `prob_threshold` pelo score local:

- `python scripts/tune_postprocess.py --config configs/dino_v3_518_r69.json --data-root data/recodai --split train --out-dir outputs/tune --thr-start 0.30 --thr-stop 0.80 --thr-step 0.05 --limit 0`

> Dica: `--limit` ajuda a iterar rápido (subconjunto), mas o melhor threshold final deve ser validado no split completo.

### 2.3 Otimização bayesiana (Optuna) do pós-processo (recomendado)

Quando você quer ir além do sweep de threshold, use Optuna para ajustar vários campos de `inference.postprocess.*` em cima de um **cache** de `prob_maps` (inferência roda 1x; trials são rápidos):

- `python scripts/optuna_tune_postprocess.py --config configs/dino_v3_518_r69_fft_gate.json --data-root data/recodai --split train --out-dir outputs/optuna --val-fraction 0.10 --trials 200 --objective mean_score --set inference.fft_gate.enabled=false`

Saídas importantes:

- `outputs/optuna/optuna_best.json`: melhor score e overrides.
- `outputs/optuna/tuned_<config>_optuna_mean_score.json`: config pronta para usar na submissão.

## 3) Criar e combinar submissões (ensemble)

### 3.1 Gerar múltiplas submissões

- `python scripts/predict_submission.py --config configs/dino_v1_718_u52.json --data-root data/recodai --split test --out outputs/sub_1.csv`
- `python scripts/predict_submission.py --config configs/dino_v2_518_basev1.json --data-root data/recodai --split test --out outputs/sub_2.csv`
- `python scripts/predict_submission.py --config configs/dino_v3_518_r69.json --data-root data/recodai --split test --out outputs/sub_3.csv`

### 3.2 Ensemblar com pesos (via scores locais)

- `python scripts/ensemble_submissions.py --data-root data/recodai --split test --subs outputs/sub_1.csv outputs/sub_2.csv outputs/sub_3.csv --method weighted --scores 0.324 0.323 0.322 --out outputs/submission.csv --diagnostics outputs/ensemble_diag.csv`

O `--diagnostics` grava um CSV por `case_id` (área final, nº de instâncias, etc.) para depuração.
