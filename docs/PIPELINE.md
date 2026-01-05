# Pipeline (treino, inferência, ensemble)

Este repo mantém a lógica em `src/forgeryseg/` e expõe CLIs em `scripts/` (wrappers leves).

Pré-requisitos:

- Dados no layout do Kaggle em `data/recodai` (ou outro `--data-root`)
- Checkpoints em `outputs/models/` (ou paths equivalentes no Kaggle via `/kaggle/input`)

## 1) Inferência (gerar submission.csv)

- Gerar submissão individual:
  - `python scripts/predict_submission.py --config configs/dino_v3_518_r69.json --data-root data/recodai --split test --out outputs/submission.csv`
  - (TTA mais forte) `python scripts/predict_submission.py --config configs/dino_v3_518_r69_fft_gate_tta_plus.json --data-root data/recodai --split test --out outputs/submission_tta_plus.csv`

Sanity-check do CSV:

- `python scripts/evaluate_submission.py --data-root data/recodai --split test --csv outputs/submission.csv`

Score local (somente quando existe GT):

- `python scripts/evaluate_submission.py --data-root data/recodai --split train --csv outputs/submission.csv --show-worst 20`

## 2) Treino (segmentação)

Treinar um modelo de segmentação e salvar checkpoint em `--out`:

- `python scripts/train_dino_decoder.py --config configs/dino_v3_518_r69.json --data-root data/recodai --out outputs/models/r69.pth --epochs 5`

K-fold estratificado (treina todos os folds se `train.fold=-1`):

- `python scripts/train_dino_decoder.py --config configs/dino_v3_518_r69.json --data-root data/recodai --out outputs/models/r69.pth --folds 5 --scheduler cosine --patience 3`

Logs por fold (CSV/JSON) ficam ao lado do `--out` (ex.: `outputs/models/r69_fold0.csv`).

## 3) Ensemble de submissões

Gerar N submissões e combinar:

- `python scripts/ensemble_submissions.py --data-root data/recodai --split test --subs outputs/sub1.csv outputs/sub2.csv --method weighted --scores 0.33 0.35 --out outputs/submission_ens.csv --diagnostics outputs/ensemble_diag.csv`

## 4) FFT gate (opcional)

O `fft_gate` revisa casos previstos como `authentic` (máscara vazia) e pode “destravar” uma máscara quando o classificador FFT indicar forjamento.

Treinar classificador FFT:

- `python scripts/train_fft_classifier.py --config configs/fft_classifier_logmag_256.json --data-root data/recodai --out outputs/models/fft_cls.pth`

Rodar submissão com `fft_gate` habilitado:

- `python scripts/predict_submission.py --config configs/dino_v3_518_r69_fft_gate.json --data-root data/recodai --split test --out outputs/submission_fft_gate.csv`

## 5) Fusão espacial + frequência (opcional)

Modelo `dinov2_freq_fusion` extrai uma representação FFT no `forward()` e faz fusão com os tokens do encoder antes do decoder.

- `python scripts/predict_submission.py --config configs/dino_v3_518_r69_freq_fusion.json --data-root data/recodai --split test --out outputs/submission_freq_fusion.csv`

Modos suportados em `freq_fusion.mode`: `logmag`, `hp_residual`, `phase_only`, `lp_hp`.

## 6) Extração multi-escala (opcional)

Modelo `dinov2_multiscale` extrai hidden states de múltiplas camadas do ViT (ex.: 3/6/9/12), projeta para o mesmo número de canais e faz fusão antes do decoder.

- `python scripts/predict_submission.py --config configs/dino_v4_518_r69_multiscale.json --data-root data/recodai --split test --out outputs/submission_multiscale.csv`
