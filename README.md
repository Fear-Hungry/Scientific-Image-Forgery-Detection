# Recod.ai/LUC — Scientific Image Forgery Detection (Kaggle)

Repositório com **pipeline + utilitários** para a competição do Kaggle **“Recod.ai/LUC - Scientific Image Forgery Detection”** (segmentação + RLE).  
Os dados **não são versionados** aqui: o projeto espera o layout do Kaggle em `data/recodai` (ver `docs/COMPETITION.md` e `docs/KAGGLE.md`).

- Página oficial: https://www.kaggle.com/competitions/recodai-luc-scientific-image-forgery-detection
- Métrica oficial (código + RLE): https://www.kaggle.com/code/metric/recodai-f1

## Quickstart

**Local (requer dados em `data/recodai` + checkpoints em `outputs/models/`):**

```bash
pip install -r requirements.txt
kaggle competitions download -c recodai-luc-scientific-image-forgery-detection -p data && unzip -q data/recodai-luc-scientific-image-forgery-detection.zip -d data/recodai
python scripts/predict_submission.py --config configs/dino_v3_518_r69_fft_gate.json --data-root data/recodai --split test --out outputs/submission.csv
```

**Kaggle (internet OFF):**

- Rode `notebooks/fase_00_submissao_kaggle.ipynb` (anexe o dataset da competição + um dataset com este repo + um dataset com `outputs/models/*.pth`).
- Saída: `/kaggle/working/submission.csv`
- Guia completo: `docs/KAGGLE.md`
- Esquema de configuração (configs): `docs/CONFIG.md`

## Dependências e reprodutibilidade

- `requirements.txt`: pinado para **rodar local/CI** de forma determinística.
- `requirements-kaggle.txt`: pinagem de libs “neutras” para Kaggle; **não fixa `torch/torchvision`** (use as versões do próprio Kaggle e ajuste só se necessário).
- Para atualizar versões: edite `requirements.txt`/`requirements-kaggle.txt`, rode `make lint test` e valide o notebook `notebooks/fase_00_submissao_kaggle.ipynb`.

## Docs

- `docs/COMPETITION.md`: formato de dados, RLE, métrica (oF1) e notas da competição.
- `docs/KAGGLE.md`: como submeter no Kaggle (internet OFF) + sanity-check.
- `docs/CONFIG.md`: schema de configs (JSON/YAML) + `--set` overrides.
- `docs/MODEL_ZOO.md`: layout esperado de `outputs/models/`.
- `docs/PIPELINE.md`: comandos de treino/inferência/ensemble/FFT gate.
- `docs/TUTORIALS.md`: receitas rápidas (treinar, calibrar pós-processo, ensemble).

## Atualizações (2026-01-04)

- Inferência mais modular: `InferenceEngine` + `SubmissionWriter` (inclui `batch_size` quando não usa tiling) e `FFTGate` opcional.
- Avaliação local: `scripts/evaluate_submission.py` valida CSV/decodificação e calcula o score (oF1) em `train/supplemental`.
- Ensemble melhorado: pesos automáticos por score (bugfix) e opção `--diagnostics` para depuração por `case_id`.
- Documentação/atalhos: `docs/CONFIG.md`, `docs/KAGGLE.md`, `Makefile` e `requirements-kaggle.txt` (pinagem para reproduzir ambiente).
- Qualidade: mais testes unitários (dataset, postprocess, FFT gate, Trainer checkpoint), CI e `pre-commit`.

## Atualizações (2026-01-05)

- Normalização correta do DINOv2 (mean/std do `timm`) embutida nos modelos (melhora convergence/underfit).
- Tuning bayesiano do pós-processo: `scripts/optuna_tune_postprocess.py` (cache de prob_maps + Optuna) e integração no `notebooks/fase_01_treino_kaggle.ipynb`.
- Modelo multi-escala (features de múltiplas camadas do ViT): `model.type=dinov2_multiscale` + exemplo em `configs/dino_v4_518_r69_multiscale.json`.

## Citação

Para citar este repositório em um paper, use `CITATION.bib`.

## Contribuindo

Veja `CONTRIBUTING.md`.
