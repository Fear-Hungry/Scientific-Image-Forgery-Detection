# Experimentos (Fase 01) — Recod.ai/LUC

Este documento define um **protocolo mínimo e reproduzível** para explorar modelos e pós-processamento, selecionar os melhores candidatos e preparar um `infer_ensemble.json` consistente para a submissão (Fase 00).

## Princípios

- **Mude 1 coisa por vez** (ablation): arquitetura/encoder, resolução (patch/tile), loss, augment, pós-processamento, ensemble.
- **Fonte de verdade do score:** `src/forgeryseg/metric.py` (oF1 por instância).
- **OOF sempre que possível**: gere probabilidades OOF e use `scripts/tune_thresholds.py` para ajustar o pós-processamento.

## 0) Baseline (comece aqui)

Treine um segmentador “bom e barato” e gere OOF:

```bash
python scripts/train_seg_smp_cv.py --config configs/seg_unetpp_tu_convnext_small.json --data-root data/recodai --output-dir outputs --folds 5
python scripts/predict_seg_oof.py --data-root data/recodai --output-dir outputs --model-id unet_tu_convnext_small --folds 5
```

## 1) Tuning de pós-processamento (oF1)

Com OOF gerado em `outputs/oof/<model_id>/fold_k/...`, rode um grid simples:

```bash
python scripts/tune_thresholds.py --data-root data/recodai --preds-root outputs/oof/unet_tu_convnext_small --folds 5 --adaptive-threshold
```

Isso escreve um JSON com os melhores hiperparâmetros (por padrão em `configs/thresholds.json`).

Sugestão: salve por modelo:

```bash
python scripts/tune_thresholds.py --data-root data/recodai --preds-root outputs/oof/unet_tu_convnext_small --folds 5 --adaptive-threshold --out-config outputs/configs/postproc_unet_tu_convnext_small.json
```

## 2) Comparar arquiteturas (mesmo budget)

Repita **o mesmo protocolo** (treino → OOF → tuning) para configs de `configs/`:

- `configs/seg_unetpp_tu_swin_tiny.json`
- `configs/seg_deeplabv3p_tu_resnest101e.json`
- `configs/seg_segformer_mit_b2.json`
- `configs/seg_dinov2_base*.json` (cuidado: exige cache HF se for rodar offline)

## 3) Ensemble (pesos) a partir de OOF

Se você já tem OOF de múltiplos modelos em `outputs/oof/<model_id>`, pode otimizar pesos (proxy de Dice) com:

```bash
python scripts/optimize_ensemble.py --data-root data/recodai --oof-dir outputs/oof --models unet_tu_convnext_small,segformer_mit_b2 --out outputs/ensemble_weights.json
```

Depois, copie os pesos para `configs/infer_ensemble.json` (campo `models`) ou gere uma variante em `outputs/configs/`.

## 4) Classifier gate (opcional)

Treinar classificador pode reduzir falsos positivos ao “pular” segmentação em imagens autênticas:

```bash
python scripts/train_cls_cv.py --config configs/cls_effnet_b4.json --data-root data/recodai --output-dir outputs --folds 5
```

Na submissão (Fase 00), `scripts/submit_ensemble.py` tenta detectar `outputs/models_cls` automaticamente.  
Se não houver classificador, o gate pode ser desativado via `--cls-skip-threshold 0.0`.

## 5) Preparar para a submissão (Fase 00)

1. Coloque checkpoints em `outputs/models_seg/...` (e opcionalmente `outputs/models_cls/...`).
2. Atualize/clone `configs/infer_ensemble.json` com:
   - TTA (`tta_modes`)
   - tile/overlap
   - pós-processamento (saída do tuning)
   - pesos do ensemble
3. Rode `notebooks/fase_00_submissao_kaggle.ipynb` (Kaggle internet OFF).

