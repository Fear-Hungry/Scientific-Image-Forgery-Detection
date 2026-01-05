# Configuração (configs/*.json|yaml)

Este projeto usa **configs tipados** (ver `src/forgeryseg/config.py`) com:

- separação entre **modelo**, **treino** e **inferência/pós-processamento**
- suporte a **JSON** (e **YAML** opcional, via `pyyaml`)
- suporte a **overrides por CLI** via `--set chave.subchave=valor`
- **paths relativos** resolvidos a partir do diretório da config e (no Kaggle) busca em `/kaggle/input`

## 1) Config de segmentação (DINOv2)

Usada por:

- `scripts/predict_submission.py`
- `scripts/train_dino_decoder.py`
- `notebooks/fase_00_submissao_kaggle.ipynb`

### Estrutura

```json
{
  "name": "v3_r69_518",
  "model": {
    "type": "dinov2",
    "input_size": 518,
    "encoder": {
      "model_name": "vit_base_patch14_reg4_dinov2",
      "checkpoint_path": null,
      "pretrained": true
    },
    "checkpoint": "outputs/models/r69.pth",
    "decoder_hidden_channels": 256,
    "decoder_dropout": 0.0,
    "freeze_encoder": true,
    "freq_fusion": {},
    "multiscale": {}
  },
  "inference": {
    "batch_size": 1,
    "tta": {"zoom_scale": 0.9, "weights": [0.6, 0.2, 0.2]},
    "tiling": {"tile_size": 1024, "overlap": 128, "batch_size": 4},
    "postprocess": {"prob_threshold": 0.5, "...": 0},
    "fft_gate": {"enabled": true, "...": "..."}
  },
  "train": {
    "epochs": 5,
    "batch_size": 4,
    "lr": 0.001,
    "weight_decay": 0.0001,
    "folds": 1,
    "aug": "basic",
    "scheduler": "none",
    "patience": 0,
    "min_delta": 0.0
  }
}
```

### Campos principais

- `model.type`: `"dinov2"`, `"dinov2_freq_fusion"` ou `"dinov2_multiscale"`
- `model.input_size`: tamanho do *letterbox* (usado em treino e inferência)
- `model.checkpoint`: checkpoint do modelo completo (usado na inferência)
- `model.encoder.*`: especificação do encoder DINOv2 (timm)
- `model.multiscale.*`: extração multi-camada (somente em `dinov2_multiscale`)
- `inference.tta.*`: TTA (por padrão: identidade + flip + zoom-out)
- `inference.batch_size`: batch de imagens (quando `tiling` está desligado)
- `inference.tiling.*`: *tiled inference* (opcional; use quando imagem é grande)
- `inference.postprocess.*`: threshold/filtros/regras para decidir `authentic` e instâncias
- `inference.fft_gate.*`: classificador FFT opcional para revisar casos `authentic`
- `train.*`: defaults de treino (podem ser sobrescritos no CLI)

### `inference.postprocess` (documentação rápida)

Campos (ver também `src/forgeryseg/postprocess.py`):

- `prob_threshold`: threshold principal (pixel) para binarizar o `prob_map`.
- `prob_threshold_low`: threshold “baixo” opcional (hysteresis). Se setado (`< prob_threshold`), mantém regiões `>= prob_threshold_low` **apenas** quando conectadas a seeds `>= prob_threshold`.
- `gaussian_sigma`: suavização do `prob_map` via Gaussian blur (0 desliga).
- `sobel_weight`: reforço de borda via Sobel no `prob_map` (0 desliga).
- `open_kernel`: abertura morfológica (remove ruído); `<=1` desliga.
- `close_kernel`: fechamento morfológico (fecha buracos); `<=1` desliga.
- `morph_order`: ordem da morfologia: `"open_close"` (padrão) ou `"close_open"`.
- `final_open_kernel`: abertura morfológica **final** (2ª passada, opcional); `<=1` desliga.
- `final_close_kernel`: fechamento morfológico **final** (2ª passada, opcional); `<=1` desliga.
- `fill_holes`: preenche buracos internos da máscara união (útil para contornos mais suaves).
- `min_area`: remove componentes com área menor que este valor.
- `min_mean_conf`: exige `mean(prob)` da máscara união acima deste valor.
- `min_prob_std`: se `std(prob_map)` for menor que este valor, retorna `authentic` (útil para “mapa flat”).
- `small_area`: limiar de área para regra especial de componentes pequenos (opcional).
- `small_min_mean_conf`: confiança mínima exigida quando `area < small_area` (opcional).
- `authentic_area_max`: heurística “authentic”: se a máscara união tiver `area < authentic_area_max` (opcional).
- `authentic_conf_max`: e `mean(prob) < authentic_conf_max`, então retorna `authentic` (opcional).

Observações:

- `authentic_*` é útil para reduzir falsos positivos pequenos/baixos; combine com `min_area`.
- Com `fft_gate` ligado, casos previstos como `authentic` podem ser reavaliados com heurísticas menos agressivas.

### `model.multiscale` (somente `dinov2_multiscale`)

- `layers`: camadas do ViT para extrair features (aceita `[2,5,8,11]` ou `[3,6,9,12]` no modelo base).
- `proj_channels`: canais após projeção 1×1 por camada (antes da fusão).
- `fuse`: `"concat"` (padrão) ou `"sum"`.
- `decoder_depth`: profundidade do head CNN após fusão (>= 1).

## 2) Config do classificador FFT

Usada por:

- `scripts/train_fft_classifier.py`

### Estrutura

```json
{
  "name": "fft_cls_logmag_256",
  "fft": {
    "mode": "logmag",
    "input_size": 256,
    "hp_radius_fraction": 0.1,
    "normalize_percentiles": [5.0, 95.0]
  },
  "model": {"backbone": "resnet18", "dropout": 0.0},
  "train": {"epochs": 5, "batch_size": 32, "lr": 0.001}
}
```

## 3) Overrides por linha de comando

Exemplos:

- Ajustar pós-processamento na submissão (sem editar o arquivo):
  - `python scripts/predict_submission.py --config configs/dino_v3_518_r69.json --data-root data/recodai --out outputs/sub.csv --set inference.postprocess.min_area=200`
- Rodar treino com k-fold e scheduler:
  - `python scripts/train_dino_decoder.py --config configs/dino_v3_518_r69.json --data-root data/recodai --out outputs/models/r69.pth --set train.folds=5 --set train.scheduler=cosine`

Valores são parseados como JSON quando possível (`true/false`, números, listas).

## 4) Logs de treino (CSV/JSON)

O `Trainer` grava logs por fold ao lado do `--out`:

- `outputs/models/r69_fold0.csv` (métricas por época)
- `outputs/models/r69_fold0.json` (histórico + resumo do fold)

E imprime no final a média dos melhores scores entre folds.
