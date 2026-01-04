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
    "freq_fusion": {}
  },
  "inference": {
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
    "scheduler": "none"
  }
}
```

### Campos principais

- `model.type`: `"dinov2"` ou `"dinov2_freq_fusion"`
- `model.input_size`: tamanho do *letterbox* (usado em treino e inferência)
- `model.checkpoint`: checkpoint do modelo completo (usado na inferência)
- `model.encoder.*`: especificação do encoder DINOv2 (timm)
- `inference.tta.*`: TTA (por padrão: identidade + flip + zoom-out)
- `inference.tiling.*`: *tiled inference* (opcional; use quando imagem é grande)
- `inference.postprocess.*`: threshold/filtros/regras para decidir `authentic` e instâncias
- `inference.fft_gate.*`: classificador FFT opcional para revisar casos `authentic`
- `train.*`: defaults de treino (podem ser sobrescritos no CLI)

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

