# Model Zoo (local layout)

Este repo **não** versiona checkpoints/pesos. Os `configs/*.json` assumem um layout local para encontrar os pesos em tempo de inferência/treino.

## Layout esperado

```
outputs/
  models/
    u52.pth
    v2.pth
    r69.pth
    r69_freq_fusion.pth
    fft_cls.pth
    fft_cls_phase.pth
```

## Mapeamento config → checkpoints

- `configs/dino_v1_718_u52.json`
  - `checkpoint`: `outputs/models/u52.pth`
- `configs/dino_v2_518_basev1.json`
  - `checkpoint`: `outputs/models/v2.pth`
- `configs/dino_v3_518_r69.json`
  - `checkpoint`: `outputs/models/r69.pth`
- `configs/dino_v3_518_r69_fft_gate.json`
  - `checkpoint`: `outputs/models/r69.pth`
  - `fft_gate.checkpoint`: `outputs/models/fft_cls.pth`
- `configs/dino_v3_518_r69_freq_fusion.json`
  - `checkpoint`: `outputs/models/r69_freq_fusion.pth`
- `configs/fft_classifier_logmag_256.json`
  - saída típica: `outputs/models/fft_cls.pth`
- `configs/fft_classifier_phase_only_256.json`
  - saída típica: `outputs/models/fft_cls_phase.pth`

## Kaggle

No Kaggle (internet OFF), a forma mais simples é criar um Kaggle Dataset contendo a pasta `outputs/models/` com esses arquivos e anexar esse dataset ao notebook de submissão.
