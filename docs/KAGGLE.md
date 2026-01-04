# Kaggle (code competition) — guia rápido

Este projeto foi feito para rodar com **internet OFF** na submissão final. A ideia é:

1) anexar o dataset da competição (imagens/máscaras)  
2) anexar um dataset com **este repo** (código)  
3) anexar um dataset com **checkpoints** em `outputs/models/*.pth`  
4) rodar o notebook de submissão e gerar `/kaggle/working/submission.csv`

## 1) Datasets a anexar no Notebook

- **Competição**: `recodai-luc-scientific-image-forgery-detection`
- **Código**: um Kaggle Dataset criado a partir deste repo (pelo menos `src/`, `configs/`, `scripts/`, `notebooks/`)
- **Pesos** (opcional, mas necessário para inferência com modelos): um Kaggle Dataset com:

```
outputs/
  models/
    r69.pth
    fft_cls.pth
    ...
```

## 2) Notebook de submissão

Abra e rode:

- `notebooks/fase_00_submissao_kaggle.ipynb`

Ele:

- auto-detecta `data_root` em `/kaggle/input/.../recodai` (ou `.../sample_submission.csv`)
- auto-detecta `code_root` procurando `src/forgeryseg`
- resolve checkpoints procurando:
  - primeiro no CWD (ex.: `/kaggle/working/outputs/models/...`)
  - depois dentro do dataset do código e/ou do dataset de pesos em `/kaggle/input/.../outputs/models/...`

Saída final:

- `/kaggle/working/submission.csv`

Configs:

- Estrutura e campos documentados em `docs/CONFIG.md`
- No CLI, você pode sobrescrever parâmetros sem editar o arquivo via `--set chave=valor` (ex.: `--set inference.postprocess.min_area=200`)

Sanity-check (recomendado):

- Para checar **formato/decodificação** do CSV no Kaggle (sem ground truth), rode:
  - `python scripts/evaluate_submission.py --data-root <DATA_ROOT> --split test --csv /kaggle/working/submission.csv`
- Para checar **score local** (quando rodar em `train`/`supplemental`):
  - `python scripts/evaluate_submission.py --data-root <DATA_ROOT> --split train --csv /kaggle/working/submission.csv --show-worst 20`

## 3) Treino no Kaggle (opcional)

Se você quiser treinar dentro do Kaggle (internet ON), rode os scripts a partir do notebook:

- Segmentação (DINOv2 + decoder):
  - `python scripts/train_dino_decoder.py --config configs/dino_v3_518_r69.json --data-root /kaggle/input/<COMP_DATASET>/recodai --out /kaggle/working/outputs/models/r69.pth --epochs 5`
- Classificador FFT (para `fft_gate`):
  - `python scripts/train_fft_classifier.py --config configs/fft_classifier_logmag_256.json --data-root /kaggle/input/<COMP_DATASET>/recodai --out /kaggle/working/outputs/models/fft_cls.pth --epochs 5`

Opções úteis:

- `--folds 5` (k-fold estratificado). Saída vira `..._fold0.pth`, `..._fold1.pth`, etc.
- `--scheduler cosine` (ou `onecycle`) para melhorar convergência.
- `--patience 3` para early stopping baseado em `val_of1` (oF1).
- Segmentação: `--aug robust` para aumentar robustez (rotações/escala leve, blur e ruído).

Notas:

- Nos configs DINOv2, `encoder.pretrained=true` é usado **só no treino** (para baixar pesos do timm); na inferência, o código ignora isso quando existe `checkpoint` do modelo completo.
- Para rodar a submissão sem internet, coloque os `.pth` em um Kaggle Dataset e anexe ao notebook final.
- Logs por fold (CSV/JSON) são gravados ao lado do `--out` (ver `docs/CONFIG.md`).

## 4) Empacotar para Kaggle Dataset (local)

Você pode gerar uma pasta pronta para upload via:

- `python scripts/package_kaggle_dataset.py --out-dir kaggle_bundle --include-models`

Depois, crie um Kaggle Dataset a partir de `kaggle_bundle/`.

## 5) Ensemble (opcional)

Se você gerar múltiplas submissões (ex.: modelos/seed diferentes), pode ensemblar via:

- `python scripts/ensemble_submissions.py --data-root data/recodai --split test --subs sub1.csv sub2.csv --method weighted --scores 0.33 0.35 --out outputs/submission_ens.csv --diagnostics outputs/ensemble_diag.csv`

O `--diagnostics` grava um CSV por `case_id` (área final, nº de instâncias, etc.) para depuração.
