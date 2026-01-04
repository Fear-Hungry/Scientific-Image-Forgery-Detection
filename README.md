# Recod.ai/LUC — Scientific Image Forgery Detection (Kaggle)

Repositório com **pipeline + utilitários** para a competição do Kaggle **“Recod.ai/LUC - Scientific Image Forgery Detection”** (segmentação + RLE).  
Os dados **não são versionados** aqui: o projeto espera o layout do Kaggle em `data/recodai` (veja “Como baixar os dados via Kaggle API”).

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

## Sumário

- [Visão geral](#visão-geral)
- [Cronograma e premiação](#cronograma-e-premiação)
- [O desafio](#o-desafio)
  - [Entrada](#entrada)
  - [Saída esperada](#saída-esperada)
  - [Por que é difícil](#por-que-é-difícil)
- [Dados (estrutura e formatos)](#dados-estrutura-e-formatos)
  - [Imagens](#imagens)
  - [Máscaras (ground truth)](#máscaras-ground-truth)
- [Notebooks (Kaggle)](#notebooks-kaggle)
- [Formato de submissão](#formato-de-submissão)
- [Run-Length Encoding (RLE)](#run-length-encoding-rle)
- [Métrica: variante de F1 (oF1)](#métrica-variante-de-f1-of1)
- [Regras e restrições do Kaggle (code competition)](#regras-e-restrições-do-kaggle-code-competition)
- [Como baixar os dados via Kaggle API](#como-baixar-os-dados-via-kaggle-api)
- [Dicas práticas (baseline e armadilhas comuns)](#dicas-práticas-baseline-e-armadilhas-comuns)
- [Pipeline (DINOv2 + CNN) — scripts deste repo](#pipeline-dinov2--cnn--scripts-deste-repo)

## Visão geral

Imagens científicas (microscopia, *western blots*, gráficos, etc.) são fundamentais para sustentar resultados em artigos. Porém, há casos de **manipulação de imagens** que podem levar a conclusões falsas.

Esta competição foca em um tipo específico e comum de fraude: **copy-move forgery** (*copia-e-cola dentro da própria imagem*), onde regiões são duplicadas para “fabricar” sinais/estruturas, esconder artefatos ou reforçar um achado.

O conjunto de dados é descrito como composto por **forjamentos confirmados** extraídos de **milhares (2.000+) de artigos retratados**, buscando refletir cenários realistas de fraude em imagens biomédicas.

O objetivo é desenvolver um modelo que:

1) **Detecte** se uma imagem é **autêntica** ou possui **forjamento** (copy-move), e  
2) **Segmente** (em nível de pixel) as **regiões copiadas** (*múltiplas instâncias podem existir na mesma imagem*).

## Cronograma e premiação

Resumo (confira sempre a página oficial para a versão definitiva):

- Início: **23/out/2025**
- Deadline de entrada e *team merger*: **08/jan/2026**
- Deadline final de submissão: **15/jan/2026**
- Fase de *forecasting* / encerramento: **06/mai/2026**
- Premiação total: **US$ 55.000** (top 5)

A competição é organizada/hospedada pela **Recod.ai** e divulgada como suportada pelo **Kaggle Research Grant Program** e pelo **IEEE Signal Processing Society Challenge Program**.

## O desafio

### Entrada

Uma imagem (PNG) de contexto biomédico/científico.

### Saída esperada

- Se **não houver** evidência de copy-move: retornar a string **`authentic`**.
- Se **houver** forjamento: retornar **máscaras binárias** (podendo haver mais de uma instância), serializadas como **RLE** e concatenadas por `;`.

### Por que é difícil

- Diversidade de formatos e resoluções (figuras compostas, múltiplos painéis, artefatos de compressão).
- Copy-move tende a ser sutil: a região copiada vem da própria imagem e, portanto, pode ter aparência quase idêntica ao restante.
- Falsificadores podem aplicar transformações e pós-processamentos (rotação/escala, borrão, ruído, ajustes de brilho/contraste, compressão JPEG).
- Forjamentos podem ser pequenos e sutis.
- Uma imagem pode conter **várias instâncias** de regiões duplicadas.
- A avaliação considera **matching ótimo** entre instâncias previstas e *ground truth* (não é apenas IoU de uma máscara “única”).

## Dados (estrutura e formatos)

O projeto espera os dados em `data/recodai`:

```
data/
  recodai/
    sample_submission.csv
    train_images/
      authentic/   # imagens autênticas (negativas)
      forged/      # imagens forjadas (positivas)
    train_masks/   # máscaras (somente para forged)
    supplemental_images/
    supplemental_masks/
    test_images/   # amostra pública (em code competitions o teste real é oculto)
```

### Imagens

- Formato: `*.png`
- Dimensões: variam bastante (há imagens pequenas e outras com milhares de pixels por lado).

### Máscaras (ground truth)

- Formato: `*.npy` (NumPy)
- Tipo: `uint8` (valores **0/1**)
- **Shape:** `(N, H, W)`
  - `N` = número de **instâncias** de regiões copiadas naquela imagem.
  - `H, W` batem com a imagem correspondente em `train_images/forged/<case_id>.png`.

> Observação: as máscaras são fornecidas para **todas** as imagens em `train_images/forged/`. Para imagens em `train_images/authentic/`, o rótulo é “sem forjamento” (submissão deve ser `authentic`).

## Notebooks (Kaggle)

Notebooks prontos para uso no Kaggle. A ideia é manter **a lógica em `src/` + `scripts/`** e deixar o notebook apenas como “orquestrador”.

- `notebooks/fase_00_submissao_kaggle.ipynb`: **submissão** (inferência + geração do `submission.csv`, pensado para internet OFF).
- `notebooks/fase_00_submissao_kaggle.py`: a mesma lógica (fonte), para versionar e revisar diffs.

Fluxo típico (Kaggle):

1) Anexe o dataset da competição (`recodai-luc-scientific-image-forgery-detection`).
2) Anexe um Kaggle Dataset com este repo (pelo menos `src/` e `configs/`), ou copie o código para o notebook.
3) (Opcional, mas necessário para inferência com pesos) Anexe um Kaggle Dataset com seus checkpoints em `outputs/models/*.pth` (mesmos paths usados em `configs/*.json`).
4) Rode `notebooks/fase_00_submissao_kaggle.ipynb` para gerar `/kaggle/working/submission.csv`.

## Formato de submissão

Arquivo: `submission.csv` (ou `submission.parquet`, conforme as regras do Kaggle) com colunas:

```csv
case_id,annotation
45,authentic
```

Regras:

- Uma linha por `case_id` (imagem).
- Se a imagem for **autêntica**: `annotation = authentic`
- Se houver forjamento:
  - `annotation` deve conter **uma ou mais** instâncias em **RLE**
  - instâncias são separadas por `;`
  - cada instância é uma **lista JSON** de inteiros `[start, length, start, length, ...]`

Exemplo (ilustrativo):

```csv
case_id,annotation
1,authentic
2,"[123, 4, 200, 3];[1000, 10]"
```

> Dica: como o conteúdo contém vírgulas e colchetes, é comum precisar de aspas (`"..."`) no CSV para não quebrar o parsing.

## Run-Length Encoding (RLE)

O RLE oficial está definido no notebook da métrica (link no topo). Pontos importantes:

- A codificação é feita em **ordem coluna-major** (*Fortran order*), equivalente a `mask.flatten(order="F")`.
- Os índices de `start` são **1-based** (primeiro pixel é `1`).
- Cada instância é uma lista de pares `(start, length)` concatenados: `[start1, len1, start2, len2, ...]`.
- A string final do `annotation` é:
  - `authentic` **ou**
  - `json_rle_instancia_1 + ';' + json_rle_instancia_2 + ...`

Trecho (essência) do encoder oficial:

```python
def rle_encode(masks):
    # masks: lista de arrays (H, W) com valores 0/1
    def _encode_single(x):
        dots = np.where(x.T.flatten() == 1)[0]  # ordem F
        run_lengths = []
        prev = -2
        for b in dots:
            if b > prev + 1:
                run_lengths.extend((b + 1, 0))  # 1-based
            run_lengths[-1] += 1
            prev = b
        return run_lengths

    return ';'.join([json.dumps(_encode_single(x)) for x in masks])
```

Para detalhes (incluindo *decode* e validações), use o notebook oficial: https://www.kaggle.com/code/metric/recodai-f1

## Métrica: variante de F1 (oF1)

A competição usa uma variante de F1 por instância, com **matching ótimo** entre instâncias previstas e *ground truth* usando o **algoritmo Húngaro** (*Hungarian algorithm*).

Em alto nível:

1) Para cada par *(pred_i, gt_j)* calcula-se o **F1 pixel-a-pixel**.
2) Monta-se uma matriz `F1[i, j]`.
3) Encontra-se a atribuição 1-para-1 que **maximiza** o F1 médio (Húngaro).
4) Aplica-se uma penalidade para excesso de instâncias previstas:

```
penalty = len(gt) / max(len(pred), len(gt))
score_imagem = mean(F1 dos pares atribuídos) * penalty
```

Casos “authentic”:

- Se `label == authentic` e `prediction == authentic` → score da imagem = **1**
- Caso contrário (um dos dois “authentic” e o outro não) → score da imagem = **0**

O score final é a média do score por imagem.

Implementação exata: https://www.kaggle.com/code/metric/recodai-f1

## Regras e restrições do Kaggle (code competition)

Conforme regras divulgadas (resumo):

- Submissões via **Notebooks**.
- Tempo limite típico: **4 horas** (CPU/GPU).
- **Internet desabilitada** no momento da submissão.
- Dados externos gratuitos e modelos pré-treinados são permitidos (ver regras oficiais).
- Arquivo final deve se chamar `submission.csv` ou `submission.parquet`.

## Como baixar os dados via Kaggle API

Pré-requisito: configurar a Kaggle API com `kaggle.json` (token) em `~/.kaggle/kaggle.json`.

Baixar:

```bash
kaggle competitions download -c recodai-luc-scientific-image-forgery-detection -p data
```

Extrair:

```bash
unzip -q data/recodai-luc-scientific-image-forgery-detection.zip -d data/recodai
```

> Observação: em *code competitions*, a pasta `test_images/` local pode conter apenas um subconjunto/placeholder. A avaliação real usa um conjunto de teste oculto no runtime do Kaggle.

## Dicas práticas (baseline e armadilhas comuns)

- **Instâncias importam:** as máscaras são `(N, H, W)`. Se seu modelo produzir uma única máscara (segmentação sem instâncias), considere separar em instâncias via componentes conexos (*connected components*) antes de codificar em RLE.
- **RLE “na marra” dá ruim:** use as funções oficiais do notebook `metric/recodai-f1` para evitar erro de ordem (*F-order*) e validações (starts em ordem crescente, etc.).
- **Aspas no CSV:** `annotation` com RLE costuma precisar de aspas no CSV.
- **Negative samples:** use `train_images/authentic` como negativos (`authentic`) para o classificador/decisor.
- **Modelos possíveis:** U-Net/DeepLab + pós-processamento; Mask R-CNN/instance segmentation; abordagens baseadas em *self-similarity* para copy-move; *Siamese* usando pares (quando existir `authentic`/`forged` com mesmo `case_id`).
- **Sem vazamento:** se você usar pares `authentic`/`forged` do mesmo `case_id`, garanta que *split* treino/validação respeite isso para não inflar métricas.

---

Este README é um guia derivado da proposta e do formato oficial da competição. Para informações legais e definitivas (regras, elegibilidade, licenças, prazos), consulte a página do Kaggle: https://www.kaggle.com/competitions/recodai-luc-scientific-image-forgery-detection

---

## Pipeline (DINOv2 + CNN) — scripts deste repo

O estado atual do repo é um pipeline **simples e reproduzível**:

- **Inferência (submissão):** `scripts/predict_submission.py` (ou `notebooks/fase_00_submissao_kaggle.ipynb` no Kaggle).
- **Pós-processamento:** `src/forgeryseg/postprocess.py` (threshold + filtros + regras de `authentic`).
- **Complementos opcionais:** `fft_gate` e `dinov2_freq_fusion` (seções abaixo).

Configurações prontas em `configs/` e CLIs em `scripts/`:

- Gerar submissão individual:
  - `python scripts/predict_submission.py --config configs/dino_v1_718_u52.json --data-root data/recodai --split test --out outputs/submission1.csv`
  - `python scripts/predict_submission.py --config configs/dino_v2_518_basev1.json --data-root data/recodai --split test --out outputs/submission2.csv`
  - `python scripts/predict_submission.py --config configs/dino_v3_518_r69.json --data-root data/recodai --split test --out outputs/submission3.csv`
- Ensemble ponderado (a partir das submissões):
  - `python scripts/ensemble_submissions.py --data-root data/recodai --split test --subs outputs/submission1.csv outputs/submission2.csv outputs/submission3.csv --scores 0.324 0.323 0.322 --method weighted --out outputs/submission.csv`
- Sanidade (oracle + baseline all-authentic no treino):
  - `python scripts/sanity_submissions.py --data-root data/recodai --out-dir outputs/sanity --split train`
- Treinar modelo de segmentação (salva checkpoint em `--out`):
  - `python scripts/train_dino_decoder.py --config configs/dino_v3_518_r69.json --data-root data/recodai --out outputs/models/r69.pth --epochs 5`

Os caminhos de checkpoints em `configs/*.json` (ex.: `outputs/models/r69.pth`) **devem existir** no ambiente de execução (ex.: anexando um Kaggle Dataset com `outputs/models/`).
Veja `docs/MODEL_ZOO.md` para o layout esperado de `outputs/models/`.

### FFT como sinal complementar (opcional)

Este repo também inclui um *gate* opcional baseado em FFT para reforçar decisões `authentic` vs `forged` quando o pós-processamento zera a máscara:

- Treinar um classificador FFT (log-magnitude):
  - `python scripts/train_fft_classifier.py --config configs/fft_classifier_logmag_256.json --data-root data/recodai --out outputs/models/fft_cls.pth`
- (Opcional) Treinar um classificador FFT (phase-only):
  - `python scripts/train_fft_classifier.py --config configs/fft_classifier_phase_only_256.json --data-root data/recodai --out outputs/models/fft_cls_phase.pth`
- Rodar submissão com `fft_gate` habilitado (exemplo):
  - `python scripts/predict_submission.py --config configs/dino_v3_518_r69_fft_gate.json --data-root data/recodai --split test --out outputs/submission_fft_gate.csv`

### Fusão espacial + frequência (opcional)

Além do *gate* (classificador), existe um modelo `dinov2_freq_fusion` que extrai uma representação FFT no `forward()` e faz fusão com os tokens do encoder antes do decoder. Exemplo:

- `python scripts/predict_submission.py --config configs/dino_v3_518_r69_freq_fusion.json --data-root data/recodai --split test --out outputs/submission_freq_fusion.csv`

Modos suportados em `freq_fusion.mode`: `logmag`, `hp_residual`, `phase_only`, `lp_hp`.
