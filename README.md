# Recod.ai/LUC — Scientific Image Forgery Detection (Kaggle)

Repositório com um *snapshot* dos dados da competição do Kaggle **“Recod.ai/LUC - Scientific Image Forgery Detection”** e um guia completo do problema, formato de submissão e métrica.

- Página oficial: https://www.kaggle.com/competitions/recodai-luc-scientific-image-forgery-detection
- Métrica oficial (código + RLE): https://www.kaggle.com/code/metric/recodai-f1

## Sumário

- [Visão geral](#visão-geral)
- [Cronograma e premiação](#cronograma-e-premiação)
- [O desafio](#o-desafio)
- [Dados (estrutura e formatos)](#dados-estrutura-e-formatos)
- [Notebooks (Kaggle)](#notebooks-kaggle)
- [Formato de submissão](#formato-de-submissão)
- [Run-Length Encoding (RLE)](#run-length-encoding-rle)
- [Métrica: variante de F1 (oF1)](#métrica-variante-de-f1-of1)
- [Regras e restrições do Kaggle (code competition)](#regras-e-restrições-do-kaggle-code-competition)
- [Como baixar os dados via Kaggle API](#como-baixar-os-dados-via-kaggle-api)
- [Dicas práticas (baseline e armadilhas comuns)](#dicas-práticas-baseline-e-armadilhas-comuns)

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

Este repositório contém os dados em `data/`:

```
data/
  recodai-luc-scientific-image-forgery-detection.zip
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

### Estatísticas do snapshot deste repo

- `train_images/authentic`: **2.377** imagens
- `train_images/forged`: **2.751** imagens (**todas com máscara** em `train_masks/`)
- `supplemental_images`: **48** imagens (com `supplemental_masks/`)
- `test_images`: **1** imagem (exemplo; o conjunto de teste real é oculto no ambiente do Kaggle)

### Imagens

- Formato: `*.png`
- Dimensões: variam bastante (no snapshot: de **64×74** até **3888×3888** no treino; *supplemental* pode ter dimensões ainda maiores).

### Máscaras (ground truth)

- Formato: `*.npy` (NumPy)
- Tipo: `uint8` (valores **0/1**)
- **Shape:** `(N, H, W)`
  - `N` = número de **instâncias** de regiões copiadas naquela imagem (no snapshot existem exemplos com `N=1,2,3,...`).
  - `H, W` batem com a imagem correspondente em `train_images/forged/<case_id>.png`.

> Observação: as máscaras são fornecidas para **todas** as imagens em `train_images/forged/`. Para imagens em `train_images/authentic/`, o rótulo é “sem forjamento” (submissão deve ser `authentic`).

## Notebooks (Kaggle)

Notebooks prontos para uso no Kaggle (internet OFF no submit). Todos importam o código do projeto em `src/forgeryseg/`.

- `notebooks/fase_00_pipeline_unico_kaggle.ipynb`: **tudo-em-um** (setup → treino opcional → inferência → `submission.csv`).
- `notebooks/fase_01_setup_offline_kaggle.ipynb`: valida instalação offline via `recodai_bundle/wheels/*.whl`.
- `notebooks/fase_03_treino_segmentacao_kaggle.ipynb`: treino do segmentador (salva em `outputs/models_seg/<model_id>/fold_<k>/best.pt`).
- `notebooks/fase_04_inferencia_submissao_kaggle.ipynb`: inferência/submissão (requer checkpoints em `outputs/models_seg/...`; pode anexar via Kaggle Dataset).

Fluxo típico (Kaggle):

1) Anexe o dataset da competição (`recodai-luc-scientific-image-forgery-detection`).
2) (Opcional) Importe este repositório como Kaggle Dataset (via GitHub) para ter `recodai_bundle/wheels/` e `src/` disponíveis offline.
3) Se treinar, os checkpoints ficam em `/kaggle/working/outputs/...`. Para reutilizar, use “Save & Create Dataset” a partir do notebook e anexe o dataset de outputs depois.

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
