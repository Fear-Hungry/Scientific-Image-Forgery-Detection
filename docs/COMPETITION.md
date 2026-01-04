# Competição (Recod.ai/LUC — Scientific Image Forgery Detection)

Este documento reúne informações práticas sobre o **formato de dados**, **submission**, **RLE** e **métrica (oF1)**.

Fontes oficiais:

- Página da competição: https://www.kaggle.com/competitions/recodai-luc-scientific-image-forgery-detection
- Métrica oficial (código + RLE): https://www.kaggle.com/code/metric/recodai-f1

## Visão geral

Imagens científicas (microscopia, *western blots*, gráficos, etc.) são fundamentais para sustentar resultados em artigos. Porém, há casos de **manipulação de imagens** que podem levar a conclusões falsas.

Esta competição foca em um tipo específico e comum de fraude: **copy-move forgery** (*copia-e-cola dentro da própria imagem*), onde regiões são duplicadas para “fabricar” sinais/estruturas, esconder artefatos ou reforçar um achado.

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

> Observação: as máscaras são fornecidas para **todas** as imagens em `train_images/forged/`. Para imagens em `train_images/authentic/`, o rótulo é “sem forjamento” (submission deve ser `authentic`).

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

## Métrica: variante de F1 (oF1)

A competição usa uma variante de F1 por instância, com **matching ótimo** entre instâncias previstas e *ground truth* usando o **algoritmo Húngaro** (*Hungarian algorithm*).

Em alto nível:

1) Para cada par *(pred_i, gt_j)* calcula-se o **F1 pixel-a-pixel**.
2) Monta-se uma matriz `F1[i, j]`.
3) Encontra-se a atribuição 1-para-1 que **maximiza** o F1 (Húngaro).
4) Aplica-se uma penalidade para excesso de instâncias previstas (ver a implementação oficial para detalhes).

Casos “authentic”:

- Se `label == authentic` e `prediction == authentic` → score da imagem = **1**
- Caso contrário → score da imagem = **0**

O score final é a média do score por imagem.

Implementação exata: https://www.kaggle.com/code/metric/recodai-f1

## Regras e restrições do Kaggle (code competition)

Conforme regras divulgadas (resumo; **confira sempre a página oficial**):

- Submissões via **Notebooks**.
- Tempo limite típico: **4 horas** (CPU/GPU).
- **Internet desabilitada** no momento da submissão.
- Dados externos gratuitos e modelos pré-treinados são permitidos (ver regras oficiais).
- Arquivo final deve se chamar `submission.csv` ou `submission.parquet`.

## Dicas práticas (baseline e armadilhas comuns)

- **Instâncias importam:** as máscaras são `(N, H, W)`. Se seu modelo produzir uma única máscara, separe em instâncias via componentes conexos antes de codificar em RLE.
- **RLE “na marra” dá ruim:** use as funções oficiais do notebook `metric/recodai-f1` para evitar erro de ordem (*F-order*) e validações.
- **Aspas no CSV:** `annotation` com RLE costuma precisar de aspas no CSV.
- **Negative samples:** use `train_images/authentic` como negativos (`authentic`) para o classificador/decisor.
- **Sem vazamento:** se você usar pares `authentic`/`forged` do mesmo `case_id`, garanta que o *split* treino/validação respeite isso.

