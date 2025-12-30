# weights_cache

Pasta para armazenar pesos **pré-treinados** (offline) usados pelo pipeline/notebooks.

## Convenção
- Nome do arquivo: `<model_name>.pth` (ex.: `tf_efficientnet_b4_ns.pth`).
- Formato: `torch.save(model.state_dict(), ...)`.

## Kaggle (internet OFF)
- Anexe um Kaggle Dataset que contenha esta pasta, por exemplo em:
  - `/kaggle/input/<seu_dataset>/weights_cache/<model_name>.pth`
  - ou renomeie o dataset para `weights_cache` para usar `/kaggle/input/weights_cache/<model_name>.pth`

## Gerar pesos
- Use `scripts/download_timm_weights.py` para baixar e salvar os pesos no formato acima.
