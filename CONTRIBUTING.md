# Contribuindo

Obrigado por considerar contribuir!

## Regras gerais

- **Não versionar dados/artefatos grandes**: `data/`, `outputs/`, `runs/`, `logs/` ficam fora do Git (ver `.gitignore`).
- **Código do pacote** fica em `src/forgeryseg/`. Scripts em `scripts/` devem ser wrappers leves (sem duplicar lógica).
- **Notebooks**: implemente primeiro em `.py` e depois sincronize para `.ipynb` (ver `scripts/sync_ipynb_from_py.py`).

## Setup (dev)

```bash
pip install -r requirements.txt
pip install -e ".[dev]"
pre-commit install
```

## Checagens antes de abrir PR

```bash
make lint
make test
```

## Estilo

- Ruff (lint + format) é o padrão do repo.
- Prefira `pathlib.Path` para paths.
- Mantenha mudanças pequenas e focadas; evite refactors fora do escopo.

## Fluxo sugerido

1. Abra uma Issue descrevendo bug/feature (com comando para reproduzir, se aplicável).
2. Faça mudanças + testes.
3. Abra PR com:
   - resumo do que mudou
   - comandos para reproduzir
   - impactos em score/tempo (se aplicável)

