# quantem.data

Python tooling for preparing and validating QuantEM datasets before publication.

## Role

This repo is the **code/tooling side** of the split:

- metadata parsing
- dataset synchronization
- repository helpers

Published dataset payloads live in the separate Hugging Face dataset repo:

- `bobleesj/quantem-data`

## Local layout

```text
quantem.data/
  pyproject.toml
  src/quantem/data/
  scripts/
```

Use this repo to prepare dataset folders locally, then publish the dataset
payload from the separate `quantem-data` checkout.
