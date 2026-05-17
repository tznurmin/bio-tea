# Taxonomic Entity Augmentation (TEA)

TEA provides taxonomic entity augmentation for biomedical NLP datasets. It
supports species-name substitution, strain-name scrambling, sentence windowing,
token-budget handling, and label materialisation for TEA curation data.

## Contents

- `packages/bio-tea`: augmentation library
- `packages/bio-tea-runner`: training/reporting runner utilities
- `examples/minimal-exp1`: a small prepared dataset and config for a usage example
- `packages/bio-tea/src/bio_tea/data/species.txt`: vendored UniProt-derived
  organism list used for species-name sampling

## Installation

From the project root:

```bash
python -m pip install ./packages/bio-tea
python -m pip install ./packages/bio-tea-runner
```

The runner package installs the Hugging Face training stack declared in
`packages/bio-tea-runner/pyproject.toml`. GPU execution requires a CUDA-enabled
PyTorch installation for the local system.

## Augmentation example

```python
from bio_tea import TEA


class WhitespaceTokenizer:
    def tokenize(self, text, **kwargs):
        return text.split()


tea = TEA(WhitespaceTokenizer(), rseed=42)

print(tea.switch("Escherichia coli was measured in culture."))
print(tea.scramble("The strain ATCC 25922 was included.", ["ATCC 25922"], force_diff=True))
```

## Dataset utility commands

The augmentation package includes command-line utilities for generated TEA
example sets:

```bash
bio-tea-inspect --help
bio-tea-sample --help
bio-tea-validate --help
bio-tea-stats --help
bio-tea-qa --help
bio-tea-manifest-compare --help
```

Utilities that operate on curated TEA source data require
[TEA_curated_data](https://github.com/tznurmin/TEA_curated_data). Download
v1.1 separately and point TEA to the extracted directory:

```bash
wget https://github.com/tznurmin/TEA_curated_data/archive/refs/tags/v1.1.tar.gz -qO - | tar -xz
mv TEA_curated_data-1.1 TEA_curated_data
export TEA_CURATED_ROOT="$PWD/TEA_curated_data"
```

`TEA_CURATED_DATA` is also accepted. TEA_curated_data is external to TEA and
separately licensed.

## Usage example

The `examples/minimal-exp1` directory contains a small prepared dataset and
config for one BioLinkBERT base fine-tuning run
(`michiyasunaga/BioLinkBERT-base`):

```bash
bio-tea-runner-exp1 \
  --config examples/minimal-exp1/config.yaml \
  --source-root examples/minimal-exp1/source \
  --tasks example \
  --sets set1 \
  --seeds 1 \
  --epochs 1 \
  --summary-out examples/minimal-exp1/results/summary.json \
  --report-out examples/minimal-exp1/results/report.json
```

PyTorch uses CUDA automatically when a CUDA-enabled PyTorch install and visible
GPU are available.

Main outputs:

```text
examples/minimal-exp1/results/summary.json
examples/minimal-exp1/results/report.json
```

Detailed run artifacts are written under:

```text
examples/minimal-exp1/results/profiles/
```

The example trains and evaluates the model but does not save fine-tuned model
weights.

## Licence

- Code: [Apache License 2.0](LICENSE).
- Vendored UniProt organism list:
  [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).
  Attribution is provided in
  [data attribution file](packages/bio-tea/src/bio_tea/data/attribution.txt).
- Downloaded [TEA_curated_data](https://github.com/tznurmin/TEA_curated_data)
  material is external and separately licensed.
