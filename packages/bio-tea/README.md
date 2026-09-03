# Taxonomic Entity Augmentation (TEA)

TEA provides taxonomic entity augmentation for biological texts. It supports
species-name substitution, strain-name scrambling, sentence windowing,
token-budget handling, and token-level label materialisation from curated data.

## Installation

```bash
python -m pip install taxonomic-entity-augmentation
```

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

The package includes command-line utilities for generated TEA example sets:

```bash
bio-tea-inspect --help
bio-tea-sample --help
bio-tea-validate --help
bio-tea-stats --help
bio-tea-qa --help
bio-tea-manifest-compare --help
```

Utilities that operate on curated TEA source data require
[TEA_curated_data](https://github.com/tznurmin/TEA_curated_data). Download v1.1
and point TEA to the extracted directory:

```bash
wget https://github.com/tznurmin/TEA_curated_data/archive/refs/tags/v1.1.tar.gz -qO - | tar -xz
mv TEA_curated_data-1.1 TEA_curated_data
export TEA_CURATED_ROOT="$PWD/TEA_curated_data"
```

`TEA_CURATED_DATA` is also accepted. TEA_curated_data is external to TEA and
separately licensed.

## Licence

- Code: [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- Vendored UniProt organism list: [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/),
  with [data attribution](https://github.com/tznurmin/bio-tea/blob/main/packages/bio-tea/src/bio_tea/data/attribution.txt)
- TEA_curated_data is external and separately licensed
