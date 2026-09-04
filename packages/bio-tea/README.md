# Taxonomic Entity Augmentation (TEA)

TEA makes taxonomic names an explicit experimental variable in biological text.
The library switches species names and scrambles strain identifiers while
keeping token labels aligned. The resulting examples support model analysis and
provide additional training material for generalisation beyond the names
represented in the original dataset.

Within each matched pair, only the transformed names differ, allowing their
effect on model output to be compared directly.

The [BioBERT fine-tuning experiments](https://github.com/tznurmin/TEA_ft)
demonstrate the workflow on two curated datasets: Pathogen Identifier and Strain
Tagger. On evaluation examples whose taxonomic names were absent from training,
augmentation raised F1 from 26.2% to 59.7% for pathogen identification and from
57.2% to 70.6% for strain tagging.

For a separate word-level perturbation analysis, see the
[taxonomic perturbation experiments](https://github.com/tznurmin/TEA_perturb).

## Species substitution

Full and abbreviated species mentions are substituted consistently throughout
the text.

![Original article excerpt and two species-substituted versions shown side by side](https://raw.githubusercontent.com/tznurmin/bio-tea/main/assets/species-substitution.png)

## Strain scrambling

Strain identifiers are scrambled while retaining their format.

![Original strain description and one strain-scrambled version shown side by side](https://raw.githubusercontent.com/tznurmin/bio-tea/main/assets/strain-scrambling.png)

## Installation

The augmentation package is lightweight, with only one runtime dependency, and
installs directly from PyPI:

```bash
pip install taxonomic-entity-augmentation
```

## Basic usage

Any tokenizer that provides a `tokenize` method can be used. This example uses
whitespace tokenization and requires no model dependencies:

```python
from bio_tea import TEA


class WhitespaceTokenizer:
    def tokenize(self, text, **kwargs):
        return text.split()


tea = TEA(WhitespaceTokenizer(), rseed=42)

print(tea.switch("Escherichia coli was measured in culture."))
print(tea.scramble("The strain ATCC 25922 was included.", ["ATCC 25922"], force_diff=True))
```

Setting `rseed` makes species substitutions and strain scrambling reproducible.

With `transformers` installed, a Hugging Face model tokenizer can be supplied
directly:

```python
from transformers import AutoTokenizer
from bio_tea import TEA


tokenizer = AutoTokenizer.from_pretrained(
    "dmis-lab/biobert-base-cased-v1.2",
    do_lower_case=False,
)
tea = TEA(tokenizer, rseed=42)

print(tea.switch("Escherichia coli was measured in culture."))
```

## Labelled datasets

[TEA_curated_data](https://github.com/tznurmin/TEA_curated_data) provides curated
biological source texts and taxonomic entity annotations for generating
labelled datasets. The augmentation pipeline can extract complete sentence
windows within a token budget and produce aligned token labels for original,
species-switched, strain-scrambled, and combined examples.

### Command-line utilities

The package also provides utilities for inspecting and validating generated
example sets. These are not required for basic augmentation.

| Command | Purpose |
| --- | --- |
| `bio-tea-inspect` | Inspect or compare labelled examples |
| `bio-tea-sample` | Export representative examples from generated datasets |
| `bio-tea-validate` | Validate one generated example set |
| `bio-tea-stats` | Summarise labels and categories in an example set |
| `bio-tea-qa` | Validate generated example sets under a results directory |
| `bio-tea-manifest-compare` | Compare training variants recorded in a dataset manifest |

Run any command with `--help` for its arguments.

### Curated dataset example

Download v1.1 and select the extracted directory as the data source:

```bash
wget https://github.com/tznurmin/TEA_curated_data/archive/refs/tags/v1.1.tar.gz -qO - | tar -xz
mv TEA_curated_data-1.1 TEA_curated_data
export TEA_CURATED_ROOT="$PWD/TEA_curated_data"
```

`TEA_CURATED_DATA` is also accepted. The external dataset is separately
licensed.

## Licence

- Code: [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- Vendored UniProt organism list: [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/),
  with [data attribution](https://github.com/tznurmin/bio-tea/blob/main/packages/bio-tea/src/bio_tea/data/attribution.txt)
- TEA_curated_data is external and separately licensed
