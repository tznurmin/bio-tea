# bio-tea-runner

For more involved fine-tuning experiments, this package configures, executes,
validates, and reports model comparisons over augmented datasets. It is separate
from the augmentation library, which can be used independently.

The runner supports configurable training matrices across tasks, dataset
variants, models, sets, random seeds, and hyperparameters. It also provides
calibration, resumable execution, artifact validation, and aggregate reporting.

Install from the package directory:

```bash
python -m pip install .
```

Console entry points include:

```bash
bio-tea-runner-exp1 --help
bio-tea-runner-exp2 --help
bio-tea-runner-exp3 --help
bio-tea-runner-report --help
bio-tea-runner-validate --help
bio-tea-runner-exp1-calibrate --help
```

The repository's [fine-tuning example](../../examples/minimal-exp1) demonstrates
the complete training and evaluation path with a small prepared dataset and
one-epoch configuration.
