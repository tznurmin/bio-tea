# Usage example

This example runs a one-epoch fine-tuning job against a small prepared dataset.
It uses BioLinkBERT base, `michiyasunaga/BioLinkBERT-base`.

From the project root:

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

PyTorch sends the run to CUDA automatically when a CUDA-enabled PyTorch install
and visible GPU are available.
