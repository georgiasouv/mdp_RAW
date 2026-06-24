# mdp — project structure & run commands

Top-level-rooted package. Install once, then run modules from the project root.

## One-time setup
```bash
cd ~/Desktop/multidetector_tom
pip install -e .        # registers the `mdp` package (editable; edits take effect live)
```
This is what makes every import work from ANY directory / terminal / SLURM job.
You only re-run it if you change pyproject.toml (e.g. add a new top-level subpackage).

## Layout
```
multidetector_tom/            <- project root == package container
├── pyproject.toml            <- declares the package (enables pip install -e .)
├── core/                     <- the harness (changes rarely)
│   ├── preprocessor.py       <- the learnable RAW->RGB module
│   ├── adapters.py           <- frozen-detector wrappers (FCOS/FRCNN/RetinaNet/DETR)
│   ├── conflict.py           <- gradient-cosine diagnostic
│   ├── train.py              <- train one preprocessor / one regime / one dataset
│   └── evaluate.py           <- mAP for a (trained prep, eval detector) pair
├── datasets/                 <- one module per dataset (grows to 5)
│   ├── dataset.py            <- generic resize + collate
│   ├── registry.py           <- name -> dataset module dispatcher
│   └── pascalraw.py          <- PASCALRAW loader (DONE)
├── dataprep/                 <- run-once-per-dataset label tools
│   ├── verify_labels.py      <- visual box-on-image check
│   └── convert_pascalraw.py  <- XML -> packed-space COCO json
├── results/
│   └── build_readout.py      <- jsons -> matrix.csv + leave-one-out.csv
├── dev/                      <- scaffolding (not the real pipeline)
│   ├── synthetic_registry.py <- fake-data shim for smoke tests
│   ├── run_smoke.py          <- smoke test w/ COCO weights (use on your GPU box)
│   └── run_smoke_sandbox.py  <- smoke test w/ random init (no downloads)
└── launch.slurm
```

## Run commands (always from project root, as MODULES with -m)
```bash
# smoke test on fake data (first, to confirm plumbing)
python -m dev.run_smoke solo_1

# real training: one regime x one dataset
python -m core.train --regime solo_1 --dataset pascalraw --num-classes 3 \
    --out runs/pascalraw__solo_1

# evaluate a trained preprocessor in front of one detector
python -m core.evaluate --prep runs/pascalraw__solo_1/preprocessor.pth \
    --detector fcos --dataset pascalraw --num-classes 3 \
    --out results_json/pascalraw__solo_1__fcos.json

# build the paper tables
python -m results.build_readout --results results_json/ --out readout
```

## Adding a dataset later
1. Write `datasets/<name>.py` exposing `build_records(name)`.
2. Add a branch in `datasets/registry.py`.
That's it — no other file changes, no re-install (same subpackage).

## Note on the `datasets` name
Our package has a `datasets/` subpackage. If you also use HuggingFace
`datasets`, there COULD be a name clash. Inside this project, `import
datasets.X` resolves to OURS (local package wins). If you ever need HF
datasets here, import it in an isolated module or alias it.
