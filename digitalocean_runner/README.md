# Clean DigitalOcean WDBC benchmark

This directory is a standalone CPU-only implementation of the fixed WDBC experiment. It validates the canonical CSV hash, prevents fold leakage, caches each training realization once for reuse across classifiers, and writes atomic per-task checkpoints for safe resume. Run commands from this directory.

## Local setup

Ubuntu 24.04 and Python 3.12 are the deployment targets.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest tests
```

The input must be the canonical LF CSV with SHA-256 `27f219231dbb30eecbfc1361407ed641ea01be43316e2c707a1baf82c9795e23`. The file is not committed.

## Run and resume

Full fixed design (3,600 classifier tasks):

```bash
python run_benchmark.py --dataset /path/to/wdbc_canonical_lf.csv --output results
```

Resume after interruption (valid completed tasks are skipped even without the flag):

```bash
python run_benchmark.py --dataset /path/to/wdbc_canonical_lf.csv --output results --resume
```

Check progress without loading data or running science:

```bash
python run_benchmark.py --output results --status
```

Analyze only after all 3,600 tasks exist:

```bash
python analyze_results.py --results results
```

`--allow-partial` is available for engineering inspection only; partial output is not final scientific output.

## Future DigitalOcean execution

After copying/cloning the project and installing its environment on the server:

```bash
tmux new -s bc35
source .venv/bin/activate
python run_benchmark.py --dataset /path/to/wdbc_canonical_lf.csv --output results
```

Detach with `Ctrl+B`, then `D`. Reattach later with:

```bash
tmux attach -t bc35
```

Use the resume command after any interrupted process. Result synchronization (for example with `rclone`) is external to these scripts.
