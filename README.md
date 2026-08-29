# Min-p-CoT

Experimental code and evaluation artifacts for applying fixed, stage-wise, and entropy-adaptive Min-p sampling to chain-of-thought reasoning.

## Project contents

- `min_p_cot.py`: stage-wise Min-p chain-of-thought generation and voting.
- `grid_search_minp.py`: grid search over stage-specific Min-p values.
- `uniform_minp_search.py`: uniform Min-p parameter search.
- `analyze_best_minp.py`: experiment analysis and comparison.
- `dynamics_minp.py`: entropy-adaptive dynamic Min-p generation and evaluation.
- `extractor.py`: answer extraction for mathematical and QA tasks.
- `ASPRM-main/`: ASPRM training and evaluation code and sample datasets.
- `outputs/`, `grid_search_results/`, `uniform_minp_search_results/`, `dynamics_minp/`, and `logs/`: retained experiment artifacts.

## Experiments represented in this repository

- Models: Qwen2.5-3B-Instruct and Llama-3.2-3B-Instruct.
- Datasets: GSM8K, MATH algebra test data, MATH500, and included ASPRM evaluation data.
- Methods: no-CoT, four-stage CoT, majority voting, static Min-p search, and entropy-adaptive dynamic Min-p.

Model weights and the main external datasets are not stored in this repository. The configuration files reference the original server-side model and dataset paths and should be adjusted for a new environment.

## Installation

```bash
pip install -r requirements.txt
```

Review the relevant JSON configuration before running an experiment.
