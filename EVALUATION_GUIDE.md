# Medical Multimodal Research - Evaluation Guide

## Quick Start

### Run Complete Pipeline
```bash
# Full pipeline (training + KB + evaluation)
python main.py pipeline

# Skip training, run KB + evaluation only
python main.py pipeline --skip-training

# Run specific stages
python main.py pipeline --stages train eval
```

### Run Individual Evaluations

#### 1. Retrieval Evaluation
```bash
python main.py eval-retrieval
python main.py eval-retrieval --kb-dir outputs/kb/kb_final_v2
```

**Outputs:**
- `outputs/evaluation/retrieval/results.json` - Full results
- `outputs/evaluation/retrieval/summary.txt` - Human-readable summary

**Metrics Computed:**
- Recall@K (K=1,5,10,20)
- Precision@K
- Mean Reciprocal Rank (MRR)
- Mean Average Precision (MAP)
- NDCG@K
- Per-diagnosis breakdown
- Mode comparison (text, image, fusion)

#### 2. Encoder & Fusion Evaluation
```bash
python main.py eval-encoders
```

**Tests:**
- Embedding quality (normalization, determinism, dimensions)
- Fusion gate behavior and statistics
- Modality alignment
- Output normalization verification

#### 3. Counterfactual Stability Evaluation
```bash
python main.py eval-counterfactual
python main.py eval-counterfactual --num-samples 100
```

**Tests:**
- Stability under modality removal (no text, no image)
- Robustness to noise
- JS divergence computation
- Per-diagnosis stability analysis

#### 4. LoRA Fine-tuning Impact
```bash
python main.py eval-lora
```

**Comparisons:**
- Base vs LoRA embeddings (L2 distance, cosine similarity)
- Alignment improvement
- Per-query analysis

## Custom Pipeline Configuration

Create a YAML config file:

```yaml
# my_experiment.yaml
training:
  lora:
    enabled: true
    epochs: 2
  fusion:
    enabled: true
    epochs: 5

evaluation:
  retrieval:
    enabled: true
  counterfactual:
    num_samples: 100
```

Run with config:
```bash
python main.py pipeline --config my_experiment.yaml
```

## Output Structure

```
outputs/
├── experiments/
│   └── 20231225_120000/          # Timestamped experiment
│       ├── config.json            # Used configuration
│       ├── pipeline_results.json  # Stage results
│       └── logs/
├── evaluation/
│   ├── retrieval/
│   │   ├── results.json
│   │   └── summary.txt
│   ├── encoders/
│   │   ├── results.json
│   │   └── summary.txt
│   ├── counterfactual/
│   │   ├── results.json
│   │   └── summary.txt
│   └── lora/
│       ├── results.json
│       └── summary.txt
└── models/
    ├── trained_lora/
    └── trained_fusion/
```

## Interpreting Results

### Retrieval Metrics
- **R@1**: Top-1 accuracy (most important for diagnosis)
- **R@5**: Recall at 5 (captures similar cases)
- **MRR**: Average reciprocal rank of first relevant result
- **MAP**: Average precision across all relevant items
- **NDCG**: Normalized discounted cumulative gain

**Good Values:**
- R@1 > 0.6 (strong)
- R@5 > 0.8 (strong)
- MRR > 0.7 (strong)

### Counterfactual Stability
- **JS Divergence < 0.15**: High stability
- **JS Divergence 0.15-0.3**: Medium stability
- **JS Divergence > 0.3**: Low stability

**Interpretation:**
- Low divergence = predictions robust to perturbations
- High divergence = predictions sensitive/uncertain

### LoRA Impact
- **Positive improvement**: LoRA helps domain adaptation
- **Distance > 0.5**: Significant embedding space shift
- **Similarity < 0.9**: Meaningful fine-tuning occurred

## Research Paper Grade Metrics

This evaluation framework provides:

1. **Standard IR Metrics**: R@K, P@K, MRR, MAP, NDCG
2. **Robustness Analysis**: Counterfactual stability, perturbation sensitivity
3. **Ablation Studies**: Mode comparison, component analysis
4. **Fine-tuning Impact**: Base vs adapted model comparison
5. **Per-Category Analysis**: Breakdown by diagnosis
6. **Statistical Significance**: Multiple runs, confidence intervals

## Advanced Usage

### Custom Metric Calculation

Add custom metrics in `scripts/evaluation/retrieval/metrics.py`:

```python
def custom_metric(self, hits: List[int]) -> float:
    # Your metric logic
    return score
```

### Batch Evaluation

```bash
# Run multiple experiments
for config in configs/*.yaml; do
    python main.py pipeline --config $config
done
```

### Export for Analysis

```python
import json
import pandas as pd

# Load results
with open('outputs/evaluation/retrieval/results.json') as f:
    results = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(results['modes'])
df.to_csv('retrieval_analysis.csv')
```

## Troubleshooting

### "KB not found"
- Run KB building first: `python main.py build-kb`
- Or skip to evaluation: `python main.py pipeline --skip-training`

### "LoRA model not available"
- Train LoRA first: `python main.py train-lora`
- Or skip LoRA evaluation in config

### "Out of memory"
- Reduce batch size in configs
- Use CPU: Add `--device cpu` to commands

## Citation

If you use this evaluation framework in your research:

```bibtex
@software{medical_multimodal_eval,
  title={Medical Multimodal Research Evaluation Framework},
  year={2025},
  publisher={GitHub},
}
```
"""

# ==============================================================================
# INSTALLATION SUMMARY
# ==============================================================================
"""
FILE STRUCTURE SUMMARY
======================

Main Entry Points:
- main.py (UPDATED with new commands)

Pipeline:
- scripts/pipeline/__init__.py (new)
- scripts/pipeline/run_pipeline.py (new)
- scripts/pipeline/config.py (new)
- scripts/pipeline/orchestrator.py (new)

Evaluation - Retrieval:
- scripts/evaluation/retrieval/__init__.py (new)
- scripts/evaluation/retrieval/run_eval.py (new)
- scripts/evaluation/retrieval/evaluator.py (new)
- scripts/evaluation/retrieval/metrics.py (new)
- scripts/evaluation/retrieval/modes.py (new)
- scripts/evaluation/retrieval/analysis.py (new)

Evaluation - Encoders:
- scripts/evaluation/encoders/__init__.py (new)
- scripts/evaluation/encoders/run_eval.py (new)
- scripts/evaluation/encoders/evaluator.py (new)
- scripts/evaluation/encoders/embedding_quality.py (new)
- scripts/evaluation/encoders/fusion_analysis.py (new)
- scripts/evaluation/encoders/modality_alignment.py (new)

Evaluation - Counterfactual:
- scripts/evaluation/counterfactual/__init__.py (new)
- scripts/evaluation/counterfactual/run_eval.py (new)
- scripts/evaluation/counterfactual/evaluator.py (new)
- scripts/evaluation/counterfactual/stability_tester.py (new)
- scripts/evaluation/counterfactual/robustness_analyzer.py (new)

Evaluation - LoRA:
- scripts/evaluation/lora/__init__.py (new)
- scripts/evaluation/lora/run_eval.py (new)
- scripts/evaluation/lora/evaluator.py (new)
- scripts/evaluation/lora/embedding_comparison.py (new)
- scripts/evaluation/lora/alignment_improvement.py (new)

Config:
- configs/pipeline_config_example.yaml (new)

Documentation:
- EVALUATION_GUIDE.md (new)

USAGE EXAMPLES
==============

# Full pipeline
python main.py pipeline

# Individual evaluations
python main.py eval-retrieval
python main.py eval-encoders
python main.py eval-counterfactual --num-samples 100
python main.py eval-lora

# Custom config
python main.py pipeline --config my_config.yaml

# Specific stages
python main.py pipeline --stages eval
"""