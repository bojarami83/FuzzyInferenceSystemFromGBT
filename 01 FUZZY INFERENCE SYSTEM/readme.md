# Interpretable Credit Scoring via Fuzzy Inference Systems from Gradient Boosting

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This repository contains the complete implementation of a methodology that transforms CatBoost gradient boosting ensembles into fully interpretable fuzzy inference systems. The approach emphasizes linguistic term generation through adaptive cutpoint optimization strategies, achieving superior sensitivity while maintaining complete transparency in credit risk assessment.

## Overview

Financial institutions require credit scoring models balancing predictive accuracy with regulatory compliance for transparency. While gradient boosting machines achieve state-of-the-art performance, their black-box nature limits adoption in regulated environments requiring explainable decisions (GDPR Article 22, US Fair Credit Reporting Act). This work addresses this challenge through systematic transformation of high-performance ensemble models into human-interpretable fuzzy rule systems.

## Key Results

Experimental evaluation on the publicly available Lending Club dataset demonstrates:

- **Superior Sensitivity**: The fuzzy system exceeds the original CatBoost baseline on the business-critical sensitivity metric
- **Complete Interpretability**: Human-readable linguistic rules with explicit decision logic
- **Regulatory Compliance**: Full transparency satisfying GDPR and FCRA requirements
- **Business Alignment**: Optimizes detection of defaults where missed predictions incur substantially higher costs than rejected applications

The approach successfully demonstrates that transparent AI can optimize business-critical metrics while providing complete explainability through systematic pattern extraction and optimized linguistic representation.

## Methodology

Our five-stage approach comprises:

### Stage 1: Baseline Model Training and Selection

Benchmark multiple tree-based ensemble models (Random Forest, LightGBM, CatBoost) on historical credit data with interpretability-oriented configurations. Select the model achieving optimal balance between predictive performance and structural simplicity for rule extraction.

Key configuration principles:
- Moderate tree depth to limit rule complexity
- Controlled ensemble size balancing accuracy and interpretability
- Native categorical feature handling avoiding black-box transformations
- Independent threshold optimization maximizing F1-score

### Stage 2: Comprehensive Rule Extraction

Extract all decision paths from the trained gradient boosting ensemble, generating a complete set of interpretable rules with associated logit contributions preserving original model behavior. Each rule represents a complete path from root to leaf with explicit feature thresholds and class predictions, maintaining the quantitative impact structure of the original ensemble.

### Stage 3: Adaptive Cutpoint Optimization

Transform numerical variables into linguistic terms through three complementary strategies automatically selected based on distribution characteristics:

**Strategy 1 - Constrained Nonlinear Optimization**: For near-uniform distributions, formulate convex optimization minimizing deviation from uniform spacing while maintaining minimum coverage constraints. Solved efficiently via quadratic programming, generating cutpoints with smooth spacing.

**Strategy 2 - Adaptive Percentile Alignment**: For skewed distributions, align cutpoints with data percentiles ensuring equal sample representation across linguistic terms despite distributional asymmetry. Prevents clustering of terms in high-density regions.

**Strategy 3 - Density-Weighted K-Means Clustering**: For multimodal distributions, cluster observed values weighted by kernel density estimates. Generate cutpoints at cluster boundaries, capturing natural groupings in the data while respecting distributional complexity.

The framework generates interpretable linguistic terms per variable with trapezoidal membership functions exhibiting controlled overlap for smooth transitions between adjacent terms. Automatic strategy selection evaluates skewness, kurtosis, and modality to choose the most appropriate optimization approach for each variable.

### Stage 4: Quality-Based Tree and Rule Filtering

Apply precision-weighted coverage metrics to identify high-quality trees from the full ensemble, producing an optimized fuzzy rule set while balancing model simplicity with predictive power preservation. Filter criteria prioritize:

- **Rule Precision**: Favor rules with high classification accuracy
- **Sample Coverage**: Ensure adequate support for statistical reliability
- **Non-Redundancy**: Eliminate semantically duplicate decision patterns
- **Structural Quality**: Select trees contributing distinct decision logic

The filtering process systematically reduces ensemble complexity while maintaining the gradient boosting structure and predictive characteristics.

### Stage 5: Transparent Fuzzy Inference

Implement single-level normalization fuzzy aggregation preventing double-normalization artifacts:

1. **Fuzzification**: Convert crisp inputs to membership degrees across linguistic terms using trapezoidal functions
2. **Rule Activation**: Compute activation strength using minimum operator (Zadeh's AND) for conjunctive antecedents
3. **Per-Tree Aggregation**: Calculate weighted average logit within each tree, preserving original contribution structure
4. **Ensemble Sum**: Aggregate tree-level logits maintaining the additive property of gradient boosting
5. **Classification**: Apply sigmoid transformation and optimized threshold for final prediction

The inference process maintains complete transparency with every input value, rule activation, and contribution traceable through the decision pipeline.

## Repository Structure
```
FuzzyInferenceSystemFromGBT/
│
├── data/                          # Dataset handling
│   ├── download_instructions.md
│   └── preprocessing.py
│
├── models/                        # Baseline model training
│   ├── train_catboost.py
│   ├── train_lightgbm.py
│   └── train_random_forest.py
│
├── rule_extraction/               # Decision tree rule extraction
│   ├── extract_rules.py
│   └── rule_quality_metrics.py
│
├── cutpoint_optimization/         # Linguistic term generation
│   ├── constrained_optimization.py
│   ├── percentile_alignment.py
│   ├── density_kmeans.py
│   └── adaptive_strategy_selector.py
│
├── fuzzy_system/                  # Fuzzy inference implementation
│   ├── membership_functions.py
│   ├── rule_base.py
│   ├── inference_engine.py
│   └── threshold_optimization.py
│
├── evaluation/                    # Performance evaluation
│   ├── metrics.py
│   └── comparative_analysis.py
│
├── visualization/                 # Plotting and analysis
│   ├── plot_membership_functions.py
│   ├── plot_performance.py
│   └── export_rules_paper.py
│
├── notebooks/                     # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_training.ipynb
│   ├── 03_rule_extraction.ipynb
│   ├── 04_cutpoint_optimization.ipynb
│   └── 05_fuzzy_inference.ipynb
│
├── requirements.txt
├── setup.py
├── README.md
└── LICENSE
```

## Installation

### Requirements

- Python 3.8+
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- Scikit-learn >= 1.0.0
- CatBoost >= 1.0.0
- LightGBM >= 3.2.0
- Matplotlib >= 3.4.0
- SciPy >= 1.7.0

### Setup
```bash
# Clone repository
git clone https://github.com/bojarami83/FuzzyInferenceSystemFromGBT.git
cd FuzzyInferenceSystemFromGBT

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package and dependencies
pip install -e .
```

## Quick Start

### 1. Data Preparation

Download Lending Club dataset from [Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club) and place in `data/` directory.
```python
from data.preprocessing import prepare_lending_club_data

X_train, X_test, y_train, y_test = prepare_lending_club_data(
    filepath='data/lending_club.csv',
    test_size=0.30,
    random_state=42
)
```

### 2. Train Baseline Models
```python
from models.train_catboost import train_catboost_optimized

catboost_model, metrics, X_encoded, mappings = train_catboost_optimized(
    X_train, y_train, X_test, y_test,
    iterations=200,
    depth=6,
    learning_rate=0.1,
    verbose=True
)
```

### 3. Extract Rules
```python
from rule_extraction.extract_rules import extract_all_rules

rules = extract_all_rules(
    model=catboost_model,
    feature_names=X_train.columns.tolist(),
    categorical_mappings=mappings
)
```

### 4. Optimize Cutpoints and Generate Linguistic Terms
```python
from cutpoint_optimization.adaptive_strategy_selector import optimize_cutpoints

linguistic_terms = optimize_cutpoints(
    X_train=X_train,
    rules=rules,
    target_terms_per_variable={'dti': 4, 'loan_amnt': 7, 'fico_range_low': 6}
)
```

### 5. Build Fuzzy System
```python
from fuzzy_system.inference_engine import FuzzyInferenceSystem

fuzzy_system = FuzzyInferenceSystem(
    linguistic_terms=linguistic_terms,
    fuzzy_rules=filtered_rules,
    threshold=0.75
)

predictions = fuzzy_system.predict(X_test)
```

### 6. Evaluate Performance
```python
from evaluation.metrics import compute_classification_metrics

metrics = compute_classification_metrics(
    y_true=y_test,
    y_pred=predictions,
    y_prob=fuzzy_system.predict_proba(X_test)
)

print(f"Sensitivity: {metrics['sensitivity']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"F1-Score: {metrics['f1']:.4f}")
```

## Dataset

This implementation uses the **Lending Club Loan Data** dataset containing loan applications from 2007-2018. The dataset includes:

- Demographic information (employment length, home ownership, location)
- Financial indicators (debt-to-income ratio, annual income, credit utilization)
- Credit history (FICO scores, delinquencies, public records)
- Loan characteristics (amount, term, interest rate, purpose)

After preprocessing, the data is partitioned into training and independent test sets using stratified sampling to maintain class distribution. Selected features prioritize regulatory-compliant variables commonly used in credit risk assessment.

**Data Availability**: Public dataset available at [Kaggle Lending Club](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

## Key Features

**Precision-Recall Trade-off**: The fuzzy system prioritizes sensitivity (detecting actual defaults) over precision (minimizing false positives), aligned with credit risk business objectives where missed defaults incur substantially higher costs than rejected applications. This deliberate trade-off reflects real-world financial decision-making priorities.

**Interpretable Linguistic Terms**: Each numerical variable transforms into human-readable categories (e.g., "Low", "Medium", "High" for debt-to-income ratio) with explicit membership functions defining term boundaries. Domain experts can validate and adjust these definitions to align with institutional risk policies.

**Complete Decision Transparency**: Every prediction traces through activated fuzzy rules, showing which applicant characteristics contributed to the classification and by what magnitude. This enables explanation generation for regulatory compliance and customer communication.

**Threshold Flexibility**: Classification thresholds adjust based on institutional risk appetite, enabling fine-grained control over the sensitivity-precision trade-off without retraining the underlying model.

## Technical Features

- **Trapezoidal Membership Functions**: Smooth transitions with controlled overlap between adjacent linguistic terms preventing discontinuous membership boundaries
- **Automated Strategy Selection**: Distribution-aware cutpoint optimization selecting appropriate method based on skewness, kurtosis, and modality characteristics
- **Per-Tree Logit Aggregation**: Preserves gradient boosting's additive structure through weighted averaging within trees before ensemble summation
- **Single-Level Normalization**: Prevents double-normalization artifacts ensuring logit contributions maintain quantitative impact across the inference pipeline
- **Systematic Threshold Optimization**: Evaluation across multiple classification thresholds with explicit documentation of performance trade-offs

## Use Cases

This methodology applies to any domain requiring both high predictive accuracy and complete model interpretability:

- **Financial Services**: Credit risk assessment, loan approval, fraud detection
- **Healthcare**: Medical diagnosis support, treatment recommendation, patient risk stratification
- **Insurance**: Policy underwriting, claims assessment, risk pricing
- **Regulatory Compliance**: Any application subject to right-to-explanation requirements

The approach particularly benefits scenarios where:
- Regulatory frameworks mandate explainable decisions
- Domain experts must validate model logic
- End-users require justification for automated decisions
- Model auditing and governance processes require transparency

## Limitations and Future Work

**Precision-Recall Trade-off**: Low precision generates substantial false positive rates potentially increasing human review costs. Future work should explore hybrid ensemble approaches combining fuzzy interpretability for borderline cases with black-box precision for clear-cut decisions.

**Rule Base Complexity**: Large rule sets may reduce practical interpretability despite individual rule clarity. Future research should investigate rule clustering techniques grouping semantically similar rules to reduce cognitive load while preserving transparency.

**XAI Comparison and Expert Validation**: This work focuses on fuzzy inference systems without comparison against modern XAI alternatives (GAMs, xNAM, monotonic boosting) or validation with credit risk domain experts. Systematic XAI benchmarking and expert evaluation would strengthen interpretability claims and assess practical utility in real-world financial contexts.

## Citation

If you use this code in your research, please cite:
```bibtex
@inproceedings{loaiza2026interpretable,
  title={Interpretable Credit Scoring through Optimized Fuzzy Inference Systems from Gradient Boosting Trees},
  author={Loaiza, Byron and Ter{\'a}n, Luis and Loza-Aguirre, Edison},
  booktitle={4th World Conference on Explainable Artificial Intelligence (XAI 2026)},
  year={2026},
  address={Fortaleza, Brazil}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

**Byron Loaiza**  
PhD Candidate, Human-IST Institute, University of Fribourg, Switzerland  
Email: byron.loaiza@unifr.ch

**Luis Terán**  
Human-IST Institute, University of Fribourg, Switzerland  
Email: luis.teran@unifr.ch

**Edison Loza-Aguirre**  
Departamento de Informática y Ciencias de la Computación  
Escuela Politécnica Nacional, Quito, Ecuador  
Email: edison.loza@epn.edu.ec

## Acknowledgments

This research was conducted as part of Byron Loaiza's doctoral thesis at the University of Fribourg under the supervision of PD Dr. Luis Terán and co-supervision of Dr. Edison Loza-Aguirre.

## Contact

For questions, issues, or collaboration opportunities, please open an issue on GitHub or contact the authors directly.

---

**Paper Submitted to**: 4th World Conference on eXplainable Artificial Intelligence (XAI 2026)  
**Conference Date**: July 1-3, 2026 | Fortaleza, Brazil  
**Code and Data Released**: Upon publication