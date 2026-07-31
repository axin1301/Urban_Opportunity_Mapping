# Generalizable Urban Opportunity Prediction from Satellite Imagery with Region–Task Mixture-of-Experts

This is the folder for the manuscript **"Generalizable Urban Opportunity Prediction from Satellite Imagery with Region–Task Mixture-of-Experts"**.

This project introduces **UrbanOpp**, a benchmark for urban opportunity prediction from satellite imagery, together with **OppMoE**, a region–task mixture-of-experts framework designed for cross-city and cross-indicator generalization.

![UrbanOpp Benchmark](./figs/fig1-1.png)

The repository is organized into three main directories:

```
.
├── generated_labels (UrbanOpp Benchmark)/
├── label_construction_50cities (UrbanOpp dataset construction pipeline)/
└── train_and_eval_codes (OppMoE)/
```

## Repository Structure

### `generated_labels/`

Contains the generated **UrbanOpp** benchmark labels, including urban opportunity indicators and the associated metadata used for training and evaluation.

### `label_construction_50cities/`

Contains the complete pipeline for constructing the **UrbanOpp** benchmark from multi-source urban data across the selected cities.

### `train_and_eval_codes/`

Contains the implementation of **OppMoE**, including model training, evaluation, and experimental scripts.

## Usage

Each directory contains its own **README.md** with detailed instructions on data preparation, execution commands, and experiment configuration. Please refer to the corresponding directory for step-by-step guidance.
