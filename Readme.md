# varify_sampling

A lightweight experimental template repository for point cloud dataset distillation and response-matching based sampling methods.

This repository is mainly used for:

* maintaining reusable experiment templates;
* synchronizing implementation updates;
* recording recent experimental variants;
* testing new response-matching strategies on low-PPC point cloud distillation settings.

## Recent Results

### AdaSADM

AdaSADM is an extension of SADM that introduces multi-layer point-wise response matching for point cloud dataset distillation.

Compared with the original SADM setting, AdaSADM incorporates adaptive matching across multiple point-wise feature layers, which provides more stable improvements under low-PPC settings.

Repository:

https://github.com/ViceMusic/MutilLayerSADM

## Repository Usage

This repository serves as a working template for ongoing experiments. New methods, ablation variants, and backbone extensions will be added and tested here before being organized into the main method repository.

## Notes

The current focus is on low-budget point cloud dataset distillation, especially under different PPC settings and backbone architectures.
