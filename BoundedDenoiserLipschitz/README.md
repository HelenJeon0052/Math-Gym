# Bounded Denoiser with Lipschitz Convergence and Jacobian Regularization



**Bounded Denoiser with Lipschitz Convergence and Jacobian Regularization**
**Period:** March 2026
**Tech:** Python, jax, Unet, Diffision, ViT

## Overview

The methodology of bounded denoiser will be experimented with Lipschitz Convergence and Jacobian Regularization.
(Jacobian Regularization → Lipschitz Continuity → Convergence)
This experiment proves that maintaining a Lipschitz constant L≤1 prevents the catastrophic error propagation typically observed in multi-iteration PnP frameworks.
the major goal of this experiment is to stabilize inverse problem pipeline in Vision models as effectively suppressing accumulation of artifacts when iterations increase at small -$\sigma$

**Core idea:**

> Use Lipschitz Convergence and Jacobian Regularization to ensure stabilization of diminishing-$\sigma$ and provide fixed-point convergence guarantees even in nonconvex and ill-posed inverse problems


## Key Contributions

1. **Investigation of bounded denoiser methodologies**

   * Work research and analysis of the specific methodologies regarding bounded denoiser's role and its ongoing experiments in Plug-and-Play optimization and in iterative inverse problems

2. **UNet, Diffusion, Vit bridging solutions**

   * Comparison among different models for examining the optimal x* and its applications with emphasis on stability and reconstruction quality.

3. **Continuation schedule integration**

   * A continuation-based update schedule into the experimental pipelines, enabling systematic control of the optimization trajectory and stability under diminishing-noise or adaptive-penalty settings.

4. **Unifying experimental framework**

   * A consistent experimental framework for evaluating denoiser regularization, model architecture, and continuation schedules.

---

## Repository Structure

```text
.
├── README.md
├── thorem.tex
├── pyproject.toml
├── src/
│   ├── models/
│   │   ├── UNet.py                
│   │   ├── diffusion.py           
│   │   └── vit.py         
│   ├── denoiser/
│   │   ├── denoiser.py
│   │   └── SN_wrappers.py
│   ├── jacobianRegLoss/
│   │   ├── jacobian_reg_loss.py        
│   │   └── 
│   ├── operators/
│   ├── pnp/
│   │   ├── ADMM_loop.py         
│   │   └── logging.py
│   ├── data/
│   │   ├── .py            
│   │   └── split_utils.py
│   └── utils/
├── notebooks/
│   ├── 00_env_check.ipynb
│   ├── 01_train_denoiser_realsn_jacreg.ipynb
│   ├── 02_pnp_admm_find_xstar.ipynb
│   └── 03_ablation_grid.ipynb
└── outputs/
    ├── checkpoints/
    └── results/
```

---


## Metrics & Evaluation

### Metrics

* PSNR / SSIM
* final primal residual $r^k$,

---

## Results (To be filled)

* Compression: **[N]%** parameter reduction
* Accuracy degradation: **$\leq$ [N]\%**

---

## Safety Notes (Medical Use)

This repository is **research-only**. Outputs from generative models can be misleading even when visually plausible.
Always evaluate with diagnosis-aware metrics and boundedness checks before any clinical interpretation.

---

## Citation

If you build on this work, cite:

* **[Author]**, jeon.isavelle@gmail.com

