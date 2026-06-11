# Brain-WM: Brain Glioblastoma World Model

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2603.07562)

This is the official repository for "**Brain-WM: Brain Glioblastoma World Model**".

## 📖 Overview

To bridge the gap between prognostic simulation and active clinical planning in glioblastoma (GBM) management, we present **Brain-WM**, a brain GBM world model that jointly enables next-step treatment planning and future MRI generation through three core innovations:

1. **Synergistic Feedback Loop**: Establishes a dynamic interplay where simulated tumor evolution informs treatment formulation, while treatment intent constrains biologically plausible progression.
2. **Y-shaped MoT Architecture**: Introduces a novel Y-shaped Mixture-of-Transformers (MoT) that structurally disentangles heterogeneous objectives, leveraging cross-task synergies while preventing feature collapse.
3. **Multi-timepoint Mask Alignment**: Anchors latent representations to anatomically grounded tumor structures to ensure progression-aware semantics.


<div align="center">
  <img src="meta_data/overview.png" alt="overview" width="1000">
</div>

Validated across multi-centric cohorts, Brain-WM provides a robust clinical sandbox for optimizing decision-making and patient healthcare.


## 📊 Datasets

Brain-WM was systematically validated across multi-centric cohorts:
* **Internal Cohort**: Aggregated from three public datasets: [LUMIERE](https://github.com/ysuter/gbm-data-longitudinal), [MU-Glioma Post](https://www.cancerimagingarchive.net/collection/mu-glioma-post/), and [UCSF-ALPTDG](https://imagingdatasets.ucsf.edu/dataset/2).
* **External Validation Cohort**: Evaluated on an independent validation cohort from the [RHUH-GBM](https://www.cancerimagingarchive.net/collection/rhuh-gbm/) and [UCSD-PTGBM](https://www.cancerimagingarchive.net/collection/ucsd-ptgbm/) datasets.

> **📌 Data Availability Note:**
> 
> 1. Due to data privacy regulations, users must independently request access to these public raw MRI datasets via their respective institutional portals.
> 2. The curated **treatment labels** organized for our experiments will be <span style="color:#0366d6;">**fully open-sourced upon the acceptance of this manuscript**</span>.

## 🚀 Getting Started

### Prerequisites

The codebase is built using Python, relying heavily on `torch` for deep learning modeling and `monai` for volumetric medical image array manipulations.

Clone the repository and set up the required environment:

```bash
    git clone https://github.com/thibault-wch/Brain-GBM-world-model.git
    cd Brain-GBM-world-model

    # Create and activate a conda environment
    conda create -n brainwm python=3.10.18
    conda activate brainwm

    # Install requirements
    pip install -r requirements.txt

```

### ⚙️ Training Pipeline

**Step 1: Configure Pre-trained Weights**
1. *Download Weights*: Obtain the pre-trained models for [Show-o2](https://huggingface.co/showlab/show-o2-1.5B/tree/main) and [Wan-VAE 2.1](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B/blob/main/Wan2.1_VAE.pth).
2. *Update Paths*: Manually point to your local downloaded weights in the following files:
   - `configs/showo2-1.5b_stage_2_a.yaml`
   - `train_stage_two.py` line 172.
3. *Adjust LoRA Parameters*: Tune the LoRA settings (`rank` and `alpha`) within the `.yaml` configuration file to match your specific computational resources and training requirements.

**Step 2: Train the Model**

Start the Brain-WM training process by executing the shell script below.

```bash
    bash train_GBM.sh

```

### 💡 Inference

Choose the appropriate script based on your target task.

* **Treatment Plan Recommendation**

```bash
    python inference_mmu.py

```

* **Future MRI Generation**

```bash
    python inference_t2i.py

```

> **📌 Important Architectural Notes:**
> * The core architectural modifications for Brain-WM are implemented within `models/modeling_showo2_qwen2_5.py`.
> * The corresponding spatial segmentor code can be found in `models/segmentor.py`.
> 
> 

## 🤝 Acknowledgments

This work is heavily based on [taming-transformers](https://github.com/CompVis/taming-transformers), [transformers](https://github.com/huggingface/transformers), [accelerate](https://github.com/huggingface/accelerate), [diffusers](https://github.com/huggingface/diffusers), [Show-o](https://github.com/showlab/Show-o/), and [Show-o2](https://github.com/showlab/Show-o/tree/main/show-o2). We extend our sincere thanks to all the authors for their outstanding open-source contributions.


If you find this code or our paper useful for your research, please star 🌟 this repository and cite our work:

```bibtex
@article{wang2026brain,
  title={{Brain-WM}: Brain Glioblastoma World Model},
  author={Wang, Chenhui and Zheng, Boyun and Bao, Liuxin and Peng, Zhihao and Woo, Peter YM and Shan, Hongming and Yuan, Yixuan},
  journal={arXiv:2603.07562},
  year={2026}
}

```