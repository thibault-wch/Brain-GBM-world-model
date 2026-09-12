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

September 12, 2026: The codebase has been reorganized and refined to improve its completeness.

## 📊 Datasets

During training, we construct dense longitudinal pairs across all available time points to provide richer supervision. Brain-WM was systematically evaluated across multicenter cohorts:

- **Internal Cohort**: Combines three public datasets: [LUMIERE](https://github.com/ysuter/gbm-data-longitudinal), [MU-Glioma Post](https://www.cancerimagingarchive.net/collection/mu-glioma-post/), and [UCSF-ALPTDG](https://imagingdatasets.ucsf.edu/dataset/2).
- **External Validation Cohort**: Uses the independent [UCSD-PTGBM](https://www.cancerimagingarchive.net/collection/ucsd-ptgbm/) dataset.



All MRI volumes are aligned to a common anatomical space before model input. We register each T1-weighted scan to the [SRI24 template](https://www.nitrc.org/projects/sri24/) with [ANTs](https://github.com/antsx/ants/) (`antsRegistrationSyNQuick` is sufficient), then apply the resulting affine transform to all other imaging modalities and masks.

## 🚀 Getting Started

### Prerequisites

The codebase is built using Python, relying heavily on `torch` for deep learning modeling and `monai` for volumetric medical image array manipulations.

Clone the repository and set up the required environment:

```bash
git clone https://github.com/thibault-wch/Brain-GBM-world-model.git
cd Brain-GBM-world-model

# Create the conda environment and install the CUDA dependencies.
bash build_env.sh
conda activate brainwm

# Cache the Qwen2.5 config and tokenizer files.
# The language-model weights are loaded from the Show-o2 base checkpoint.
hf download Qwen/Qwen2.5-1.5B-Instruct \
    config.json tokenizer.json tokenizer_config.json vocab.json merges.txt

# Cache the SigLIP config and weights used to construct the semantic
# vision encoder and its positional embedding.
hf download google/siglip-so400m-patch14-384 \
    --include "config.json" "*.safetensors" "*.safetensors.index.json" "pytorch_model*.bin"
 ```

### ⚙️ Training Pipeline

**Step 1: Configure Pre-trained Weights**
1. *Download Weights*: Obtain the pre-trained models for [Show-o2](https://huggingface.co/showlab/show-o2-1.5B/tree/main) and [Wan-VAE 2.1](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B/blob/main/Wan2.1_VAE.pth).
2. *Update Paths*: Manually point to your local downloaded weights in  `configs/showo2_1.5b_stage_2_a.yaml`
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
*Evaluation metrics are computed after full-volume padding restores both predictions and references to the original spatial extent.*
> **📌 Important Architectural Notes:**
> * The core architectural modifications for Brain-WM are implemented within `models/modeling_showo2_qwen2_5.py`.
> * The corresponding mask aligner code can be found in `models/segmentor.py`.
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