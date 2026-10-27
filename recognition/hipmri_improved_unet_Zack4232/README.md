# HipMRI Segmentation with 2D Improved U-Net

## Project Scope

This project studies multi-class segmentation of HipMRI 2D images.
The planned baseline is standard 2D U-Net, and the main model is
2D Improved U-Net at COMP3710 Normal difficulty.

The study will examine whether the improved model produces better
boundaries in difficult regions and whether the gains justify its
computing cost. The target prostate test Dice is at least 0.75.

No model has been implemented or trained yet.

## Verified Environment

- Platform: Rangpur.
- GPU: NVIDIA A100-PCIE-40GB.
- PyTorch: 2.14.1+cu130.
- PyTorch CUDA build: 13.0.
- CUDA availability and a small GPU matrix multiplication test passed.
- The Slurm test job completed with exit code 0.
- The successful test used partition `comp3710`, account `comp3710`,
  one GPU, and `--mem=0`.

Basic GPU computation has been verified. Full training speed and peak
memory use have not been measured. Memory settings need review before
full training.

## Data Audit

Data location on Rangpur:

`/home/groups/comp3710/HipMRI_Study_open/keras_slices_data`

| Split | Number of Cases | Image-Mask Pairs |
|---|---:|---:|
| Training | 31 | 11,460 |
| Validation | 4 | 660 |
| Test | 3 | 540 |
| Total | 38 | 12,660 |

There is no overlap between splits based on case IDs in filenames.
Images and masks match by case, week, and slice number. No missing
pairs, duplicate keys, or unexpected filenames were found.

A sample of 93 training pairs had matching shapes of 256 x 128.
No shape or affine mismatch, non-finite values, or non-integer mask
values were found in this sample. Full content validation is pending.

The test set contains only three cases. Final evaluation should
include per-case results rather than treating slices as independent
patients.

## Label Definitions

The original dataset's `BriefDataDescription.txt` defines:

| Value | Class |
|---|---|
| 0 | Background |
| 1 | Body |
| 2 | Bones |
| 3 | Bladder |
| 4 | Rectum |
| 5 | Prostate |

Week 0 was acquired before treatment. Weeks 1-7 were acquired weekly
during treatment.

Labels 0-5 were observed in the sampled course 2D data. Visual checks
of alignment and consistency with the original label definitions
are pending. The physical meaning of the stored pixel spacing has
not yet been verified.

## Planned Evaluation

- Per-class Dice and IoU.
- Baseline comparison using the same patient splits.
- Parameter count, peak GPU memory, and per-case inference time.
- Visual analysis of 3-5 representative failure cases.
- An engineering recommendation based on measured results.

## Feasibility Review Status

The one-page feasibility review and tutor check-off are pending.
These preliminary checks do not replace that requirement.

## Data Acknowledgement and References

Data were provided by HipMRI Study.

- Dataset: Labelled weekly MR images of the male pelvis.
  https://doi.org/10.25919/45t8-p065
- Dowling et al. (2015), Automatic Substitute Computed Tomography
  Generation and Contouring for Magnetic Resonance Imaging (MRI)-Alone
  External Beam Radiation Therapy From Standard MRI Sequences.
  https://doi.org/10.1016/j.ijrobp.2015.08.045

Data use must follow the supplied terms, including non-commercial use,
no attempts to identify or contact participants, and proper
acknowledgement and citation.

Datasets and model weights will not be committed to this repository.

## Artificial Intelligence Usage Disclosure

OpenAI ChatGPT/Codex assisted with understanding the assignment,
setting up Rangpur, diagnosing Slurm settings, drafting preliminary
data-check scripts, and preparing documentation.

Environment and data-check scripts were run on Rangpur and their
outputs were inspected. Model implementation and training have not
started. This disclosure will be updated as the project develops.

## Initial Visual Inspection

The figure below shows training case 004 at week 0. Of its 60 slices,
31 contain prostate label 5. Slices 22, 30, 37, 44, and 52 were selected
across the prostate-positive slice range.

![Training MRI, ground-truth masks, and organ overlays](figures/training_data_preview.png)

The five displayed pairs show no obvious global displacement or flip.
This is a limited visual check, not confirmation of all cases or
fine boundary accuracy.

The prostate label occupies a small region in slices 22 and 52 and
larger regions in the selected intermediate slices. This supports
including small-target slices in later performance analysis.

Images are shown in stored array orientation. Anatomical directions
have not been assigned. These are ground-truth previews, not model
predictions or model failure cases.

To reproduce the figure in an environment with NumPy, NiBabel, and
Matplotlib installed, run from this project directory:

```bash
python visualize_data.py
```
