# Final Formal Segmentation Results

## Evaluation protocol

This report summarizes the six completed 50-epoch formal experiments. The fixed train/validation/test splits were preserved within each dataset for a fair U-Net versus My Model comparison. In every experiment, the best checkpoint was selected solely by validation Dice. The held-out test set was evaluated only after that checkpoint had been reloaded; test results did not influence checkpoint selection.

Images and masks were evaluated at 224×224 resolution with a binary threshold of 0.5. HD95 and ASSD are pixel-space surface distances at the resized 224×224 resolution, calculated using the same convention across all experiments; lower values are better. Dice and mIoU are overlap metrics; higher values are better.

FLOPs are reported per image for U-Net and per image–text pair for My Model, in decimal GFLOPs (10⁹ FLOPs). A multiply-add counts as 2 FLOPs. U-Net counts convolutional operations; My Model counts convolution, transposed-convolution, and linear operations. The frozen text encoder used by My Model is an offline compatibility embedding pipeline and is not included in the model-forward FLOPs: `google-bert/bert-base-uncased` produces fixed 10×768 text embeddings, which are then consumed by My Model.

## QaTa-COV19

Best epochs: U-Net **33**; My Model **26**.

### Validation

| Model | Dice | mIoU | HD95 (pixels) | ASSD (pixels) | Params | FLOPs (GFLOPs) |
|---|---:|---:|---:|---:|---:|---:|
| U-Net | 0.7534 | 0.6443 | 35.8104 | 11.6468 | 14,751,873 | 45.3403 |
| My Model | 0.7693 | 0.6667 | 29.1823 | 11.3475 | 39,933,345 | 53.5596 |

### Test

| Model | Dice | mIoU | HD95 (pixels) | ASSD (pixels) | Params | FLOPs (GFLOPs) |
|---|---:|---:|---:|---:|---:|---:|
| U-Net | 0.7712 | 0.6726 | 32.3874 | 9.9128 | 14,751,873 | 45.3403 |
| My Model | 0.7833 | 0.6915 | 30.1917 | 13.4908 | 39,933,345 | 53.5596 |

## MosMedData+

Best epochs: U-Net **42**; My Model **50**.

### Validation

| Model | Dice | mIoU | HD95 (pixels) | ASSD (pixels) | Params | FLOPs (GFLOPs) |
|---|---:|---:|---:|---:|---:|---:|
| U-Net | 0.7442 | 0.6235 | 20.5379 | 9.2034 | 14,751,873 | 45.3403 |
| My Model | 0.7100 | 0.5867 | 24.6745 | 16.1590 | 39,933,345 | 53.5596 |

### Test

| Model | Dice | mIoU | HD95 (pixels) | ASSD (pixels) | Params | FLOPs (GFLOPs) |
|---|---:|---:|---:|---:|---:|---:|
| U-Net | 0.7627 | 0.6378 | 15.5413 | 4.5886 | 14,751,873 | 45.3403 |
| My Model | 0.7322 | 0.6067 | 19.0764 | 10.9518 | 39,933,345 | 53.5596 |

## BUSI

Best epochs: U-Net **42**; My Model **40**.

### Validation

| Model | Dice | mIoU | HD95 (pixels) | ASSD (pixels) | Params | FLOPs (GFLOPs) |
|---|---:|---:|---:|---:|---:|---:|
| U-Net | 0.7157 | 0.6094 | 42.2711 | 19.6245 | 14,751,873 | 45.3403 |
| My Model | 0.7473 | 0.6534 | 31.9240 | 17.7226 | 39,933,345 | 53.5596 |

### Test

| Model | Dice | mIoU | HD95 (pixels) | ASSD (pixels) | Params | FLOPs (GFLOPs) |
|---|---:|---:|---:|---:|---:|---:|
| U-Net | 0.7438 | 0.6558 | 33.8636 | 17.9123 | 14,751,873 | 45.3403 |
| My Model | 0.7786 | 0.6916 | 29.4132 | 17.6729 | 39,933,345 | 53.5596 |

### BUSI text provenance

The BUSI text annotations used by My Model came from the official MTGT release, not from the original LViT release. According to MTGT provenance, these annotations were simulated and physician-validated. The official MTGT train and test workbooks were combined globally and mapped by exact image filename to the unchanged local fixed train/validation/test splits; the workbooks' original partitioning was not used. Records for normal images absent from the cleaned 647-image dataset were ignored.

## Verified formal artifacts and caveats

The six referenced checkpoints were present and their PyTorch ZIP archives passed integrity checks. The metric sources and matching checkpoints were:

- QaTa-COV19 × U-Net: `results/unet_qata/final_metrics.json`; `results/unet_qata/best_unet_qata.pth`.
- QaTa-COV19 × My Model: `results/lvit_qata/final_metrics_full50.json`; `results/lvit_qata/best_lvit_qata_full50.pth`.
- MosMedData+ × U-Net: `results/unet_mosmed/final_metrics.json`; `results/unet_mosmed/best_unet_mosmed.pth`.
- MosMedData+ × My Model: `results/lvit_mosmed/final_metrics.json`; `results/lvit_mosmed/best_lvit_mosmed.pth`.
- BUSI × U-Net: `results/unet_busi/final_metrics.json`; `results/unet_busi/best_unet_busi.pth`.
- BUSI × My Model: `results/lvit_busi/final_metrics.json`; `results/lvit_busi/best_lvit_busi.pth`.

No smoke-test metric or checkpoint was used. QaTa-COV19 × My Model is the sole formal run with nonstandard `full50` artifact names; its formal JSON explicitly identifies 50 requested and completed epochs and points to the `full50` checkpoint. The older `best_lvit_qata.pth` was therefore excluded. QaTa-COV19 × U-Net stores validation surface-distance fields under `val_hd95` and `val_assd`, whereas the other formal JSON files use explicit `_pixels` field names; the report treats them as pixels consistently with that run's 224×224 evaluation implementation. MosMedData+ × U-Net's formal JSON omits an `epochs_completed` field, although it contains the final validation/test metrics and matching best-checkpoint path. FLOPs use the same 2-FLOP multiply-add convention, but the artifact descriptions differ in counted operator scope as documented above, and My Model's frozen offline BERT embedding generation is outside its per-forward FLOPs.
