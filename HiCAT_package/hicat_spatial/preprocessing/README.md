# Image feature extraction

Call `hicat_spatial.preprocessing.extract_image_features` with a raw H&E image and a
spot-coordinate table. Coordinates are raw-image pixel centers: `pixel_x`
indexes image columns and `pixel_y` indexes image rows. Centers outside the raw
image are rejected because they usually indicate swapped axes or coordinates
from a different resolution.

Spot windows may cross an image edge. The shared HiCAT aggregation layer clips
such windows to the available grid. With the default `normalize_by="overlap"`,
it normalizes by observed overlap rather than the full square, avoiding simple
edge attenuation. Set `ignore_zero_features=True` when zero grid vectors denote
masked background.

## UNI

UNI preprocessing preserves the top-left origin and pads the bottom/right edge.
When `scale_value` is below 1, HiCAT scales both spot centers and spot-window
size for backend aggregation, then restores raw-image coordinates in the final
AnnData object.

Default checkpoint location:

```text
hicat_spatial/preprocessing/uni/checkpoints/
  vit_large_patch16_224.dinov2.uni_mass100k/
    pytorch_model.bin
```

## HIPT

HIPT preprocessing also preserves the top-left origin and pads bottom/right.
Provide a checkpoint directory containing both files:

```text
checkpoints/
  vit256_small_dino.pth
  vit4k_xs_dino.pth
```

The upstream shifted-embedding path is used when `no_shift=False`; set
`no_shift=True` to request the upstream unshifted path. HiCAT does not modify
the imported HIPT model implementation.

CUDA requests automatically fall back to CPU when CUDA is unavailable. CPU
execution can be substantially slower for both image models.
