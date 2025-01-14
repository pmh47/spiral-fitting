
# Autosegmentation by Spiral Fitting

This is an attempt at autosegmentation (i.e. automated digital unrolling) of papyrus scrolls as part of the  [Vesuvius Challenge](https://scrollprize.org).

See this [report](https://docs.google.com/document/d/1ZOIqtG7IbgaW4moWmjYCIeBsyQRNoonQe_KtAnQp8gw/view) for more details on the method.


## Instructions

### Required data
- Scroll 1 in OME-Zarr format
  - this is only used for visualisation, hence exact normalisation etc. is unimportant
  - only the /2 scale is used
  - path at L22 of fit_spiral.py
- horizontal fiber predictions from Sean, [here](https://dl.ash2txt.org/community-uploads/bruniss/Fiber-and-Surface-Models/GP-Predictions/3d-zarr/mask-hz-only_rescaled.zarr/)
  - path at L10 of extract_fibre_ccs_hz.py
- vertical fiber predictions from Sean, [here](https://dl.ash2txt.org/community-uploads/bruniss/scrolls/s1/fibers/new-fiber-preds/hz_regular/)
  - path at L10 of extract_fibre_ccs_vt.py
  - these first need converting to zarr with [grids_to_zarr.py](https://dl.ash2txt.org/community-uploads/bruniss/scrolls/helper-scripts/grids_to_zarr.py) (change paths at L189-190)
- surface predictions from Sean, [here](https://dl.ash2txt.org/community-uploads/bruniss/Fiber-and-Surface-Models/GP-Predictions/updated_zarrs/surface2xt-updated_ome.zarr/)
  - path at L17 of extract_surface_tracks.py
  - also at L23 of fit_spiral.py
- the ground-truth mesh for the GP banner, [here](https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/paths/20231231235900_GP/)
  - this is only used for visualisation and evaluation
  - path at L32 of fit_spiral.py

### Hardware & software environment
- any modern CPU; 96GB RAM; 24GB VRAM (e.g. RTX 3090 / A5000)
- only tested under recent versions of Ubuntu (but no exotic dependencies)
- see environment.yaml for a suitable conda environment (for pytorch 2.3 and cuda 12.1 on Linux)

### Data preprocessing
- extract skeletonised 2D connected components for horizontal and vertical fibers:
  - `python extract_fibre_ccs_hz.py`
  - `python extract_fibre_ccs_vt.py`
- extract various representations from surface predictions (normals; skeletonised 2D connected components; relative winding numbers):
  - `python extract_surface_tracks.py`

### Spiral fitting
- `WANDB_MODE=disabled python fit_spiral.py`
- results will be written to a subfolder of ../output/postprocessing

### Outputs
Various outputs are created periodically (every 1000 optimisation steps) during fitting:

- mesh OBJs including UV coordinates (meshes/)
  - we output per-winding, per-chunk (fixed lengths of scroll, each roughly 10% of the GP region), and complete meshes; the per-chunk are a good balance between file size and coverage
  - these can be directly used in ThaumatoAnakalyptor's mesh_to_surface to create a rendering
  - if used for ink-detection, note the layer order is reversed, since normals point outwards
- rough per-winding renderings (rendered_*.jpg)
  - these are generated on-the-fly and very efficiently, but are coarser than renders from mesh_to_surface
  - there is a color-bar along the bottom indicating the local winding angle, and red lines indicate lines of constant scan-space z
- cross-section images showing…
  - the predicted segmentation overlaid on the scroll (spiral_on_scroll_*); the colors indicate the canonical angle in each winding, and match the colorbar on rendered_* images
  - the ground-truth GP mesh (cyan) overlaid on predicted segmentation (red) (spiral_on_gp_*)
  - the raw surface predictions used in the losses (green), overlaid on the predicted segmentation (red) (spiral_on_pred_*)
  - *for all these section images, the current segmentation is shown by a narrow gaussian density centered on the segment, rather than a single line; the local maximum isoline of the density is the actual path of the mesh*
  - *the section images are output for a regularly-spaced subset of slices, denoted e.g. s1814 in the filenames; these slice-numbers are at 4x downsampled resolution*
- longwise sectional images tracking the umbilicus, showing the predicted segmentation (red) and the raw surface predictions (green) (spiral_zx_* & spiral_zy_*)


