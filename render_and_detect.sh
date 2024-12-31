#!/usr/bin/bash

meshes=../out/postprocessing/2024-12-28_slice-0-3420/meshes/fitted

for f in $meshes/c* ; do
    echo $f
    PYTHONPATH=../ThaumatoAnakalyptor \
        python \
        -m ThaumatoAnakalyptor.mesh_to_surface \
        --format jpg \
        --r 15 \
        --nr_workers 4 \
        $f/mesh.obj \
        ../data/zarr/Scroll1_masked.zarr
    python ../Vesuvius-Grandprize-Winner/fast_inference_timesformer.py \
        --compile \
        --quality 1 \
        --src_sd 1 \
        --reverse \
        --start_idx 3 \
        --stop_idx 29 \
        --model_path ../Vesuvius-Grandprize-Winner/timesformer_weights.ckpt \
        --layer_path $f/layers \
        --out_path $f/predictions.jpg
done

