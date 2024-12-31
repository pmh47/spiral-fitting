
import zarr
import cc3d
import pickle
import kimimaro
import numpy as np
from tqdm import tqdm


predictions_zarr_path = '../data/original/bruniss/Fiber-and-Surface-Models/GP-Predictions/3d-zarr/mask-hz-only_rescaled.zarr'
cc_zarr_path = '../out/fibres/2024-10-31_bruniss_hz-only/cc.zarr/2'
skeleton_zarr_path = '../out/fibres/2024-10-31_bruniss_hz-only/skeleton.zarr/2'
skeleton_pkl_path = '../out/fibres/2024-10-31_bruniss_hz-only/skeleton.pkl'
shape_yx_transposed = True  # use for 2024-10-31 predictions at least
early_downsample_factor = 2  # applied immediately after loading, using striding
late_downsample_factor = 2  # applied after cc'ing, using max-pooling
predictions_threshold = 200  # 200 for surfaces & older hz-fibres; 0.5 for new vt-fibres
z_min, z_max = 0, 3594  # wrt target (maybe-downsampled) volume
z_chunk_depth = 80  # should be as large as fits in memory (to minimise number of cc 'boundaries' requiring expensive merging)
skeletonisation_chunk_depth = 300  # should be large enough to track long vertical fibres
dust_threshold = 150  # wrt target (downsampled) volume



def main():

    print('loading zarr')
    predictions_zarr_array = zarr.open(predictions_zarr_path, mode='r')

    overall_downsample_factor = early_downsample_factor * late_downsample_factor
    downsampled_shape = np.asarray(predictions_zarr_array.shape) // overall_downsample_factor
    if shape_yx_transposed:  # bruniss predictions from 2024-10-31 have y/x *shape* transposed but not pixels!
        downsampled_shape = downsampled_shape[[0, 2, 1]]
    assert downsampled_shape[1] < downsampled_shape[2]  # true for Scroll 1 at least
    def downsample(x, factor, min_or_max=np.max):
        assert x.ndim == 3
        x = x.reshape(x.shape[0] // factor, factor, x.shape[1] // factor, factor, x.shape[2] // factor, factor)
        x = min_or_max(x, axis=5)
        x = min_or_max(x, axis=3)
        x = min_or_max(x, axis=1)
        return x

    def get_chunks():
        for z_chunk_min in range(z_min, z_max, z_chunk_depth):

            z_chunk_max = min(z_chunk_min + z_chunk_depth, z_max, downsampled_shape[0] - 2)  # -2 is to work around a crackle bug
            print(f'preparing slices {z_chunk_min}-{z_chunk_max}')
            predictions = predictions_zarr_array[z_chunk_min * overall_downsample_factor : z_chunk_max * overall_downsample_factor]

            print(f'  downsampling and thresholding')
            predictions = predictions[::early_downsample_factor, ::early_downsample_factor, ::early_downsample_factor]
            predictions = (predictions > predictions_threshold).astype(np.uint8)

            print('  dusting')
            cc3d.dust(predictions, threshold=dust_threshold * late_downsample_factor**3, connectivity=6, in_place=True)

            print('  yielding to connected_components_stack')
            yield predictions.transpose(1, 2, 0)  # zyx --> yxz

    cc_labels_crackle, num_ccs = cc3d.connected_components_stack(get_chunks(), connectivity=6, return_N=True)  # note this is indexed in yxz order!
    print(f'found {num_ccs} connected components in total')

    print('initialising cc zarr')
    cc_zarr_array = zarr.open(
        cc_zarr_path,
        mode='w',
        shape=downsampled_shape,
        chunks=(64, 64, 64),
        dtype=cc_labels_crackle.dtype,
        fill_value=0,
        write_empty_chunks=False,
        compressor=zarr.Blosc(cname='zstd', clevel=3)
    )

    for z_chunk_min in range(z_min, z_max, z_chunk_depth):

        z_chunk_max = min(z_chunk_min + z_chunk_depth, z_max, downsampled_shape[0] - 2)
        print(f'postprocessing slices {z_chunk_min}-{z_chunk_max}')

        cc_labels = cc_labels_crackle[:, :, (z_chunk_min - z_min) * late_downsample_factor : (z_chunk_max - z_min) * late_downsample_factor].transpose(2, 0, 1)

        if late_downsample_factor > 1:
            print('  downsampling')
            # FIXME: where both fg, should dither selection of label instead of taking max, to preserve connectivity of both
            cc_labels = downsample(cc_labels, factor=late_downsample_factor)

        print('  writing to zarr')
        if shape_yx_transposed:
            cc_zarr_array[z_chunk_min : z_chunk_max, :cc_labels.shape[1], :cc_labels.shape[2]] = cc_labels[:, :cc_zarr_array.shape[1], :cc_zarr_array.shape[2]]
        else:
            cc_zarr_array[z_chunk_min : z_chunk_max] = cc_labels
        del cc_labels

    print('initialising skeleton zarr')
    skeleton_zarr_array = zarr.open(
        skeleton_zarr_path,
        mode='w',
        shape=downsampled_shape,
        chunks=(64, 64, 64),
        dtype=cc_labels_crackle.dtype,
        fill_value=0,
        write_empty_chunks=False,
        compressor=zarr.Blosc(cname='zstd', clevel=3)
    )
    skeleton_segment_idx = 1

    all_tracks = []
    for z_chunk_min in range(z_min, z_max, skeletonisation_chunk_depth):
        print(f'skeletonising chunk starting at slice {z_chunk_min}')

        cc_labels = cc_zarr_array[z_chunk_min : z_chunk_min + skeletonisation_chunk_depth]

        print('  skeletonising')
        skeletons = kimimaro.skeletonize(
          cc_labels,
          teasar_params={
            "scale": 2.,
            "const": 10,
            "pdrf_scale": 100000,
            "pdrf_exponent": 4,
          },
          anisotropy=(1, 1, 1),
          dust_threshold=dust_threshold,
          fix_branching=True,  # default True
          fix_borders=True,  # default True
          fill_holes=False,  # default False
          progress=True,  # default False, show progress bar
          parallel=4,  # <= 0 all cpu, 1 single process, 2+ multiprocess
          parallel_chunk_size=100,  # how many skeletons to process before updating progress bar
        )
        cc_labels_shape = cc_labels.shape
        del cc_labels

        skeleton_volume = np.zeros(cc_labels_shape, dtype=np.uint32)
        for skeleton in tqdm(skeletons.values(), desc='  extracting simple paths'):
            while True:
                if len(skeleton.edges) == 0:
                    break
                paths = skeleton.interjoint_paths()  # can't use return_indices here since the indices are valid only within each connected component, not for the full skeleton
                longest_path_vertex_zyxs = max(paths, key=len)
                if len(longest_path_vertex_zyxs) < 10:
                    break
                coords = longest_path_vertex_zyxs.astype(np.int64)
                skeleton_volume[coords[:, 0], coords[:, 1], coords[:, 2]] = skeleton_segment_idx
                if False:  # useful for visualisation
                    for delta_z in range(-4, 4):
                        skeleton_volume[coords[:, 0] + delta_z, coords[:, 1], coords[:, 2]] = skeleton_segment_idx
                        skeleton_volume[coords[:, 0] + delta_z, coords[:, 1] + 1, coords[:, 2]] = skeleton_segment_idx
                        skeleton_volume[coords[:, 0] + delta_z, coords[:, 1] + 1, coords[:, 2] + 1] = skeleton_segment_idx
                        skeleton_volume[coords[:, 0] + delta_z, coords[:, 1], coords[:, 2] + 1] = skeleton_segment_idx
                skeleton_segment_idx += 1
                all_tracks.append(coords + [z_chunk_min, 0, 0])
                longest_path_vertex_indices = set(np.where((longest_path_vertex_zyxs[:, None, :] == skeleton.vertices[None, :, :]).all(axis=-1))[1])
                skeleton.edges = np.asarray([edge for edge in skeleton.edges if edge[0] not in longest_path_vertex_indices and edge[1] not in longest_path_vertex_indices], dtype=np.uint32)
        print(f'  found {skeleton_segment_idx - 1} paths')

        print('  writing to zarr')
        skeleton_zarr_array[z_chunk_min : z_chunk_min + skeletonisation_chunk_depth] = skeleton_volume

    print('writing to pickle')
    with open(skeleton_pkl_path, 'wb') as fp:
        pickle.dump(all_tracks, fp)


if __name__ == '__main__':
    main()
