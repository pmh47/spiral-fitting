
import math
import zarr
import cc3d
import torch
import pickle
import kimimaro
import numpy as np
import networkx as nx
from tqdm import tqdm
import torch.nn.functional as F
from collections import defaultdict

from segmentation_utils import scroll1_umbilicus_z_to_yx as get_umbilicus_z_to_yx, pairwise_line_segment_intersections


predictions_zarr_path = '../data/original/bruniss/Fiber-and-Surface-Models/GP-Predictions/updated_zarrs/surface2xt-updated_ome.zarr/0'
cc_zarr_path = '../out/surfaces/surface2xt-updated/cc.zarr/2'
skeleton_zarr_path = '../out/surfaces/surface2xt-updated/skeleton.zarr/2'
skeleton_pkl_path = '../out/surfaces/surface2xt-updated/skeleton.pkl'
tracks_zarr_path = '../out/surfaces/surface2xt-updated/tracks.zarr/2'
tracks_pkl_path = '../out/surfaces/surface2xt-updated/tracks.pkl'
displacements_zarr_path = '../out/surfaces/surface2xt-updated/displacements_v3.zarr/2'
displacements_pkl_path = '../out/surfaces/surface2xt-updated/displacements_v3.pkl'
normals_pkl_path = '../out/surfaces/surface2xt-updated/normals.pkl'
interwindings_pkl_path = '../out/surfaces/surface2xt-updated/interwindings_0.95-agreement.pkl'
shape_yx_transposed = True  # use for 2024-10-31 predictions at least
early_downsample_factor = 1  # applied immediately after loading, using striding
late_downsample_factor = 4  # applied after cc'ing, using max-pooling
predictions_threshold = 150
z_min, z_max = 0, 3420  # wrt target (maybe-downsampled) volume


def downsample(x, factor, min_or_max=np.max):
    assert x.ndim == 3
    x = x.reshape(x.shape[0] // factor, factor, x.shape[1] // factor, factor, x.shape[2] // factor, factor)
    x = min_or_max(x, axis=5)
    x = min_or_max(x, axis=3)
    x = min_or_max(x, axis=1)
    return x


def get_skeleton_tracks(cc_labels, skeleton_segment_idx, dust_threshold, z_offset, verbose=True):

    if verbose:
        print('  skeletonising')
    skeletons = kimimaro.skeletonize(
      cc_labels,
      teasar_params={
        "scale": 1.,
        "const": 2,
        "pdrf_scale": 100000,
        "pdrf_exponent": 4,
      },
      anisotropy=(1, 1, 1),
      dust_threshold=dust_threshold,
      fix_branching=True,  # default True
      fix_borders=True,  # default True
      fill_holes=False,  # default False
      progress=verbose,  # default False, show progress bar
      parallel=8,  # <= 0 all cpu, 1 single process, 2+ multiprocess
      parallel_chunk_size=250,  # how many skeletons to process before updating progress bar
    )
    cc_labels_shape = cc_labels.shape
    del cc_labels

    all_tracks = []
    skeleton_volume = np.zeros(cc_labels_shape, dtype=np.uint32)
    for skeleton in tqdm(skeletons.values(), desc='  extracting simple paths', disable=not verbose):
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
                    skeleton_volume[(coords[:, 0] + delta_z).clip(0, skeleton_volume.shape[0] - 1), coords[:, 1], coords[:, 2]] = skeleton_segment_idx
                    skeleton_volume[(coords[:, 0] + delta_z).clip(0, skeleton_volume.shape[0] - 1), coords[:, 1] + 1, coords[:, 2]] = skeleton_segment_idx
                    skeleton_volume[(coords[:, 0] + delta_z).clip(0, skeleton_volume.shape[0] - 1), coords[:, 1] + 1, coords[:, 2] + 1] = skeleton_segment_idx
                    skeleton_volume[(coords[:, 0] + delta_z).clip(0, skeleton_volume.shape[0] - 1), coords[:, 1], coords[:, 2] + 1] = skeleton_segment_idx
            skeleton_segment_idx += 1
            all_tracks.append(coords + [z_offset, 0, 0])
            longest_path_vertex_indices = set(np.where((longest_path_vertex_zyxs[:, None, :] == skeleton.vertices[None, :, :]).all(axis=-1))[1])
            skeleton.edges = np.asarray([edge for edge in skeleton.edges if edge[0] not in longest_path_vertex_indices and edge[1] not in longest_path_vertex_indices], dtype=np.uint32)

    if verbose:
        print(f'  found {skeleton_segment_idx - 1} paths')

    return all_tracks, skeleton_volume


def extract_horizontal():

    z_chunk_depth = 1  # wrt downsampled output volume; this is effectively the width of 'ribbons' we slice the surfaces into
    z_chunk_stride = 2
    dust_threshold = 5  # wrt downsampled output volume

    print('loading predictions zarr')
    predictions_zarr_array = zarr.open(predictions_zarr_path, mode='r')

    overall_downsample_factor = early_downsample_factor * late_downsample_factor
    downsampled_shape = np.asarray(predictions_zarr_array.shape) // overall_downsample_factor
    if shape_yx_transposed:  # bruniss predictions from 2024-10-31 have y/x *shape* transposed but not pixels!
        downsampled_shape = downsampled_shape[[0, 2, 1]]
    assert downsampled_shape[1] < downsampled_shape[2]  # true for Scroll 1 at least

    print('initialising cc zarr')
    cc_zarr_array = zarr.open(
        cc_zarr_path,
        mode='w',
        shape=downsampled_shape,
        chunks=(64, 64, 64),
        dtype=np.uint32,
        fill_value=0,
        write_empty_chunks=False,
        compressor=zarr.Blosc(cname='zstd', clevel=3)
    )

    print('initialising skeleton zarr')
    skeleton_zarr_array = zarr.open(
        skeleton_zarr_path,
        mode='w',
        shape=downsampled_shape,
        chunks=(64, 64, 64),
        dtype=np.uint32,
        fill_value=0,
        write_empty_chunks=False,
        compressor=zarr.Blosc(cname='zstd', clevel=3)
    )

    all_tracks = []
    for z_chunk_min in range(z_min, z_max, z_chunk_stride):

        z_chunk_max = min(z_chunk_min + z_chunk_depth, z_max, downsampled_shape[0] - 2)  # -2 is to work around a crackle bug
        print(f'processing slices {z_chunk_min}-{z_chunk_max}')
        predictions = predictions_zarr_array[z_chunk_min * overall_downsample_factor : z_chunk_max * overall_downsample_factor]

        print(f'  downsampling and thresholding')
        predictions = predictions[::early_downsample_factor, ::early_downsample_factor, ::early_downsample_factor]
        predictions = (predictions > predictions_threshold).astype(np.uint8)

        print('  dusting')
        cc3d.dust(predictions, threshold=dust_threshold * late_downsample_factor**3, connectivity=6, in_place=True)

        print('  finding connected components')
        cc_labels, num_ccs = cc3d.connected_components(predictions, connectivity=6, return_N=True)
        print(f'  found {num_ccs} connected components in total')

        if late_downsample_factor > 1:
            print('  downsampling')
            # FIXME: where both fg, should dither selection of label instead of taking max, to preserve connectivity of both
            cc_labels = downsample(cc_labels, factor=late_downsample_factor)

        print('  writing to zarr')
        # Note we don't renumber the cc's to be unique; this is ok provided they're only used for visualisastion/debugging
        if shape_yx_transposed:
            cc_zarr_array[z_chunk_min : z_chunk_max, :cc_labels.shape[1], :cc_labels.shape[2]] = cc_labels[:, :cc_zarr_array.shape[1], :cc_zarr_array.shape[2]]
        else:
            cc_zarr_array[z_chunk_min : z_chunk_max] = cc_labels

        tracks_for_chunk, skeleton_volume = get_skeleton_tracks(cc_labels, len(all_tracks) + 1, dust_threshold, z_chunk_min)
        all_tracks.extend(tracks_for_chunk)

        print('  writing to zarr')
        if shape_yx_transposed:
            skeleton_zarr_array[z_chunk_min : z_chunk_max, :skeleton_volume.shape[1], :skeleton_volume.shape[2]] = skeleton_volume[:, :skeleton_zarr_array.shape[1], :skeleton_zarr_array.shape[2]]
        else:
            skeleton_zarr_array[z_chunk_min : z_chunk_max] = skeleton_volume

    print('writing to pickle')
    with open(skeleton_pkl_path, 'wb') as fp:
        pickle.dump(all_tracks, fp)


def extract_vertical():

    stride = 2  # all wrt downsampled volume
    dust_threshold = 5

    print('loading predictions zarr')
    predictions_zarr_array = zarr.open(predictions_zarr_path, mode='r')

    assert early_downsample_factor == 1
    downsample_factor = late_downsample_factor
    downsampled_shape = np.asarray(predictions_zarr_array.shape) // downsample_factor
    if shape_yx_transposed:  # bruniss predictions from 2024-10-31 have y/x *shape* transposed but not pixels!
        downsampled_shape = downsampled_shape[[0, 2, 1]]
    assert downsampled_shape[1] < downsampled_shape[2]  # true for Scroll 1 at least

    min_yx = np.max(predictions_zarr_array.shape[1:])
    max_yx = 0
    for z in tqdm(range(0, predictions_zarr_array.shape[0], predictions_zarr_array.shape[0] // 20), desc='finding yx range'):
        predictions = predictions_zarr_array[z]
        yxs = np.stack(np.where(predictions > predictions_threshold), axis=-1)
        if len(yxs) > 0:
            min_yx = np.minimum(min_yx, yxs.min(axis=0))
            max_yx = np.maximum(max_yx, yxs.max(axis=0))
    min_yx //= downsample_factor
    max_yx //= downsample_factor

    print('initialising tracks zarr')
    tracks_zarr_array = zarr.open(
        tracks_zarr_path,
        mode='w',
        shape=downsampled_shape,
        chunks=(64, 64, 64),
        dtype=np.uint16,
        fill_value=0,
        write_empty_chunks=False,
        compressor=zarr.Blosc(cname='zstd', clevel=3)
    )

    def prepare_slice(predictions):
        predictions = (predictions > predictions_threshold).astype(np.uint8)
        if predictions.max() == 0:
            return None
        cc3d.dust(predictions, threshold=dust_threshold * downsample_factor**3, connectivity=6, in_place=True)
        cc_labels, num_ccs = cc3d.connected_components(predictions, connectivity=6, return_N=True)
        if downsample_factor > 1:
            # FIXME: where both fg, should dither selection of label instead of taking max, to preserve connectivity of both
            cc_labels = cc_labels.reshape(cc_labels.shape[0] // downsample_factor, downsample_factor, cc_labels.shape[1] // downsample_factor, downsample_factor,)
            cc_labels = np.max(cc_labels, axis=3)
            cc_labels = np.max(cc_labels, axis=1)
        return cc_labels

    all_tracks = []

    for y_downsampled in tqdm(range(min_yx[0], max_yx[0], stride), desc='extracting zx-plane vertical tracks'):
        predictions = predictions_zarr_array[:, y_downsampled * downsample_factor, :]
        cc_labels = prepare_slice(predictions)
        if cc_labels is None:
            continue
        tracks_for_chunk_yzx, skeleton_volume = get_skeleton_tracks(cc_labels[None], len(all_tracks) + 1, dust_threshold, z_offset=y_downsampled, verbose=False)
        all_tracks.extend([track[:, [1, 0, 2]] for track in tracks_for_chunk_yzx])
        if shape_yx_transposed:
            tracks_zarr_array[:, y_downsampled, :skeleton_volume.shape[2]] = skeleton_volume[0, :, :tracks_zarr_array.shape[2]]
        else:
            tracks_zarr_array[:, y_downsampled, :] = skeleton_volume[0]

    for x_downsampled in tqdm(range(min_yx[1], max_yx[1], stride), desc='extracting zy-plane vertical tracks'):
        predictions = predictions_zarr_array[:, :, x_downsampled * downsample_factor]
        cc_labels = prepare_slice(predictions)
        if cc_labels is None:
            continue
        tracks_for_chunk_xzy, skeleton_volume = get_skeleton_tracks(cc_labels[None], len(all_tracks) + 1, dust_threshold, z_offset=x_downsampled, verbose=False)
        all_tracks.extend([track[:, [1, 2, 0]] for track in tracks_for_chunk_xzy])
        if shape_yx_transposed:
            tracks_zarr_array[:, :skeleton_volume.shape[1], x_downsampled] = skeleton_volume[0, :, :tracks_zarr_array.shape[1]]
        else:
            tracks_zarr_array[:, :, x_downsampled] = skeleton_volume[0]

    print(f'writing {len(all_tracks)} tracks to pickle')
    with open(tracks_pkl_path, 'wb') as fp:
        pickle.dump(all_tracks, fp)


grad_crop_size = 10  # wrt early-downsampled; used for estimating normals
grad_magnitude_threshold = 10.  # ignore nearby pixels with gradients less than this when estimating normals
grad_orientable_threshold = 0.1  # ignore nearby pixels with gradients whose abs dot with from-umbilicus vector is less than this


sobel_filter_z = torch.tensor([
    [
        [1., 2., 1.],
        [2., 4., 2.],
        [1., 2., 1.],
    ], [
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
    ], [
        [-1., -2., -1.],
        [-2., -4., -2.],
        [-1., -2., -1.],
    ],
])
sobel_filters = torch.stack([
    sobel_filter_z,
    sobel_filter_z.permute(2, 0, 1),
    sobel_filter_z.permute(1, 2, 0),
]).unsqueeze(1).cuda()


def estimate_normals(predictions, current_zyxs, umbilicus_z_to_yx, z_chunk_min):

    # Find the umbilicus coordinates for the slice containing each point
    umbilicus_yxs = torch.from_numpy(
        umbilicus_z_to_yx(current_zyxs[:, 0].cpu() + z_chunk_min * late_downsample_factor)
    ).cuda()
    umbilicus_zyxs = torch.cat([current_zyxs[:, :1], umbilicus_yxs], dim=1)
    umbilicus_to_point_directions = F.normalize(current_zyxs - umbilicus_zyxs, dim=1)

    # Extract a crop around each point, and compute sobel gradients
    crops = torch.stack([
        predictions[zyx[0] - grad_crop_size // 2 : zyx[0] + grad_crop_size // 2, zyx[1] - grad_crop_size // 2 : zyx[1] + grad_crop_size // 2, zyx[2] - grad_crop_size // 2 : zyx[2] + grad_crop_size // 2]
        for zyx in torch.unbind(current_zyxs, dim=0)
    ], dim=0).to(torch.float32) # / 255.
    grads = F.conv3d(crops[:, None], sobel_filters)  # crop, zyx, z-in-crop, y-in-crop, x-in-crop

    # Ensure normals point outwards, i.e. have positive (or zero) dot-product with direction vector from umbilicus
    normalized_grads = F.normalize(grads, dim=1)
    grads_dot_from_umbilicus = (normalized_grads * umbilicus_to_point_directions[:, :, None, None, None]).sum(dim=1, keepdim=True)
    grads *= torch.where(grads_dot_from_umbilicus < 0., -1, 1.)  # don't use sign, since that may be zero (which would zero out the grad)
    grads = torch.where(grads_dot_from_umbilicus.abs() < grad_orientable_threshold, 0., grads)  # zero out gradients that aren't orientable since they're tangential to the umbilicus

    # Take the average of greater-than-threshold-magnitude gradients in each patch as an estimate
    # of the normal direction; set normal to zero if no grad is large (so no propagation for that track)
    grad_magnitudes = torch.linalg.norm(grads, dim=1, keepdim=True)
    grad_magnitudes_large = grad_magnitudes > grad_magnitude_threshold
    any_grad_large = grad_magnitudes_large.any(dim=(-1, -2, -3)).squeeze(1)
    mean_large_grads = (grads * grad_magnitudes_large).sum(dim=(-1, -2, -3)) / grad_magnitudes_large.sum(dim=(-1, -2, -3))
    normals = F.normalize(mean_large_grads, dim=1)

    return normals, any_grad_large


def sample_fg_points_stratified(predictions, stratification_grid_spacing, crop_size):
    predictions_shape_tensor = torch.tensor(predictions.shape, device=predictions.device)
    selected_zyxs = []
    for z0 in range(crop_size // 2, predictions.shape[0] - crop_size // 2, stratification_grid_spacing * late_downsample_factor):
        for y0 in range(crop_size // 2, predictions.shape[1] - crop_size // 2, stratification_grid_spacing * late_downsample_factor):
            for x0 in range(crop_size // 2, predictions.shape[2] - crop_size // 2, stratification_grid_spacing * late_downsample_factor):
                predictions_cell = predictions[
                    z0 : z0 + stratification_grid_spacing * late_downsample_factor,
                    y0 : y0 + stratification_grid_spacing * late_downsample_factor,
                    x0 : x0 + stratification_grid_spacing * late_downsample_factor,
                ]
                cell_fg_points = torch.stack(torch.where(predictions_cell > predictions_threshold), axis=-1)
                cell_fg_points += torch.tensor([z0, y0, x0], device=cell_fg_points.device)
                cell_fg_points = cell_fg_points[(cell_fg_points < predictions_shape_tensor - crop_size // 2).all(dim=1), :]
                if len(cell_fg_points) > 0:
                    selected_idx = torch.randint(len(cell_fg_points), size=[])
                    selected_zyxs.append(cell_fg_points[selected_idx])
    return torch.stack(selected_zyxs, dim=0) if len(selected_zyxs) > 0 else torch.empty([0, 3], dtype=torch.int64, device=predictions.device)


def extract_normals():

    stratification_grid_spacing = 10  # wrt downsampled volume
    z_chunk_depth = 48  # wrt the downsampled output volume
    crop_size = 10  # wrt early-downsampled; used for estimating normals

    umbilicus_z_to_yx = get_umbilicus_z_to_yx(downsample_factor=early_downsample_factor)

    print('loading predictions zarr')
    predictions_zarr_array = zarr.open(predictions_zarr_path, mode='r')

    overall_downsample_factor = early_downsample_factor * late_downsample_factor
    downsampled_shape = np.asarray(predictions_zarr_array.shape) // overall_downsample_factor
    if shape_yx_transposed:  # bruniss predictions from 2024-10-31 have y/x *shape* transposed but not pixels!
        downsampled_shape = downsampled_shape[[0, 2, 1]]
    assert downsampled_shape[1] < downsampled_shape[2]  # true for Scroll 1 at least

    all_points_and_normals = []
    for z_chunk_min in range(z_min, z_max, z_chunk_depth):

        z_chunk_max = min(z_chunk_min + z_chunk_depth, z_max, downsampled_shape[0] - 1)
        print(f'processing slices {z_chunk_min}-{z_chunk_max}')
        predictions = predictions_zarr_array[z_chunk_min * overall_downsample_factor : z_chunk_max * overall_downsample_factor]

        print(f'  downsampling and thresholding')
        predictions = predictions[::early_downsample_factor, ::early_downsample_factor, ::early_downsample_factor]
        predictions = torch.from_numpy(predictions).cuda()
        predictions_shape = torch.tensor(predictions.shape, device='cuda', dtype=torch.int64)

        print(f'  sampling foreground points')
        selected_zyxs = sample_fg_points_stratified(predictions, stratification_grid_spacing, crop_size)
        print(f'  found {len(selected_zyxs)} points')

        print(f'  calculating normals')
        normals, any_grad_large = estimate_normals(predictions, selected_zyxs, umbilicus_z_to_yx, z_chunk_min)
        selected_zyxs = selected_zyxs[any_grad_large] / late_downsample_factor + torch.tensor([z_chunk_min, 0, 0], device=selected_zyxs.device)
        points_and_normals = torch.stack([selected_zyxs, normals[any_grad_large]], dim=1)
        all_points_and_normals.extend(points_and_normals.cpu().numpy())

        del points_and_normals, normals, any_grad_large, selected_zyxs, predictions

    print('writing to pickle')
    with open(normals_pkl_path, 'wb') as fp:
        pickle.dump(np.stack(all_points_and_normals), fp)


def extract_interwindings():

    point_spacing = 5  # measured in (downsampled) voxels
    max_distance = 50  # wrt downsampled; maximum distance to search for next sheet
    agreement_threshold = 0.95  # this fraction of paths between two tracks must agree on a numbering-difference for it to be recorded

    umbilicus_z_to_yx = get_umbilicus_z_to_yx(downsample_factor=early_downsample_factor * late_downsample_factor)

    print('loading pickle')
    with open(skeleton_pkl_path, 'rb') as fp:
        tracks = pickle.load(fp)

    print('filtering tracks')
    tracks = [track for track in tracks if np.any((track[:, 0] >= z_min) & (track[:, 0] < z_max))]

    # Here we assume the tracks are horizontal, i.e. have constant z
    zs = sorted(set([int(track[0, 0]) for track in tracks]))

    def get_points_and_perpendiculars_on_track(track_yx, point_spacing, umbilicus_yx, perpendicular_smoothing_radius=5):
        cumulative_lengths = np.cumsum(np.linalg.norm(np.diff(track_yx, axis=0), axis=-1))
        sampled_yxs = []
        perpendicular_yxs = []
        current_length = point_spacing / 2  # start with a margin of half point_spacing
        idx = 0
        while True:
            while idx < len(cumulative_lengths) and cumulative_lengths[idx] < current_length:
                idx += 1
            if idx >= len(cumulative_lengths):
                break
            prev_length = cumulative_lengths[idx - 1] if idx > 0 else 0
            next_length = cumulative_lengths[idx]
            alpha = (current_length - prev_length) / (next_length - prev_length)
            point = track_yx[idx - 1] * (1 - alpha) + track_yx[idx] * alpha
            sampled_yxs.append(point)
            direction = track_yx[min(idx + perpendicular_smoothing_radius, len(track_yx) - 1)] - track_yx[max(idx - perpendicular_smoothing_radius - 1, 0)]
            perpendicular_yx = np.array([-direction[1], direction[0]])
            perpendicular_yx /= np.linalg.norm(perpendicular_yx)
            perpendicular_yxs.append(perpendicular_yx)
            current_length += point_spacing

        sampled_yxs = np.stack(sampled_yxs, axis=0)
        perpendicular_yxs = np.stack(perpendicular_yxs, axis=0)

        umbilicus_to_point = sampled_yxs - umbilicus_yx.astype(np.float32)
        umbilicus_to_point /= np.linalg.norm(umbilicus_to_point, axis=-1, keepdims=True)
        perpendiculars_dot_from_umbilicus = np.sum(perpendicular_yxs * umbilicus_to_point, axis=-1)
        perpendicular_yxs *= np.where(perpendiculars_dot_from_umbilicus < 0, -1, 1)[:, None]

        not_orientable = np.abs(perpendiculars_dot_from_umbilicus) < grad_orientable_threshold
        sampled_yxs = sampled_yxs[~not_orientable]
        perpendicular_yxs = perpendicular_yxs[~not_orientable]

        return sampled_yxs, perpendicular_yxs

    point_pairs_and_number_differences = []

    for z in zs:
        track_original_indices = [idx for idx, track in enumerate(tracks) if track[0, 0] == z]
        track_yxs = [track[:, 1:].astype(np.float32) for track in tracks if track[0, 0] == z]
        track_endpoint_yxs = [np.stack([track[:-1], track[1:]], axis=1) for track in track_yxs]  # [track], point-on-track, start/end, yx
        umbilicus_yx = umbilicus_z_to_yx(z)
        pair_to_yxs = defaultdict(list)
        for track_idx, track_yx in enumerate(tqdm(track_yxs, desc=f'finding edges for z = {z}')):

            sampled_yx, perpendicular_yx = get_points_and_perpendiculars_on_track(track_yx.astype(np.float32), point_spacing, umbilicus_yx)
            if len(sampled_yx) == 0:
                continue

            if False:
                import matplotlib.pyplot as plt
                plt.plot(track_yx[:, 1], track_yx[:, 0])
                plt.scatter(sampled_yx[:, 1], sampled_yx[:, 0], color='red')
                for point, perp in zip(sampled_yx, perpendicular_yx):
                    plt.plot([point[1], point[1] + perp[1]], [point[0], point[0] + perp[0]], color='blue')
                plt.gca().set_aspect('equal')
                plt.gca().invert_yaxis()
                plt.show()

            # Look outwards along the perpendiculars, finding the nearest track-segment that each intersects
            perpendicular_end_yx = sampled_yx + perpendicular_yx * max_distance
            nearest_intersecting_track_idx = np.full(len(sampled_yx), -1, dtype=np.int64)
            nearest_intersecting_track_distance = np.full(len(sampled_yx), np.inf, dtype=np.float32)
            nearest_intersecting_track_yx = np.zeros([len(sampled_yx), 2], dtype=np.float32)
            for other_track_idx in range(len(track_yxs)):
                if other_track_idx == track_idx:
                    continue
                # Stride of 4 in the following is just for efficiency
                if np.min(np.sum((track_yxs[other_track_idx][::4, None] - track_yx[None, ::4]) ** 2, axis=-1)) > max_distance ** 2:
                    continue
                intersects, intersection_yxs = pairwise_line_segment_intersections(np.stack([sampled_yx, perpendicular_end_yx], axis=1), track_endpoint_yxs[other_track_idx], return_yxs=True)
                # Both the above are indexed by point-along-current track, segment-of-other-track (and yx for intersection_yxs)
                for perpendicular_idx in np.where(intersects.any(axis=1))[0]:
                    distances = np.linalg.norm(intersection_yxs[perpendicular_idx, intersects[perpendicular_idx]] - sampled_yx[perpendicular_idx], axis=-1)
                    distance = distances.min()
                    if distance < nearest_intersecting_track_distance[perpendicular_idx]:
                        nearest_intersecting_track_idx[perpendicular_idx] = other_track_idx
                        nearest_intersecting_track_distance[perpendicular_idx] = distance
                        nearest_intersecting_track_yx[perpendicular_idx] = sampled_yx[perpendicular_idx] + distance * perpendicular_yx[perpendicular_idx]
            for perpendicular_idx, other_track_idx in enumerate(nearest_intersecting_track_idx):  # i.e. for each perpendicular
                if other_track_idx != -1:
                    pair_to_yxs[(int(track_idx), int(other_track_idx))].append((sampled_yx[perpendicular_idx], nearest_intersecting_track_yx[perpendicular_idx]))
        
        if False:
            import matplotlib.pyplot as plt
            for track_idx in range(len(track_yxs)):
                inner_points = track_yxs[track_idx]
                plt.plot(inner_points[:, 1], inner_points[:, 0], color='red')
                outer_track_indices = [pair[1] for pair in pair_to_count if pair[0] == track_idx]
                for outer_track_idx in outer_track_indices:
                    outer_points = track_yxs[outer_track_idx]
                    plt.plot(outer_points[:, 1], outer_points[:, 0], color='blue')
                plt.gca().set_aspect('equal')
                plt.gca().invert_yaxis()
                plt.title(f'{track_idx}: {outer_track_indices}')
                plt.show()

        # Now pair_to_count has one entry per pair of 'inner' and 'outer' tracks that are adjacent; the counts indicate how 
        # many times each pairing arose due to different points/perpendiculars on the inner track hitting the same outer track
        graph = nx.DiGraph()
        for (source, target), yxs in pair_to_yxs.items():
            graph.add_edge(source, target, count=len(yxs), yxs=yxs)

        # Entirely remove any nodes (tracks) that are part of a cycle
        for cycle in list(nx.simple_cycles(graph)):
            graph.remove_nodes_from(cycle)

        print(f'graph has {len(graph.nodes)} nodes and {len(graph.edges)} edges')

        # The choice to include directly- and two-connected pairs is for efficiency and somewhat arbitrary;
        # in theory we could include all pairs of tracks that are connected by a path of arbitrary length
        def two_connected_pairs():
            for source, midpoint in graph.edges:
                for target in graph.successors(midpoint):
                    yield source, target

        num_pairs = 0
        # for source, target in set(graph.edges) | set(two_connected_pairs()):
        for source, target in graph.edges:
            paths = nx.all_simple_edge_paths(graph, source, target, cutoff=5)
            number_difference_to_count = defaultdict(int)
            for path in paths:
                count_for_path = math.prod([graph.edges[edge]['count'] for edge in path])
                number_difference_to_count[len(path)] += count_for_path
            majority_number_difference = max(number_difference_to_count, key=number_difference_to_count.get)
            majority_count = number_difference_to_count[majority_number_difference]
            if majority_count / sum(number_difference_to_count.values()) >= agreement_threshold:
                if majority_number_difference == 1:
                    start_and_end_yxs = graph.edges[(source, target)]['yxs']
                    point_pairs_and_number_differences.extend([
                        (np.concatenate([np.float32([z]), start_yx]), np.concatenate([np.float32([z]), end_yx]), majority_number_difference)
                        for start_yx, end_yx in start_and_end_yxs
                    ])
                    num_pairs += len(start_and_end_yxs)
                else:
                    pass
        print(f'recorded {num_pairs} displacements')

    print('writing to pickle')
    with open(interwindings_pkl_path, 'wb') as fp:
        pickle.dump(point_pairs_and_number_differences, fp)


if __name__ == '__main__':
    np.random.seed(0)
    torch.random.manual_seed(0)
    extract_horizontal()
    extract_vertical()
    extract_normals()
    extract_interwindings()

