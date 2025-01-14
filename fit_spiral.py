
import os
import zarr
import torch
import wandb
import pickle
import kornia
import trimesh
import datetime
import numpy as np
import scipy.ndimage
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from tqdm import tqdm
import pyro.distributions
from einops import rearrange

from segmentation_utils import scroll1_umbilicus_z_to_yx, scroll1_z_to_gt_min_max_yx, interp1d, pairwise_line_segment_intersections


scroll_zarr_path = '../data/zarr/Scroll1_masked.zarr/2'
predictions_zarr_path = '../data/original/bruniss/Fiber-and-Surface-Models/GP-Predictions/updated_zarrs/surface2xt-updated_ome.zarr/2'
horizontal_fibres_pkl_path = '../out/fibres/2024-10-31_bruniss_hz-only/skeleton.pkl'
vertical_fibres_pkl_path = '../out/fibres/2024-12-26_bruniss_vt-reg/skeleton.pkl'
tracks_pkl_paths = [
    '../out/surfaces/1213_aug_erode_threshold/skeleton.pkl',
    '../out/surfaces/1213_aug_erode_threshold/tracks.pkl',
]
interwindings_path = '../out/surfaces/1213_aug_erode_threshold/interwindings_0.95-agreement.pkl'
points_and_normals_path = '../out/surfaces/1213_aug_erode_threshold/normals.pkl'
gp_mesh_path = '../data/original/Scroll1/PHercParis4.volpkg/paths/20231231235900_GP/20231231235900_GP.obj'
cache_path = '../cache'
downsample_factor = 4
spiral_outward_sense = 'ACW'  # CW | ACW

default_config = {
    'random_seed': 0,
    'learning_rate': 5.e-4,
    'cosine_lr_schedule': False,
    'num_training_steps': 20000,
    'num_euler_timesteps': 16,
    'num_flow_timesteps': 1,
    'flow_bounds_z_margin': 40,
    'flow_bounds_radius': 450,
    'flow_voxel_resolution': 12,
    'gap_expander_logit_resolution': 32,
    'fibre_loss_mean_not_ptp': True,
    'fibre_loss_margin': 0.1,
    'num_points_per_fibre': 100,
    'sample_fibres_by_length': True,
    'winding_number_num_pairs': 2000,
    'normals_num_points': 2000,
    'regularisation_num_points': 1500,
    'radius_num_fibres': 200,
    'radius_num_tracks': 500,
    'loss_weight_fibre_track_radius': 5.e0,
    'loss_weight_fibre_track_dt': 4.e0,
    'loss_weight_fibre_direction': 5.e0,
    'loss_weight_surface_count': 1.e1,
    'loss_weight_surface_normal': 2.e2,
    'loss_weight_stretch': 2.e2,
    'loss_weight_umbilicus': 1.,
    'loss_start_fibre_track_dt': 10000,
    'loss_stop_surface_count': 20000,
}


def get_spiral_yxs(num_windings, dr_per_winding, inter_point_spacing, group_by_winding=False):

    # Note this is not differentiable wrt dr_per_winding nor inter_point_spacing!

    # r = b * theta => b = drpw / 2pi
    # ...so r = dr_per_winding * theta / (2 * pi)

    # Kth winding has average radius (K + 0.5) * dr_per_winding => circumference (K + 0.5) * dr_per_winding * 2 * pi
    # ...so should have (K + 0.5) * dr_per_winding * 2 * pi / inter_point_spacing steps
    # can construct these thetas directly, then r's via formula

    thetas = [
        winding_idx * 2 * torch.pi + torch.arange(
            0, 2 * np.pi,
            step=inter_point_spacing / (winding_idx + 0.5) / float(dr_per_winding),
            device='cuda'
        )
        for winding_idx in range(num_windings)
    ]
    radii = [dr_per_winding * thetas_for_winding / (2 * torch.pi) for thetas_for_winding in thetas]

    yxs = [
        torch.stack([torch.sin(thetas_for_winding), torch.cos(thetas_for_winding)], dim=-1) * radii_for_winding[:, None]
        for thetas_for_winding, radii_for_winding in zip(thetas, radii)
    ]

    if group_by_winding:
        return yxs
    else:
        return torch.cat(yxs, dim=0)


def get_winding_xy(winding_idx, theta, dr_per_winding):
    winding_radius = winding_idx * dr_per_winding + theta / (2 * np.pi) * dr_per_winding
    return torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1) * winding_radius[..., None]


def get_theta(relative_yx):
    relative_yx = torch.stack([
        relative_yx[..., 0],
        torch.where(relative_yx[..., 1].abs() < 1.e-10, 1.e-10, relative_yx[..., 1]),
    ], dim=-1)  # avoid NaN gradients from atan2 / sqrt
    theta = torch.arctan2(relative_yx[..., 0], relative_yx[..., 1]) % (2 * np.pi)  # [0, 2pi]; zero along x-axis
    return theta, relative_yx


def get_theta_and_radii(relative_yx, dr_per_winding):
    theta, relative_yx = get_theta(relative_yx)
    radius = torch.linalg.norm(relative_yx, dim=-1)
    # The spiral has radius 0 at winding angle 0 then increases linearly at rate dr_per_winding
    # Note get_fibre_loss assumes this form!
    shifted_radius = radius - theta / (2 * np.pi) * dr_per_winding
    return theta, radius, shifted_radius


def get_bounding_windings(relative_yx, dr_per_winding):
    # The spiral has radius 0 at winding angle 0 then increases linearly at rate dr_per_winding
    # Want to find the two windings that bracket yx
    # If theta=+eps, then these are given by floor/ceil of radius / dr_per_winding
    # For other theta, we shift the point radially so 'as if' it were at theta=0
    theta, radius, shifted_radius = get_theta_and_radii(relative_yx, dr_per_winding)
    inner_winding = torch.floor(shifted_radius / dr_per_winding)
    outer_winding = torch.ceil(shifted_radius / dr_per_winding)
    return theta, radius, inner_winding, outer_winding


def get_spiral_density(relative_yx, dr_per_winding=10., sigma=3.):

    theta, radius, inner_winding, outer_winding = get_bounding_windings(relative_yx, dr_per_winding)
    def evaluate_kernel(winding_idx):
        winding_xy = get_winding_xy(winding_idx, theta, dr_per_winding)
        distance = torch.linalg.norm(winding_xy.flip(-1) - relative_yx, dim=-1)
        return torch.exp(-distance ** 2 / sigma ** 2)
    result = evaluate_kernel(inner_winding) + evaluate_kernel(outer_winding)
    return result.clip(0., 1.)


class ExplicitFlowField(nn.Module):

    def __init__(self, resolution, spatial_scale_factor=6, lr_scale_factor=1.e-1):
        super().__init__()
        self.flow_scales = [1., lr_scale_factor]
        self.flows = nn.ParameterList([
            nn.Parameter(torch.zeros([cfg['num_flow_timesteps'], 3, *shape]))
            for shape in [
                [resolution[0] // spatial_scale_factor, resolution[1] // spatial_scale_factor, resolution[2] // spatial_scale_factor],
                resolution,
            ]
        ])

    def __call__(self, t):
        flow_shapes = np.asarray([flow.shape[2:] for flow in self.flows])
        max_shape = tuple(flow_shapes.max(axis=0))
        if cfg['num_flow_timesteps'] == 1:
            t_scaled = 0.
        else:
            t_scaled = (t.clamp(-1. + 1.e-4, 1. - 1.e-4) + 1) / 2 * (cfg['num_flow_timesteps'] - 1)
        t_idx_before = int(t_scaled)
        flows_interpolated = [
            rearrange(
                F.interpolate(flow[t_idx_before : t_idx_before + 2], size=max_shape, mode='trilinear'),
              't zyx z y x -> t z y x zyx'
            ) * flow_scale
            for flow, flow_scale in zip(self.flows, self.flow_scales)
        ]
        flows_interpolated = [
            torch.lerp(flow_interpolated[0], flow_interpolated[1], t_scaled % 1.) if cfg['num_flow_timesteps'] > 1 else flow_interpolated[0]
            for flow_interpolated in flows_interpolated
        ]
        return sum(flows_interpolated)


def sample_field(zyx, field):
    # zyx :: *, zyx; field :: z, y, x, zyx
    zyx = zyx / torch.tensor(field.shape[:-1], device=zyx.device) * 2. - 1.  # z, y, x, zyx
    orig_shape = zyx.shape
    zyx = zyx.view(1, -1, 1, 1, 3)
    field_samples = F.grid_sample(
        input=rearrange(field, 'z y x zyx -> 1 zyx z y x'),
        grid=zyx.flip(-1),
        align_corners=True,
        mode='bilinear',
        padding_mode='border',
    )  # 1, zyx, n, 1, 1
    return field_samples.squeeze(0).squeeze(-2).squeeze(-1).T.view(*orig_shape[:-1], 3)  # *, zyx


class EulerDiffeomorphicTransform(pyro.distributions.transforms.Transform):

    domain = pyro.distributions.constraints.real_vector
    codomain = domain

    def __init__(self, flow_net, flow_min_corner_zyx, flow_max_corner_zyx, timesteps, truncate_at_step=None, event_dim=0, cache_size=0):
        super().__init__(cache_size=cache_size)
        self.flow_net = flow_net
        self.flow_min_corner_zyx = flow_min_corner_zyx
        self.flow_max_corner_zyx = flow_max_corner_zyx
        self.timesteps = timesteps
        self.truncate_at_step = truncate_at_step
        self._event_dim = event_dim
        self._flow_zyx_by_t = []
        for timestep in range(timesteps):
            t = torch.tensor(timestep / (timesteps - 1) * 2 - 1, device=flow_min_corner_zyx.device)
            self._flow_zyx_by_t.append(self.flow_net(t) / timesteps)

    def _call(self, input_zyx, inverse=False):

        # Euler integration of the temporally-varying flow to give a diffeomorphism
        # This assumes the flow & diffeomorphism represent shifts in normalised units [0,1] over the flow region
        current_zyx_scaled = (input_zyx - self.flow_min_corner_zyx) / (self.flow_max_corner_zyx - self.flow_min_corner_zyx)
        for timestep in range(self.timesteps if self.truncate_at_step is None else self.truncate_at_step):
            if inverse:
                flow_zyx = -self._flow_zyx_by_t[self.timesteps - 1 - timestep]
            else:
                flow_zyx = self._flow_zyx_by_t[timestep]
            shifts_zyx = sample_field(current_zyx_scaled * torch.tensor(flow_zyx.shape[:-1], device=current_zyx_scaled.device), flow_zyx)
            current_zyx_scaled = current_zyx_scaled + shifts_zyx
        transformed_zyx = current_zyx_scaled * (self.flow_max_corner_zyx - self.flow_min_corner_zyx) + self.flow_min_corner_zyx

        return transformed_zyx

    def _inverse(self, input_yx):
        return self._call(input_yx, inverse=True)


class GapExpanderParams(nn.Module):

    def __init__(self, resolution, min_z, max_z, num_windings, dr_per_winding):
        super().__init__()
        self.num_by_winding = (2 * torch.pi * (torch.arange(1, num_windings) + 0.5) * dr_per_winding / resolution + 0.5).to(torch.int64)
        self.num_z = int((max_z - min_z) / resolution)
        self.logits = nn.Parameter(torch.zeros([1, 1, self.num_z, sum(self.num_by_winding)]))


class GapExpandingTransform(pyro.distributions.transforms.Transform):

    domain = pyro.distributions.constraints.real_vector
    codomain = domain

    def __init__(self, params, dr_per_winding, min_z, max_z, event_dim=0, cache_size=0):
        super().__init__(cache_size=cache_size)
        self.params = params
        self.dr_per_winding = dr_per_winding
        self.min_z = min_z
        self.max_z = max_z

    def get_transformed_winding_radii(self, theta, z):
        # This returns the sequence of winding radii (true, not shifted) for the radials given by theta and z
        num_windings = len(self.params.num_by_winding)
        winding_first_logit_idx = torch.cat([torch.zeros([1]), torch.cumsum(self.params.num_by_winding, dim=0)]).to(theta.device)
        theta_normalised = theta / (2 * torch.pi)
        winding_coords = torch.lerp(winding_first_logit_idx[:-1], winding_first_logit_idx[1:], theta_normalised[..., None])  # *, winding-idx
        winding_coords_normalised = winding_coords / winding_first_logit_idx[-1] * 2 - 1
        z_normalised = (z - self.min_z) / (self.max_z - self.min_z) * 2 - 1
        # Pin the 0th logit (i.e. theta=0 on 1th winding) to be zero, to avoid a jump going from winding #0 to #1
        logits = torch.cat([torch.zeros_like(self.params.logits[..., :1]), self.params.logits[..., 1:]], dim=-1)  # 1, 1, z, winding-angle
        # Note the 0th logit/scale/distance here adjusts the gap directly outside the 0th winding (with the 0th winding being always canonical)
        logits_by_winding = F.grid_sample(
            logits,
            torch.stack([winding_coords_normalised, z_normalised[..., None].tile(num_windings)], dim=-1).view(1, -1, num_windings, 2),
            mode='bilinear',
            padding_mode='border',
            align_corners=True,
        ).squeeze(1).squeeze(0).view(*theta.shape, num_windings)  # *, winding-idx
        scales_by_winding = torch.exp(logits_by_winding * 1.e1)
        inter_winding_distances = self.dr_per_winding * scales_by_winding
        winding_zero_radii = self.dr_per_winding * theta_normalised
        winding_radii = winding_zero_radii[..., None] + torch.cat([torch.zeros_like(inter_winding_distances[..., :1]), torch.cumsum(inter_winding_distances, dim=-1)[..., :-1]], dim=-1)
        return winding_radii

    def _call(self, input_zyx):
        theta, original_radius, inner_winding, _ = get_bounding_windings(input_zyx[..., 1:], self.dr_per_winding)
        transformed_winding_radii = self.get_transformed_winding_radii(theta, input_zyx[..., 0])
        inner_winding_clipped = inner_winding.to(torch.int64).clip(max=transformed_winding_radii.shape[-1] - 2)
        transformed_inner_radius = transformed_winding_radii[*np.indices(input_zyx.shape[:-1]), inner_winding_clipped]
        transformed_outer_radius = transformed_winding_radii[*np.indices(input_zyx.shape[:-1]), inner_winding_clipped + 1]
        original_inner_radius = (inner_winding_clipped + theta / (2 * torch.pi)) * self.dr_per_winding
        original_outer_radius = original_inner_radius + self.dr_per_winding
        frac = (original_radius - original_inner_radius) / (original_outer_radius - original_inner_radius)
        transformed_radius = torch.lerp(transformed_inner_radius, transformed_outer_radius, frac)
        delta_radius = transformed_radius - original_radius
        outward_direction = torch.cat([torch.zeros_like(input_zyx[..., :1]), F.normalize(input_zyx[..., 1:], dim=-1)], dim=-1)
        transformed_zyx = input_zyx + outward_direction * delta_radius[..., None]
        return transformed_zyx

    def _inverse(self, input_zyx):
        theta, transformed_radius, _ = get_theta_and_radii(input_zyx[..., 1:], self.dr_per_winding)
        transformed_winding_radii = self.get_transformed_winding_radii(theta, input_zyx[..., 0])
        inner_winding_indices = torch.searchsorted(transformed_winding_radii, transformed_radius[..., None]).squeeze(-1) - 1
        inner_winding_clipped = inner_winding_indices.clip(min=0, max=transformed_winding_radii.shape[-1] - 2)  # if shifted_radius is exactly zero, avoid this being -1

        transformed_inner_radius = transformed_winding_radii[*np.indices(input_zyx.shape[:-1]), inner_winding_clipped]
        transformed_outer_radius = transformed_winding_radii[*np.indices(input_zyx.shape[:-1]), inner_winding_clipped + 1]
        original_inner_radius = (inner_winding_clipped + theta / (2 * torch.pi)) * self.dr_per_winding
        original_outer_radius = original_inner_radius + self.dr_per_winding
        frac = (transformed_radius - transformed_inner_radius) / (transformed_outer_radius - transformed_inner_radius)
        original_radius = torch.lerp(original_inner_radius, original_outer_radius, frac)
        delta_radius = original_radius - transformed_radius
        outward_direction = torch.cat([torch.zeros_like(input_zyx[..., :1]), F.normalize(input_zyx[..., 1:], dim=-1)], dim=-1)
        transformed_zyx = input_zyx + outward_direction * delta_radius[..., None]

        return transformed_zyx


class VaryingScaleTransform(pyro.distributions.transforms.Transform):

    # This scales in the yx plane by a z-dependent value

    domain = pyro.distributions.constraints.real_vector
    codomain = domain

    def __init__(self, scale_logits, min_z, max_z, event_dim=0, cache_size=0):
        super().__init__(cache_size=cache_size)
        self.min_z = min_z
        self.max_z = max_z
        self.scale_logits = scale_logits  # z, yx

    def _call(self, input_zyx, inverse=False):
        zs = input_zyx[..., :1]
        normalised_zs = (zs.view(-1) - self.min_z) / (self.max_z - self.min_z) * 2 - 1
        log_scale_yx = F.grid_sample(
            rearrange(self.scale_logits, 'z yx -> 1 yx z 1'),
            torch.stack([torch.zeros_like(normalised_zs), normalised_zs], dim=-1)[None, None],
            padding_mode='border',
            align_corners=True
        ).squeeze(2).squeeze(0).T.view(*input_zyx.shape[:-1], 2)
        scale_yx = torch.exp(log_scale_yx * (-1 if inverse else 1))
        return torch.cat([zs, input_zyx[..., 1:] * scale_yx], dim=-1)

    def _inverse(self, input_zyx):
        return self._call(input_zyx, inverse=True)


class UmbilicusTransform(pyro.distributions.transforms.Transform):

    # This translates in the yx plane by a z-dependent value (i.e. shears the volume) s.t. the origin is moved to the umbilicus

    domain = pyro.distributions.constraints.real_vector
    codomain = domain

    def __init__(self, umbilicus_zyx, event_dim=2, cache_size=0):
        super().__init__(cache_size=cache_size)
        self._event_dim = event_dim
        yx_filtered = scipy.ndimage.gaussian_filter1d(umbilicus_zyx[:, 1:].cpu().numpy(), sigma=75., axis=0, mode='nearest')
        self._yx = torch.from_numpy(yx_filtered).to(umbilicus_zyx.device)
        self._z = umbilicus_zyx[:, :1]

    def _call(self, input_zyx, inverse=False):
        centre_yx = interp1d(input_zyx[..., 0], self._z.to(input_zyx.device), self._yx.to(input_zyx.device))
        return input_zyx + torch.cat([torch.zeros_like(centre_yx[..., :1]), centre_yx], dim=-1) * (-1 if inverse else 1)

    def _inverse(self, input_zyx):
        return self._call(input_zyx, inverse=True)


class SpiralAndTransform(nn.Module):

    def __init__(self, flow_timesteps, flow_min_corner_zyx, flow_max_corner_zyx, umbilicus_zyx):

        super().__init__()

        self.flow_timesteps = flow_timesteps
        self.flow_min_corner_zyx = flow_min_corner_zyx
        self.flow_max_corner_zyx = flow_max_corner_zyx
        self.spiral_intensity = 200 / 255
        self.dr_per_winding_scale = 10.  # larger value increases effective learning rate

        self.umbilicus_transform = UmbilicusTransform(umbilicus_zyx)
        self.dr_per_winding_logit = nn.Parameter(torch.tensor(10. / self.dr_per_winding_scale, dtype=torch.float32))

        flow_resolution = (flow_max_corner_zyx - flow_min_corner_zyx) // cfg['flow_voxel_resolution']
        self.flow_net = ExplicitFlowField(flow_resolution)

        self.log_scale_yx = nn.Parameter(torch.zeros([int(flow_max_corner_zyx[0] - flow_min_corner_zyx[0]) // 50, 2], dtype=torch.float32))

        self.gap_expander_params = GapExpanderParams(
            resolution=cfg['gap_expander_logit_resolution'],
            min_z=flow_min_corner_zyx[0],
            max_z=flow_max_corner_zyx[0],
            num_windings=50,
            dr_per_winding=8,  # this is a nominal (fixed) winding spacing which we only use to calculate the number of logits
        )

    @property
    def device(self):
        return self.log_scale_yx.device

    def get_slice_to_spiral_transform(self, truncate_at_step=None):
        diffeomorphism = EulerDiffeomorphicTransform(self.flow_net, self.flow_min_corner_zyx, self.flow_max_corner_zyx, timesteps=self.flow_timesteps, truncate_at_step=truncate_at_step)
        gap_expander = GapExpandingTransform(self.gap_expander_params, self.get_dr_per_winding(), self.flow_min_corner_zyx[0], self.flow_max_corner_zyx[0])
        if spiral_outward_sense == 'CW':
            maybe_flip = []
        else:
            assert spiral_outward_sense == 'ACW'
            # To make spiral go anticlockwise in slice space (going outwards from the centre), flip it horizontally
            maybe_flip = [pyro.distributions.transforms.AffineTransform(loc=0., scale=torch.tensor([1., 1., -1.], device=self.device))]
        return pyro.distributions.transforms.ComposeTransform([
            gap_expander,  # this needs to stay as the first since it makes assumptions about winding radii
            *maybe_flip,
            diffeomorphism,
            VaryingScaleTransform(self.log_scale_yx, self.flow_min_corner_zyx[0], self.flow_max_corner_zyx[0]),
            self.umbilicus_transform,
        ]).inv

    def get_dr_per_winding(self):
        return F.softplus(self.dr_per_winding_logit * self.dr_per_winding_scale)

    def get_spiral_density(self, spiral_zyx):
        return get_spiral_density(spiral_zyx[..., 1:], dr_per_winding=self.get_dr_per_winding(), sigma=1.5) * self.spiral_intensity


def load_and_slice_gp_mesh(eval_zs, vis_zs, vis_hw):

    all_zs = np.concatenate([eval_zs, vis_zs])
    cache_filename = f'{cache_path}/gp-lines_ds-{downsample_factor}_slices-{hash(tuple(all_zs))}.pkl'
    if os.path.exists(cache_filename):
        with open(cache_filename, 'rb') as fp:
            lines = pickle.load(fp)
    else:
        print('loading gp mesh')
        gp_mesh = trimesh.load(gp_mesh_path, skip_materials=True)
        gp_mesh.vertices /= downsample_factor
        lines, to_3ds, face_indices = trimesh.intersections.mesh_multiplane(gp_mesh, [0, 0, 0], [0, 0, 1], all_zs)
        # lines has one entry per slice; each entry is a tensor of lines represented by pairs of yx points, indexed [line-idx, start/end, yx]
        # since our plane is based at the origin, to_3ds only does a z-translation to move the 2d lines back to the correct plane
        assert all((to_3d[:3, :3] == np.eye(3)).all() for to_3d in to_3ds)
        assert (to_3ds[:, 2, 3] == all_zs).all()
        os.makedirs(cache_path, exist_ok=True)
        with open(cache_filename, 'wb') as fp:
            pickle.dump(lines, fp)

    eval_lines = lines[:len(eval_zs)]
    vis_lines = lines[len(eval_zs):]

    print('enumerating gp line-segments')
    z_to_equal_radius_zyx_pairs = {
        eval_zs[slice_idx]: torch.from_numpy(np.concatenate([
            eval_lines[slice_idx],  # xy
            np.full([*eval_lines[slice_idx].shape[:2], 1], eval_zs[slice_idx])  # z
        ], axis=-1)[..., ::-1].astype(np.float32))
        for slice_idx in range(len(eval_zs))
    }

    print('visualising gp line-segments')
    canvas = np.zeros([len(vis_zs), *vis_hw], dtype=np.uint8)
    for slice_idx in range(len(vis_zs)):
        yxs = vis_lines[slice_idx]
        fractions = np.asarray([0.25, 0.5, 0.75])
        interpolated = yxs[:, 0, :, None] * fractions + yxs[:, 1, :, None] * (1 - fractions)
        yxs = np.concatenate([yxs.reshape(-1, 2), interpolated.transpose(0, 2, 1).reshape(-1, 2)], axis=0)
        canvas[slice_idx, *(yxs[:, ::-1] + 0.5).astype(np.int64).T] = 1

    return z_to_equal_radius_zyx_pairs, torch.from_numpy(canvas)


@torch.inference_mode()
def evaluate_wrt_gp(slice_to_spiral_transform, dr_per_winding, z_to_umbilicus_yx, z_to_gp_equal_radius_zyx_pairs, gp_spiral_bounds):

    # First, measure how close the GP line-segments are to our nearest windings (in *spiral* space)
    # This complains if the GP is not parallel with ours, particularly strongly so if ours is dense
    # TODO: measure the distances in scroll space, to be more accurate where the diffeomorphism has large 2nd derivative
    gp_equal_radius_zyx_pairs = torch.cat([
        gp_equal_radius_zyx_pairs_for_z.cuda()
        for gp_equal_radius_zyx_pairs_for_z in z_to_gp_equal_radius_zyx_pairs.values()
    ], dim=0)
    gp_pairs_spiral_zyx = slice_to_spiral_transform(gp_equal_radius_zyx_pairs)
    theta, _, shifted_radius = get_theta_and_radii(gp_pairs_spiral_zyx[..., 1:], dr_per_winding)
    crosses_zero = (theta.amax(dim=1) > 7/8 * torch.pi) & (theta.amin(dim=1) < torch.pi / 8)  # if we cross zero, shifted_radius is not directly useable
    _, _, inner_winding, outer_winding = get_bounding_windings(gp_pairs_spiral_zyx[..., 1:], dr_per_winding)
    nearest_windings = torch.where(shifted_radius % dr_per_winding > dr_per_winding / 2, outer_winding, inner_winding)
    frac_gp_jumping_windings = (nearest_windings[:, 0] != nearest_windings[:, 1])[~crosses_zero].float().mean()

    # Ideally, along a given radial (ignoring wiggles), there should be a 1:1 pairing of our windings and GP windings
    # For many radials, we therefore find all points our and GP windings cross that radial
    # Then, just find average difference, pairing them off 1:1 from the innermost outwards
    # FIXME: this assumes the GP starts at the same papyrus winding-index in all slices -- whereas actually there's a jump at s1350
    #  relatedly, the innermost winding below may be off-by-one -- inner/outer winding doesn't really tell which is *closer* (should be
    #  whichever yields better metrics overall; arguably should decide this globally, at all radials/radii)
    num_radials = 100
    innermost_winding_idx = int(inner_winding.amin())
    mean_winding_distances = []
    for slice_z in z_to_gp_equal_radius_zyx_pairs:
        radii_by_radial, yxs_by_radial, winding_indices_by_radial = get_winding_positions_on_radials(
            torch.tensor(slice_z).cuda(),
            thetas=torch.linspace(0.01, 2. * torch.pi, num_radials + 1)[:-1],
            max_radius=gp_equal_radius_zyx_pairs[..., 1:].amax() - gp_equal_radius_zyx_pairs[..., 1:].amin(),  # FIXME: this is rather conservative
            slice_to_spiral_transform=slice_to_spiral_transform,
            dr_per_winding=dr_per_winding,
            z_to_umbilicus_yx=z_to_umbilicus_yx,
        )
        for radial_idx in range(len(radii_by_radial)):
            winding_radii = radii_by_radial[radial_idx][torch.where(winding_indices_by_radial[radial_idx] > innermost_winding_idx)[0].amin() :]
            radial_line = yxs_by_radial[radial_idx][[0, -1]]  # start/end, yx
            gp_intersects, gp_potential_intersection_yxs = pairwise_line_segment_intersections(z_to_gp_equal_radius_zyx_pairs[slice_z][..., 1:], radial_line[None], return_yxs=True)
            gp_intersection_yxs = gp_potential_intersection_yxs[gp_intersects]  # all these yxs lie on the radial
            if gp_intersection_yxs.shape[0] == 0:
                continue
            umbilicus_yx = torch.from_numpy(z_to_umbilicus_yx(slice_z).astype(np.float32))
            gp_radii = torch.linalg.norm(gp_intersection_yxs - umbilicus_yx, dim=-1)
            gp_radii = torch.sort(gp_radii)[0]
            if winding_radii.shape[0] < gp_radii.shape[0]:  # this means the windings have stretched out excessively
                winding_radii = F.pad(winding_radii, (0, gp_radii.shape[0] - winding_radii.shape[0]), value=winding_radii[-1] if winding_radii.shape[0] > 0 else 0.)
            distances = (gp_radii - winding_radii[:gp_radii.shape[0]]).abs()
            mean_winding_distances.append(distances.mean())
    mean_radial_winding_distance = torch.stack(mean_winding_distances).mean()

    print(f'evaluation wrt gp: frac_gp_jumping_windings = {frac_gp_jumping_windings:.2f}, mean_radial_winding_distance = {mean_radial_winding_distance:.1f}')

    return {
        'frac_gp_jumping_windings': frac_gp_jumping_windings.item(),
        'mean_radial_winding_distance': mean_radial_winding_distance.item(),
    }


def get_fibre_and_track_losses(slice_to_spiral_transform, dr_per_winding, num_fibres_per_step, fibre_zyxs, fibre_lengths, direction=None):
    fibre_probabilities = fibre_lengths / fibre_lengths.sum() if cfg['sample_fibres_by_length'] else None
    fibre_indices = np.random.choice(len(fibre_zyxs), min(num_fibres_per_step, len(fibre_zyxs)), p=fibre_probabilities, replace=False)
    all_slice_zyxs = []
    for fibre_idx in fibre_indices:
        zyxs_for_fibre = fibre_zyxs[fibre_idx]
        assert len(zyxs_for_fibre) >= 2
        slice_zyxs = zyxs_for_fibre[np.random.choice(len(zyxs_for_fibre), min(cfg['num_points_per_fibre'], len(zyxs_for_fibre)), replace=False)]
        all_slice_zyxs.append(slice_zyxs)

    lengths = tuple(map(len, all_slice_zyxs))
    all_slice_zyxs = torch.from_numpy(np.concatenate(all_slice_zyxs, axis=0).astype(np.float32)).cuda()

    all_spiral_zyxs = slice_to_spiral_transform(all_slice_zyxs)
    all_theta, _, all_shifted_radii = get_theta_and_radii(all_spiral_zyxs[..., 1:], dr_per_winding)

    end_indices = torch.cumsum(torch.tensor(lengths), dim=0)
    for idx in range(len(lengths)):
        begin = end_indices[idx - 1].item() if idx > 0 else 0
        end = end_indices[idx].item()
        theta = all_theta[begin : end]  # these are views!
        shifted_radii = all_shifted_radii[begin : end]
        gap_degrees_across_theta_zero = ((2 * torch.pi - theta.amax()) + theta.amin()) * 180 / torch.pi
        if gap_degrees_across_theta_zero < 5.:
            # In this case, the fibre/track approaches theta's discontinuity at 0=2pi very closely on both sides, hence presumably crosses it
            # We assume that the fibre does not make >1 winding, hence does not go 'all the way round' away from theta=0
            # Thus it can be chopped into two sections, split at theta=0, with one section having 'large' thetas and the other 'small'
            # We need to find the biggest gap, and chop it at this gap; this could be almost any theta
            # First sort the thetas; then it's simply the biggest gap (since the desired gap is definitely not across the discontinuity)
            theta_order = torch.argsort(theta)
            theta[:] = theta[theta_order]
            shifted_radii[:] = shifted_radii[theta_order]
            biggest_gap_idx = theta.diff().argmax()  # index into now-sorted theta and shifted-radii; the big gap is between this and the next
            # Now we know which 'end' of the track each point belongs to; we need to re-adjust those on one end (we arbitrarily choose the
            # 'large theta' end) to have radii that are as if they were at the small end
            # Note this makes an assumption on how get_theta_and_radii calculates shifted_radii!
            # Specifically, it sets shifted_radius = radius - theta / (2 * np.pi) * dr_per_winding; we reverse this, then re-apply with new thetas
            # This means moving the shifted-radii outward by one winding-spacing
            shifted_radii[biggest_gap_idx + 1:] += dr_per_winding.detach()

    # FIXME: for vertical fibres (and vertical surface-tracks!), the above check/adjustment is not good
    #  In particular, these may 'meander' across the theta=0 discontinuity, without a single definitive crossing and a large gap for distant theta
    #  We could just drop these tracks entirely; this is the simplest not-wrong thing to do
    #  Note for vertical surface-tracks we don't have orientation=v set, hence can't just gate on this

    spiral_zyxs_by_track = torch.nn.utils.rnn.pad_sequence(torch.split(all_spiral_zyxs, lengths), batch_first=True)  # track, point-in-track, zyx
    shifted_radii_by_track = torch.nn.utils.rnn.pad_sequence(torch.split(all_shifted_radii, lengths), batch_first=True)  # track, point-in-track
    thetas_by_track = torch.nn.utils.rnn.pad_sequence(torch.split(all_theta, lengths), batch_first=True)
    mask_by_track = torch.nn.utils.rnn.pad_sequence([torch.ones(length, dtype=torch.bool) for length in lengths], batch_first=True).to(all_theta.device)
    lengths = torch.tensor(lengths, device=all_theta.device)

    too_near_umbilicus = ((shifted_radii_by_track < dr_per_winding * 5) & (shifted_radii_by_track != 0)).any(dim=1)
    spiral_zyxs_by_track = spiral_zyxs_by_track[~too_near_umbilicus, :, :]  # ignore the innermost few windings
    shifted_radii_by_track = shifted_radii_by_track[~too_near_umbilicus, :]
    thetas_by_track = thetas_by_track[~too_near_umbilicus, :]
    mask_by_track = mask_by_track[~too_near_umbilicus, :]
    lengths = lengths[~too_near_umbilicus]

    mean_radii = shifted_radii_by_track.sum(dim=1) / lengths
    if cfg['fibre_loss_mean_not_ptp']:
        radius_deviations = (shifted_radii_by_track - mean_radii[:, None]).abs()
        radius_deviations_hinge = F.relu(radius_deviations - dr_per_winding.detach() * cfg['fibre_loss_margin'])
        radius_deviations_hinge = radius_deviations_hinge * mask_by_track
        mean_radius_deviation = (radius_deviations_hinge.sum(dim=1) / lengths).mean()
    else:
        # TODO: vectorise as above...
        raise NotImplementedError
        ptp = F.relu(shifted_radii.amax() - shifted_radii.amin() - dr_per_winding.detach() * cfg['fibre_loss_margin'])
        mean_radius_deviation = ptp

    if direction == 'h':
        mean_zs = spiral_zyxs_by_track[..., 0].sum(dim=1) / lengths
        z_deviations = (spiral_zyxs_by_track[..., 0] - mean_zs[..., None]).abs()
        z_deviations_hinge = F.relu(z_deviations - dr_per_winding.detach() * cfg['fibre_loss_margin'])
        z_deviations_hinge = z_deviations_hinge * mask_by_track
        mean_z_or_theta_deviation = (z_deviations_hinge.sum(dim=1) / lengths).mean()
    elif direction == 'v':
        sincos_theta = torch.stack([torch.sin(thetas_by_track), torch.cos(thetas_by_track)], axis=-1)
        mean_sincos = (sincos_theta * mask_by_track[:, :, None]).sum(dim=1) / lengths[:, None]
        sincos_deviations = (sincos_theta - mean_sincos[:, None]).abs() * (30. * dr_per_winding.detach())  # scaling by 30 * drpw is arbitrary, to get it in similar units to z-deviation
        sincos_deviations = sincos_deviations * mask_by_track[:, :, None]
        mean_z_or_theta_deviation = (sincos_deviations.sum(dim=1) / lengths[:, None]).mean()
    elif direction is None:
        mean_z_or_theta_deviation = torch.zeros([])
    else:
        assert False

    modulus = mean_radii % dr_per_winding
    nearest_winding_shifted_radius = torch.where(modulus < dr_per_winding / 2, mean_radii - modulus, mean_radii + dr_per_winding - modulus)
    track_to_nearest_winding_distances = (shifted_radii_by_track - nearest_winding_shifted_radius[..., None]).abs()
    track_to_nearest_winding_distances = track_to_nearest_winding_distances * mask_by_track
    mean_distance = (track_to_nearest_winding_distances.sum(dim=1) / lengths).mean()

    return mean_radius_deviation, mean_distance, mean_z_or_theta_deviation


def get_winding_number_loss(slice_to_spiral_transform, dr_per_winding, point_pairs_and_number_differences):

    pair_indices = np.random.randint(len(point_pairs_and_number_differences), size=[cfg['winding_number_num_pairs']])

    all_inner_and_outer_slice_zyxs = []
    all_relative_numbers = []
    for pair_idx in pair_indices:
        inner_slice_zyx, outer_slice_zyx, relative_number = point_pairs_and_number_differences[pair_idx]
        inner_and_outer_slice_zyxs = np.stack([inner_slice_zyx, outer_slice_zyx], axis=0)
        all_inner_and_outer_slice_zyxs.append(torch.from_numpy(inner_and_outer_slice_zyxs).cuda(non_blocking=True).to(torch.float32))
        all_relative_numbers.append(relative_number)
    all_inner_and_outer_slice_zyxs = torch.stack(all_inner_and_outer_slice_zyxs, dim=0)
    all_relative_numbers = torch.tensor(all_relative_numbers).cuda(non_blocking=True)

    all_inner_and_outer_spiral_zyxs = slice_to_spiral_transform(all_inner_and_outer_slice_zyxs)

    # Note we use radii (not shifted_radii) here to avoid potential issues where a pair crosses theta=0
    # This introduces a bias if the pair is not actually normal to the spiral-winding
    _, radii, _ = get_theta_and_radii(all_inner_and_outer_spiral_zyxs[..., 1:], dr_per_winding)
    inner_radii, outer_radii = torch.unbind(radii, dim=1)
    radius_differences = outer_radii - inner_radii
    expected_radius_differences = all_relative_numbers * dr_per_winding
    radius_difference_abs_errors = (radius_differences - expected_radius_differences).abs()
    too_near_umbilicus = inner_radii < dr_per_winding * 5  # ignore the innermost few windings
    radius_difference_abs_errors = radius_difference_abs_errors[~too_near_umbilicus]

    return radius_difference_abs_errors.mean()


def get_stratified_normals_loss(slice_to_spiral_transform, points_and_normals):

    point_indices = np.random.randint(len(points_and_normals), size=[cfg['normals_num_points']])
    slice_zyxs, slice_normal_zyxs = torch.unbind(torch.from_numpy(points_and_normals[point_indices]).cuda(), dim=1)

    spiral_zyxs = slice_to_spiral_transform(slice_zyxs)
    spiral_outward_direction_yx = F.normalize(spiral_zyxs[:, 1:], dim=1)  # since the spiral centre is at the origin, moving in the direction of a given point means moving radially
    expected_spiral_normals_zyx = torch.cat([torch.zeros_like(spiral_outward_direction_yx[..., :1]), spiral_outward_direction_yx], dim=1)
    slice_shifted_zyxs = slice_zyxs + slice_normal_zyxs
    spiral_shifted_zyxs = slice_to_spiral_transform(slice_shifted_zyxs)
    predicted_spiral_normals_zyx = F.normalize(spiral_shifted_zyxs - spiral_zyxs, dim=1)
    cosine_distances = 1. - (expected_spiral_normals_zyx * predicted_spiral_normals_zyx).sum(dim=1)

    normals_loss = cosine_distances.mean(dim=0)

    return normals_loss


def get_stretch_regularisation_loss(slice_to_spiral_transform, points_and_normals):

    # First sample spatially-uniform points on windings in scroll space and get the surface normals
    point_indices = np.random.randint(len(points_and_normals), size=[cfg['regularisation_num_points']])
    scroll_zyx, scroll_normal_zyx = torch.unbind(torch.from_numpy(points_and_normals[point_indices]).cuda(), dim=1)
    spiral_zyx = slice_to_spiral_transform(scroll_zyx)

    # Construct a perpendicular direction vector, i.e. lying in the (scroll-space) plane of the surface
    random_zyx = torch.rand_like(scroll_normal_zyx) * 2. - 1.
    scroll_perpendicular_zyx = random_zyx - (random_zyx * scroll_normal_zyx).sum(dim=-1, keepdim=True) * scroll_normal_zyx
    scroll_perpendicular_zyx = F.normalize(scroll_perpendicular_zyx, dim=-1)

    # Perturb in the (scroll-space) winding plane, then transform this to spiral space; require
    # the distance in spiral space to be equal to that in scroll space
    epsilon = 1.
    spiral_shifted_zyx = slice_to_spiral_transform(scroll_zyx + scroll_perpendicular_zyx * epsilon)
    spiral_distance = torch.linalg.norm(spiral_shifted_zyx - spiral_zyx, dim=-1)
    # FIXME: skip (set mask to zero) if random_zyx is unluckily very-nearly-parallel with scroll_normal_zyx
    mask = 1.
    stretch_loss = ((spiral_distance - epsilon).abs() * mask).mean()

    return stretch_loss


def get_winding_positions_on_radials(slice_z, thetas, max_radius, slice_to_spiral_transform, dr_per_winding, z_to_umbilicus_yx):
    theta_slice, radius_slice = torch.meshgrid(thetas, torch.arange(1., max_radius), indexing='ij')
    radials_yx_slice = torch.from_numpy(z_to_umbilicus_yx(slice_z.cpu()).astype(np.float32)) + torch.stack([torch.sin(theta_slice), torch.cos(theta_slice)], dim=-1) * radius_slice[..., None]
    radials_zyx_slice = torch.cat([slice_z.expand(radials_yx_slice.shape[:2])[..., None], radials_yx_slice.cuda()], dim=-1)
    radials_zyx_spiral = slice_to_spiral_transform(radials_zyx_slice)
    _, _, inner_winding_idx, _ = get_bounding_windings(radials_zyx_spiral[..., 1:], dr_per_winding)
    radii_by_radial = []
    yxs_by_radial = []
    winding_indices_by_radial = []
    for radial_idx in range(inner_winding_idx.shape[0]):
        winding_change_indices = torch.where(torch.diff(inner_winding_idx[radial_idx], prepend=inner_winding_idx[radial_idx, :1]))[0].cpu()
        radii_by_radial.append(radius_slice[radial_idx, winding_change_indices])
        yxs_by_radial.append(radials_yx_slice[radial_idx, winding_change_indices])
        winding_indices_by_radial.append(inner_winding_idx[radial_idx, winding_change_indices])
    return radii_by_radial, yxs_by_radial, winding_indices_by_radial


@torch.inference_mode
def save_mesh(slice_to_spiral_transform, dr_per_winding, slice_zs, scroll_slices, scroll_slices_downsample_factor, z_begin, grid_zs, gp_spiral_bounds, out_path, name='mesh', glued=True):

    max_num_windings = 80  # truncate (conservatively) to avoid OOM
    grid_spacing = 10  # pixels, in downsampled volume
    outermost_winding_idx, gp_min_z_spiral, gp_max_z_spiral = gp_spiral_bounds
    outermost_winding_idx = min(outermost_winding_idx, max_num_windings)
    spiral_yxs = get_spiral_yxs(outermost_winding_idx, dr_per_winding, grid_spacing, group_by_winding=True)
    inner_winding = torch.cat([torch.full((len(yxs_for_winding),), winding_idx) for winding_idx, yxs_for_winding in enumerate(spiral_yxs)], dim=0)
    num_thetas_by_winding = torch.tensor(list(map(len, spiral_yxs)))
    spiral_yxs = torch.cat(spiral_yxs, dim=0)
    spiral_zs = torch.arange(gp_min_z_spiral, gp_max_z_spiral, grid_spacing, dtype=torch.float32, device=spiral_yxs.device)
    spiral_zyxs = torch.cat([spiral_zs[:, None, None].expand(-1, spiral_yxs.shape[0], 1), spiral_yxs[None, :, :].expand(spiral_zs.shape[0], -1, 2)], dim=-1)
    scroll_zyxs = slice_to_spiral_transform.inv(spiral_zyxs)

    # Note that our normals point outward (hence ink-detection must be with reverse order)
    #  If we do change this, mustn't change the behaviour of glueing below!
    normal_delta = 0.1  # 'pixels', in spiral space
    spiral_outward_direction_yx = F.normalize(spiral_zyxs[..., 1:], dim=-1)  # since the spiral centre is at the origin, moving in the direction of a given point means moving radially
    spiral_outward_direction_zyx = torch.cat([torch.zeros_like(spiral_outward_direction_yx[..., :1]), spiral_outward_direction_yx], dim=-1)
    shifted_spiral_zyxs = spiral_zyxs + spiral_outward_direction_zyx * normal_delta
    shifted_scroll_zyxs = slice_to_spiral_transform.inv(shifted_spiral_zyxs)
    normal_zyxs = F.normalize(shifted_scroll_zyxs - scroll_zyxs, dim=-1)

    def save_mesh_for_range(begin, end, suffix):

        if begin == end:
            return
        assert begin < end

        vertex_xyzs_flat = scroll_zyxs[:, begin : end].reshape(-1, 3).flip(-1)
        normal_xyzs_flat = normal_zyxs[:, begin : end].reshape(-1, 3).flip(-1)

        indices = torch.arange(scroll_zyxs.shape[0] * (end - begin)).view(scroll_zyxs.shape[0], end - begin)
        top_left = indices[:-1, :-1].flatten()
        top_right = indices[:-1, 1:].flatten()
        bottom_left = indices[1:, :-1].flatten()
        bottom_right = indices[1:, 1:].flatten()
        faces = torch.cat([
            torch.stack([bottom_left, top_left, top_right], dim=1),
            torch.stack([bottom_left, top_right, bottom_right], dim=1)
        ], dim=0)

        # In thaumato's mesh_to_surface, this texture file (only) defines the output rendering dimensions
        texture = Image.fromarray(np.full([scroll_zyxs.shape[0] * grid_spacing * downsample_factor, (end - begin) * grid_spacing * downsample_factor, 3], 255, dtype=np.uint8))

        uvs = torch.stack(torch.meshgrid(
            torch.linspace(0., 1., scroll_zyxs.shape[0] + 1)[:-1],  # 'v' (i.e. 1th coord, interpreted by vc as upward in the image)
            torch.linspace(0., 1., (end - begin) + 1)[:-1].flip(0),  # 'u' (i.e. 0th coord, interpreted by vc as rightward in the image)
            indexing='ij'
        ), dim=-1).flip(-1)  # along-scroll, around-windings, across-image / up-image

        mesh = trimesh.Trimesh(
            vertices=vertex_xyzs_flat.cpu() * downsample_factor,
            vertex_normals=normal_xyzs_flat.cpu(),
            visual=trimesh.visual.TextureVisuals(uv=uvs.reshape(-1, 2), image=texture),
            faces=faces,
        )
        mesh.visual.material.name = 'mesh'
        subfolder = f'{out_path}/meshes/{name}/{suffix}'
        os.makedirs(subfolder, exist_ok=True)
        mesh.export(f'{subfolder}/mesh.obj')

    first_theta_of_winding = torch.cat([torch.zeros([1], dtype=torch.int64), num_thetas_by_winding.cumsum(dim=0)])
    for winding_idx in tqdm(range(outermost_winding_idx), desc='saving winding meshes'):
        last_idx = (first_theta_of_winding[winding_idx + 1] + 1).clip(max=scroll_zyxs.shape[1] - 1)
        save_mesh_for_range(first_theta_of_winding[winding_idx], last_idx, f'w{winding_idx:03}')
    chunk_size = 3200 // grid_spacing
    for chunk_begin in tqdm(range(0, scroll_zyxs.shape[1], chunk_size), desc='saving chunked meshes'):
        chunk_end = min(chunk_begin + chunk_size + 1, scroll_zyxs.shape[1] - 1)
        save_mesh_for_range(chunk_begin, chunk_end, f'c{chunk_begin // chunk_size:03}')
    print('saving full mesh')
    save_mesh_for_range(0, scroll_zyxs.shape[1], 'full')

    num_lerp_steps = int(grid_spacing / scroll_slices_downsample_factor)
    mip_deltas = torch.linspace(-0.5, 0.5, 3, device=scroll_zyxs.device)
    lerp_steps = torch.linspace(0., 1., num_lerp_steps + 1, device=scroll_zyxs.device)[:-1]
    rendered_scroll = torch.zeros([])
    for mip_delta in tqdm(mip_deltas, desc='rendering mip'):
        rendering_zyxs = scroll_zyxs + mip_delta * normal_zyxs
        rendering_zyxs = torch.lerp(rendering_zyxs[:-1, None, :], rendering_zyxs[1:, None, :], lerp_steps[None, :, None, None]).view(-1, rendering_zyxs.shape[1], 3)
        rendering_zyxs = torch.lerp(rendering_zyxs[:, :-1, None], rendering_zyxs[:, 1:, None], lerp_steps[None, None, :, None]).view(rendering_zyxs.shape[0], -1, 3)
        normalised_zyxs = (rendering_zyxs - torch.tensor([z_begin, 0, 0], device=rendering_zyxs.device)) / torch.tensor(scroll_slices.shape, device=rendering_zyxs.device) / scroll_slices_downsample_factor * 2. - 1.
        # TODO: maybe do this in blocks -- currently we convert the full scroll_slices to float32
        rendered_scroll_mip_slice = F.grid_sample(scroll_slices[None, None].to(torch.float32), normalised_zyxs[None, None].flip(-1).cpu()).squeeze()
        rendered_scroll = torch.maximum(rendered_scroll, rendered_scroll_mip_slice)
    rendered_scroll = rendered_scroll[:, :, None].expand(-1, -1, 3).clone()

    theta, _ = get_theta(spiral_yxs)  # TODO: could just return these from get_spiral_yxs
    theta = torch.repeat_interleave(theta[:-1], num_lerp_steps)
    theta_colours = kornia.color.hsv_to_rgb(torch.stack([theta[None], *[torch.ones_like(theta[None])] * 2])).squeeze(1).T * 128.
    theta_strip = theta_colours[None].expand(6, -1, 3)
    winding_start_x_coords = torch.where(theta.diff(prepend=theta[:1]) < 0)[0]
    dark_red = torch.tensor([128., 0., 0.], device=rendered_scroll.device)
    rendered_scroll[:, winding_start_x_coords] = dark_red

    # FIXME: theoretically these 'z isolines' may be non-unique in each column, e.g. if the scroll were folded into a U shape, hence argmin is not sufficient
    # FIXME: this should use the rendering_zyxs from the centre slice of the MIP (not the last as currently)
    for grid_z in grid_zs:
        nearest_z_ys = (rendering_zyxs[..., 0] - grid_z).abs().argmin(dim=0)
        rendered_scroll[nearest_z_ys, torch.arange(rendered_scroll.shape[1]), :] = dark_red

    rendered_scroll = torch.cat([rendered_scroll, theta_strip.cpu()], dim=0)
    rendered_image = Image.fromarray(rendered_scroll.to(torch.uint8).numpy())

    draw = ImageDraw.Draw(rendered_image)
    last_zs_printed_x = 0
    winding_start_x_coords = winding_start_x_coords.cpu()
    for x_coord_idx in range(len(winding_start_x_coords)):
        x_coord = winding_start_x_coords[x_coord_idx]
        winding_idx = inner_winding[x_coord // num_lerp_steps].item()
        draw.text(
            xy=(x_coord.item() + 4, 2),
            text=str(winding_idx),
        )
        if last_zs_printed_x == 0 or x_coord - last_zs_printed_x > 768:
            last_zs_printed_x = x_coord
            for grid_z in grid_zs:
                nearest_y = (rendering_zyxs[:, x_coord, 0] - grid_z).abs().argmin()
                draw.text(
                    xy=(x_coord.item() + 2, nearest_y.item()),
                    text=str(grid_z.item()),
                    fill=(128, 0, 0),
                    anchor='lm',
                    font_size=9,
                )
        if x_coord_idx > 0:
            prev_x_coord = winding_start_x_coords[x_coord_idx - 1]
            slice_image = rendered_image.crop((prev_x_coord.item(), 0, x_coord.item(), rendered_image.height))
            prev_winding_idx = inner_winding[prev_x_coord // num_lerp_steps].item()
            slice_image.save(f'{out_path}/rendered_{name}_w{prev_winding_idx:03}.png')

    rendered_image.save(f'{out_path}/rendered_{name}_full.png')


def fit_spiral_3d(scroll_zarr, predictions_zarr, horizontal_fibre_zyxs, vertical_fibre_zyxs, surface_track_zyxs, point_pairs_and_number_differences, points_and_normals, z_begin, z_end, z_to_umbilicus_yx, z_to_gp_min_max_yx, out_path):

    num_slices_for_visualisation = 50
    num_slices_for_gp_eval = 50
    rendering_slices_downsample_factor = 2  # stride the scroll by this along zyx for rendering

    device = torch.device('cuda')

    all_zs = np.arange(z_begin, z_end)
    zs_for_visualisation = np.linspace(z_begin, z_end - 1, min(num_slices_for_visualisation, z_end - 1 - z_begin), dtype=np.int64)
    zs_for_gp_eval = np.linspace(z_begin, z_end - 1, min(num_slices_for_gp_eval, z_end - 1 - z_begin), dtype=np.int64)

    umbilicus_zyx = torch.from_numpy(np.concatenate([all_zs[:, None], z_to_umbilicus_yx(all_zs)], axis=-1).astype(np.float32)).to(device)

    all_zs = torch.from_numpy(all_zs).to(device)

    z_to_gp_equal_radius_zyx_pairs, gp_slices_for_visualisation = load_and_slice_gp_mesh(zs_for_gp_eval, zs_for_visualisation, vis_hw=scroll_zarr.shape[1:])

    horizontal_fibre_lengths = np.asarray([fibre_zyx.shape[0] for fibre_zyx in horizontal_fibre_zyxs]).astype(np.float32)
    vertical_fibre_lengths = np.asarray([fibre_zyx.shape[0] for fibre_zyx in vertical_fibre_zyxs]).astype(np.float32)
    surface_track_lengths = [
        np.asarray([track_zyx.shape[0] for track_zyx in track_zyxs]).astype(np.float32)
        for track_zyxs in surface_track_zyxs
    ]

    predictions_subvolume = torch.from_numpy(predictions_zarr[z_begin : z_end])
    assert predictions_subvolume.dtype == torch.uint8

    if True:  # for bruniss predictions zarr, which have *shape* transpose but not pixels!
        assert predictions_subvolume.shape[1:] == (2024, 1972)
        predictions_subvolume = torch.cat([predictions_subvolume[:, :1972], torch.zeros([predictions_subvolume.shape[0], 1972, 2024 - 1972], dtype=predictions_subvolume.dtype)], dim=2)

    scroll_slices_for_visualisation = (torch.from_numpy(scroll_zarr[zs_for_visualisation]).to(torch.float32) / np.iinfo(scroll_zarr.dtype).max * 0.75 * 255).to(torch.uint8)
    scroll_slices_for_rendering = (torch.from_numpy(scroll_zarr[z_begin : z_end : rendering_slices_downsample_factor, ::rendering_slices_downsample_factor, ::rendering_slices_downsample_factor]).to(torch.int32) // (np.iinfo(scroll_zarr.dtype).max // 255)).to(torch.uint8)

    slice_yx = torch.stack(torch.meshgrid(
        torch.arange(predictions_subvolume.shape[1], dtype=torch.float32),
        torch.arange(predictions_subvolume.shape[2], dtype=torch.float32),
        indexing='ij'
    ), axis=-1).to(device)

    gp_min_max_yx_slice = z_to_gp_min_max_yx(all_zs.cpu()).reshape(all_zs.shape[0], 2, 2)  # slice, min/max, yx
    flow_min_corner_spiral_zyx = torch.tensor([z_begin - cfg['flow_bounds_z_margin'], -cfg['flow_bounds_radius'], -cfg['flow_bounds_radius']], dtype=torch.int64, device=device)
    flow_max_corner_spiral_zyx = torch.tensor([z_end + cfg['flow_bounds_z_margin'], cfg['flow_bounds_radius'], cfg['flow_bounds_radius']], dtype=torch.int64, device=device)

    @torch.inference_mode
    def save_overlay(slice_to_spiral_transform, suffix):

        # TODO: maybe use the smoothed umbilicus here, to avoid weird swirls appearing
        flow_corners_zyx = slice_to_spiral_transform.inv(torch.stack([flow_min_corner_spiral_zyx, flow_max_corner_spiral_zyx], dim=0).to(torch.float32)).to(torch.int64)
        flow_min_corner_zyx = flow_corners_zyx.amin(dim=0)
        flow_max_corner_zyx = flow_corners_zyx.amax(dim=0)

        def draw_boxes(canvas, gp_min_max_yx):
            def draw_box(min_corner_yx, max_corner_yx):
                canvas[min_corner_yx[0] : max_corner_yx[0], min_corner_yx[1]: min_corner_yx[1] + 1] = 150
                canvas[min_corner_yx[0] : max_corner_yx[0], max_corner_yx[1]: max_corner_yx[1] + 1] = 150
                canvas[min_corner_yx[0] : min_corner_yx[0] + 1, min_corner_yx[1]: max_corner_yx[1]] = 150
                canvas[max_corner_yx[0] : max_corner_yx[0] + 1, min_corner_yx[1]: max_corner_yx[1]] = 150
            draw_box(flow_min_corner_zyx[1:], flow_max_corner_zyx[1:])
            draw_box(gp_min_max_yx[0], gp_min_max_yx[1])

        def overlay_on_predictions_or_gp(spiral, slice, mask, gp_min_max_yx, name, cyan):
            spiral_density_vis = (spiral * 255).to(torch.uint8)
            canvas = torch.stack([spiral_density_vis, slice // 2, slice // 2 if cyan else torch.zeros_like(slice)], dim=-1) * mask[..., None]
            draw_boxes(canvas, gp_min_max_yx)
            canvas = canvas.cpu().numpy()
            Image.fromarray(canvas).save(f'{out_path}/spiral_on_{name}_{suffix}.png', compress_level=3)

        def overlay_on_scroll(slice_zyx, spiral_zyx, spiral_density, slice, gp_min_max_yx, name):
            slice_min = slice[slice > 0].amin()
            slice_normalised = (slice - slice_min) / (slice.amax() - slice_min)
            spiral_density_normalised = spiral_density / spiral_density.amax()
            theta, _ = get_theta(spiral_zyx[..., 1:])
            theta_colours = kornia.color.hsv_to_rgb(torch.stack([theta, *[torch.ones_like(theta)] * 2])).permute(1, 2, 0) * 0.5
            spiral_density_coloured = spiral_density_normalised[..., None].expand(-1, -1, 3) * theta_colours
            canvas = slice_normalised[..., None].expand(-1, -1, 3) * (1. - spiral_density_normalised[..., None]) + spiral_density_coloured
            canvas *= (slice > 0)[..., None]
            canvas = (canvas * 255).to(torch.uint8)
            draw_boxes(canvas, gp_min_max_yx)
            canvas = Image.fromarray(canvas.cpu().numpy())
            draw = ImageDraw.Draw(canvas)
            _, yxs_by_radial, winding_indices_by_radial = get_winding_positions_on_radials(
                slice_z=slice_zyx[0, 0, :1],
                thetas=torch.arange(torch.pi / 8, 2 * torch.pi, torch.pi / 4),
                max_radius=slice_zyx[..., 1:].amax(),
                slice_to_spiral_transform=slice_to_spiral_transform,
                dr_per_winding=dr_per_winding,
                z_to_umbilicus_yx=z_to_umbilicus_yx,
            )
            for radial_idx in range(len(yxs_by_radial)):
                for idx in range(winding_indices_by_radial[radial_idx].shape[0]):
                    marker_yx = yxs_by_radial[radial_idx][idx]
                    if (marker_yx > 0).all() and (marker_yx < torch.tensor(slice.shape)).all() and slice[*marker_yx.to(torch.int64)] > 0:
                        winding_idx = int(winding_indices_by_radial[radial_idx][idx].item())
                        if winding_idx > 0 and winding_idx % 5 == 0:
                            draw.point(tuple(marker_yx)[::-1])
                            draw.text(
                                tuple(marker_yx)[::-1],
                                str(winding_idx)
                            )
            canvas.save(f'{out_path}/spiral_on_{name}_{suffix}.png', compress_level=3)

        for vis_slice_idx, slice_z in enumerate(tqdm(zs_for_visualisation, desc='visualising slices')):
            slice_zyx = torch.cat([torch.full([*slice_yx.shape[:2], 1], slice_z, device=device), slice_yx], dim=-1)

            spiral_zyx = slice_to_spiral_transform(slice_zyx)
            spiral_density = spiral_and_transform.get_spiral_density(spiral_zyx)
            slice = scroll_slices_for_visualisation[vis_slice_idx].to(device)
            gp_min_max_yx = z_to_gp_min_max_yx(slice_z).reshape(2, 2).astype(np.int64)
            overlay_on_scroll(slice_zyx, spiral_zyx, spiral_density, slice, gp_min_max_yx, f'scroll_s{slice_z:04}')
            overlay_on_predictions_or_gp(spiral_density, predictions_subvolume[slice_z - z_begin].to(device), slice > 0., gp_min_max_yx, f'pred_s{slice_z:04}', cyan=False)
            overlay_on_predictions_or_gp(spiral_density, gp_slices_for_visualisation[vis_slice_idx].to(device) * 255, slice > 0., gp_min_max_yx, f'gp_s{slice_z:04}', cyan=True)

        def save_section(section_zyx, umbilicus_vs, name):
            section_density = spiral_and_transform.get_spiral_density(slice_to_spiral_transform(section_zyx))
            section_density_vis = (section_density * 255).to(torch.uint8).T.cpu()
            section_predictions = torch.from_numpy(predictions_zarr[*(section_zyx + 0.5).to(torch.int64).cpu().clip(torch.zeros([], dtype=torch.int64), torch.tensor(predictions_zarr.shape) - 1).numpy().T])
            canvas = torch.stack([section_density_vis, section_predictions // 2, torch.zeros_like(section_density_vis)], dim=-1)
            canvas[(umbilicus_vs).to(torch.int64).cpu(), torch.arange(canvas.shape[1])] = 128
            Image.fromarray(canvas.numpy()).save(f'{out_path}/spiral_{name}_{suffix}.png', compress_level=3)

        section_zx = torch.meshgrid(all_zs, torch.arange(flow_min_corner_zyx[2], flow_max_corner_zyx[2], device=all_zs.device), indexing='ij')
        section_zyx = torch.stack([section_zx[0], umbilicus_zyx[:, 1, None].expand(-1, section_zx[0].shape[1]), section_zx[1]], dim=-1)
        save_section(section_zyx, umbilicus_zyx[:, 2] - flow_min_corner_zyx[2], 'zx')
        section_zy = torch.meshgrid(all_zs, torch.arange(flow_min_corner_zyx[1], flow_max_corner_zyx[1], device=all_zs.device), indexing='ij')
        section_zyx = torch.stack([section_zy[0], section_zy[1], umbilicus_zyx[:, 2, None].expand(-1, section_zy[0].shape[1])], dim=-1)
        save_section(section_zyx, umbilicus_zyx[:, 1] - flow_min_corner_zyx[1], 'zy')

        vis_slice_idx = len(scroll_slices_for_visualisation) // 2
        slice_z = zs_for_visualisation[vis_slice_idx]
        slice_zyx = torch.cat([torch.full([*slice_yx.shape[:2], 1], slice_z, device=device), slice_yx], dim=-1)
        mask = scroll_slices_for_visualisation[vis_slice_idx].to(device) > 0.

        tracked_point_zyx_spiral = torch.tensor([
            [slice_z, radius * torch.sin(theta), radius * torch.cos(theta)]
            for theta in torch.arange(0., 2 * torch.pi, torch.pi / 6)
            for radius in torch.arange(25., 400., 40.)
        ]).cuda()
        tracked_point_colours = torch.rand_like(tracked_point_zyx_spiral) * 0.7 + 0.3
        tracked_point_zyx_slice_by_t = []
        for timestep in tqdm(range(spiral_and_transform.flow_timesteps), desc='visualising timesteps'):
            slice_to_spiral_transform_trunc = spiral_and_transform.get_slice_to_spiral_transform(truncate_at_step=timestep)
            spiral_density = spiral_and_transform.get_spiral_density(slice_to_spiral_transform_trunc(slice_zyx))
            spiral_density_vis = (spiral_density * 255).to(torch.uint8)
            predictions_slice = predictions_subvolume[slice_z - z_begin].to(device)
            canvas = torch.stack([spiral_density_vis, predictions_slice // 2, torch.zeros_like(predictions_slice)], dim=-1) * mask[..., None]
            tracked_point_zyx_slice_by_t.append((slice_to_spiral_transform_trunc.inv(tracked_point_zyx_spiral) + 0.5).to(torch.int64))
            for idx, tracked_point_zyx_slice in enumerate(tracked_point_zyx_slice_by_t):
                # Note we just drop the z-coordinate here, i.e. project into the visualised slice!
                # colours = (tracked_point_colours * (128 if idx < len(tracked_point_zyx_slice_by_t) - 1 else 255)).to(torch.uint8)
                colours = (tracked_point_colours * 255 * (idx + 1) / len(tracked_point_zyx_slice_by_t)).to(torch.uint8)
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        canvas[*(tracked_point_zyx_slice[:, 1:] + torch.tensor([dy, dx], device=tracked_point_zyx_slice.device)).T, :] = colours
            Image.fromarray(canvas.cpu().numpy()).save(f'{out_path}/spiral_latest_s{slice_z:04}_seq_t{timestep:03}.png', compress_level=3)

    @torch.inference_mode
    def get_gp_spiral_bounds(dr_per_winding):
        gp_corner_yx_slice = torch.from_numpy(np.array([
            [gp_min_max_yx_slice[:, 0, 0], gp_min_max_yx_slice[:, 0, 1]],
            [gp_min_max_yx_slice[:, 0, 0], gp_min_max_yx_slice[:, 1, 1]],
            [gp_min_max_yx_slice[:, 1, 0], gp_min_max_yx_slice[:, 1, 1]],
            [gp_min_max_yx_slice[:, 1, 0], gp_min_max_yx_slice[:, 0, 1]],
        ], dtype=np.float32)).to(all_zs.device).permute(2, 0, 1)  # slice, corner, yx
        gp_corner_zyx_slice = torch.cat([all_zs[:, None, None].expand(-1, 4, 1).to(torch.float32), gp_corner_yx_slice], dim=-1)
        gp_corner_zyx_spiral = slice_to_spiral_transform(gp_corner_zyx_slice)
        # We use bbox-corners for z range, which is rather conservative
        min_z_spiral = gp_corner_zyx_spiral[..., 0].amin().item()
        max_z_spiral = gp_corner_zyx_spiral[..., 0].amax().item()
        # To find the outermost winding, look along each bbox-edge (in each slice separately), and find
        # the *minimum* winding crossing the edge. This is the point where the spiral is most stretched
        # for that edge. We then find the max of these minima across all bbox-edges and slices
        bbox_edge_yxs_slice = torch.lerp(
            gp_corner_yx_slice[:, :, None, :],
            gp_corner_yx_slice.roll(1, dims=1)[:, :, None, :],
            torch.linspace(0., 1., 20, device=gp_corner_yx_slice.device)[None, None, :, None]
        )  # slice, edge, point-on-edge, yx
        bbox_edge_zyxs_slice = torch.cat([all_zs[:, None, None, None].expand(-1, *bbox_edge_yxs_slice.shape[1:3], 1).to(torch.float32), bbox_edge_yxs_slice], dim=-1)
        bbox_edge_zyxs_spiral = slice_to_spiral_transform(bbox_edge_zyxs_slice)
        _, _, _, outer_winding_idx = get_bounding_windings(bbox_edge_zyxs_spiral[..., 1:], dr_per_winding)
        min_outer_winding_idx_per_edge = outer_winding_idx.amin(dim=2)  # slice, edge
        return min_outer_winding_idx_per_edge.amax().int(), min_z_spiral, max_z_spiral

    def save_current_meshes(suffix):
        dr_per_winding = spiral_and_transform.get_dr_per_winding()
        gp_spiral_bounds = get_gp_spiral_bounds(dr_per_winding)
        save_mesh(slice_to_spiral_transform, dr_per_winding, all_zs, scroll_slices_for_rendering, rendering_slices_downsample_factor, z_begin, zs_for_visualisation[::2], gp_spiral_bounds, out_path, name=suffix, glued=False)

    num_training_steps = cfg['num_training_steps']

    spiral_and_transform = SpiralAndTransform(flow_timesteps=cfg['num_euler_timesteps'], umbilicus_zyx=umbilicus_zyx, flow_min_corner_zyx=flow_min_corner_spiral_zyx, flow_max_corner_zyx=flow_max_corner_spiral_zyx)
    spiral_and_transform.to(device)

    optimiser = torch.optim.Adam(spiral_and_transform.parameters(), lr=cfg.learning_rate)
    if cfg['cosine_lr_schedule']:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda step: max(0.0, 0.5 * (1.0 + np.cos(np.pi * 2.0 * step / num_training_steps))))
    else:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda step: 1.)

    def save_model(suffix):
        torch.save([spiral_and_transform.state_dict(), optimiser.state_dict()], f'{out_path}/checkpoint_{suffix}.ckpt')

    def load_model(path):
        transformed_spiral_state, optimiser_state = torch.load(path, map_location='cpu')
        spiral_and_transform.load_state_dict(transformed_spiral_state)
        optimiser.load_state_dict(optimiser_state)

    if False:
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=5, warmup=2, active=2, repeat=1),
            on_trace_ready=lambda p: p.export_chrome_trace(f'{out_path}/profile.out'),
            record_shapes=True,
            with_stack=True,
        )
        profiler.start()
    else:
        profiler = None

    for iteration in tqdm(range(num_training_steps)):

        slice_to_spiral_transform = spiral_and_transform.get_slice_to_spiral_transform()
        dr_per_winding = spiral_and_transform.get_dr_per_winding()

        if iteration % 1000 == 0:
            save_overlay(slice_to_spiral_transform, f'{iteration:06}')
            maybe_gp_metrics = evaluate_wrt_gp(slice_to_spiral_transform, dr_per_winding, z_to_umbilicus_yx, z_to_gp_equal_radius_zyx_pairs, get_gp_spiral_bounds(dr_per_winding))
        else:
            maybe_gp_metrics = {}

        if iteration % 1000 == 0:
            save_model(f'{iteration:06}')
            save_current_meshes(f'{iteration:06}')

        losses = {}

        horizontal_fibre_radius_loss, horizontal_fibre_dt_loss, fibre_horizontalness_loss = get_fibre_and_track_losses(slice_to_spiral_transform, dr_per_winding, cfg['radius_num_fibres'], horizontal_fibre_zyxs, horizontal_fibre_lengths, direction='h')
        vertical_fibre_radius_loss, vertical_fibre_dt_loss, fibre_verticalness_loss = get_fibre_and_track_losses(slice_to_spiral_transform, dr_per_winding, cfg['radius_num_fibres'], vertical_fibre_zyxs, vertical_fibre_lengths, direction='v')
        total_track_radius_loss = 0.
        total_track_dt_loss = 0.
        for track_zyxs, track_lengths in zip(surface_track_zyxs, surface_track_lengths):
            track_radius_loss, track_dt_loss, _ = get_fibre_and_track_losses(slice_to_spiral_transform, dr_per_winding, cfg['radius_num_tracks'], track_zyxs, track_lengths)
            total_track_radius_loss += track_radius_loss
            total_track_dt_loss += track_dt_loss
        losses['fibre_track_radius'] = (horizontal_fibre_radius_loss + vertical_fibre_radius_loss + total_track_radius_loss / len(surface_track_zyxs)) * cfg['loss_weight_fibre_track_radius']
        losses['fibre_track_dt'] = (horizontal_fibre_dt_loss + vertical_fibre_dt_loss + total_track_dt_loss / len(surface_track_zyxs)) * cfg['loss_weight_fibre_track_dt'] if iteration > cfg['loss_start_fibre_track_dt'] else torch.zeros([])
        losses['fibre_horizontal'] = (fibre_horizontalness_loss + fibre_verticalness_loss) * cfg['loss_weight_fibre_direction']

        # TODO: rename surface_count to winding_number in losses and cfg
        if iteration < cfg['loss_stop_surface_count']:
            losses['surface_count'] = get_winding_number_loss(slice_to_spiral_transform, dr_per_winding, point_pairs_and_number_differences) * cfg['loss_weight_surface_count']
        else:
            losses['surface_count'] = torch.zeros([])

        losses['surface_normal'] = get_stratified_normals_loss(slice_to_spiral_transform, points_and_normals) * cfg['loss_weight_surface_normal']

        losses['stretch'] = get_stretch_regularisation_loss(slice_to_spiral_transform, points_and_normals) * cfg['loss_weight_stretch']

        losses['umbilicus'] = slice_to_spiral_transform(umbilicus_zyx)[..., 1:].abs().mean() * cfg['loss_weight_umbilicus']  # i.e. require umbilicus to map to spiral origin

        loss = sum(losses.values())

        loss.backward()
        optimiser.step()
        optimiser.zero_grad()
        lr_scheduler.step()
        if profiler is not None:
            profiler.step()

        if iteration % 1000 == 0 or iteration < 5000 and iteration % 200 == 0:
            print(f'step {iteration}: loss = {loss.item():.1f}, ' + ', '.join(f'{name} = {value.item():.1f}' for name, value in losses.items()))

        wandb.log({
            'total_loss': loss.item(),
            **{name + '_loss': value for name, value in losses.items()},
            **{'gp_' + name: value for name, value in maybe_gp_metrics.items()},
        })

        if loss.isnan().item():
            print('aborting due to NaN')
            return

    save_overlay(spiral_and_transform.get_slice_to_spiral_transform(), 'fitted')
    save_current_meshes('fitted')
    save_model('fitted')


def main():

    np.random.seed(cfg['random_seed'])
    torch.random.manual_seed(cfg['random_seed'])

    umbilicus = scroll1_umbilicus_z_to_yx(downsample_factor)
    gp_bounds = scroll1_z_to_gt_min_max_yx(downsample_factor)

    print('loading zarrs')
    scroll_zarr_array = zarr.open(scroll_zarr_path, mode='r')
    predictions_zarr_array = zarr.open(predictions_zarr_path, mode='r')

    print('loading pkls')
    with open(horizontal_fibres_pkl_path, 'rb') as fp:
        horizontal_fibre_tracks = pickle.load(fp)
    with open(vertical_fibres_pkl_path, 'rb') as fp:
        vertical_fibre_tracks = pickle.load(fp)
    surface_tracks = []
    for tracks_pkl_path in tracks_pkl_paths:
        with open(tracks_pkl_path, 'rb') as fp:
            surface_tracks.append(pickle.load(fp))
    with open(interwindings_path, 'rb') as fp:
        point_pairs_and_number_differences = pickle.load(fp)
    with open(points_and_normals_path, 'rb') as fp:
        points_and_normals = pickle.load(fp)

    slice_range = 0, 3420

    print('filtering fibres/tracks/normals')
    def filter_tracks(tracks):
        return [track for track in tracks if np.any((track[:, 0] >= slice_range[0]) & (track[:, 0] < slice_range[1]))]
    horizontal_fibre_tracks = filter_tracks(horizontal_fibre_tracks)
    vertical_fibre_tracks = filter_tracks(vertical_fibre_tracks)
    surface_tracks = [filter_tracks(tracks) for tracks in surface_tracks]
    points_and_normals = points_and_normals[(points_and_normals[:, 0, 0] >= slice_range[0]) & (points_and_normals[:, 0, 0] < slice_range[1])]
    point_pairs_and_number_differences = [
        (start_zyx, end_zyx, number_difference)
        for start_zyx, end_zyx, number_difference in point_pairs_and_number_differences
        if np.any((start_zyx[0] >= slice_range[0]) & (start_zyx[0] < slice_range[1]))
        or np.any((end_zyx[0] >= slice_range[0]) & (end_zyx[0] < slice_range[1]))
    ]
    print(f'  found {len(horizontal_fibre_tracks):,} horizontal & {len(vertical_fibre_tracks):,} vertical fibres, {" + ".join([f"{len(tracks):,}" for tracks in surface_tracks])} surface tracks, {len(points_and_normals):,} normals, {len(point_pairs_and_number_differences):,} interwinding pairs')

    out_path = f'../out/postprocessing/{datetime.date.today()}_slice-{slice_range[0]}-{slice_range[1]}'
    if not wandb.run.name.startswith('dummy-'):
        out_path += '_' + wandb.run.name
    os.makedirs(out_path, exist_ok=True)

    fit_spiral_3d(scroll_zarr_array, predictions_zarr_array, horizontal_fibre_tracks, vertical_fibre_tracks, surface_tracks, point_pairs_and_number_differences, points_and_normals, slice_range[0], slice_range[1], umbilicus, gp_bounds, out_path)


if __name__ == '__main__':
    wandb.init(project='scrolls', config=default_config)
    wandb.define_metric('gp_frac_gp_jumping_windings', summary='last')
    wandb.define_metric('gp_mean_radial_winding_distance', summary='last')
    cfg = wandb.config
    main()
