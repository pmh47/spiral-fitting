import torch
import numpy as np
import scipy.interpolate


def read_thaumato_umbilicus(path):
    with open(path, 'rt') as fp:
        lines = fp.readlines()
    xyzs = np.asarray([
        tuple(map(int, line.strip().replace(' ', '').split(',')))
        for line in lines
    ])[:, [2, 0, 1]] - 500
    return scipy.interpolate.interp1d(xyzs[:, 2], xyzs[:, :2], axis=0, fill_value='extrapolate')  # z -> xy


def scroll1_umbilicus_z_to_yx(downsample_factor):
    from scroll1_umbilicus import umbilicus_zyx
    umbilicus_zyx = np.asarray(umbilicus_zyx).astype(np.float32)
    umbilicus_zyx /= downsample_factor
    return scipy.interpolate.interp1d(umbilicus_zyx[:, 0], umbilicus_zyx[:, 1:], axis=0, fill_value='extrapolate')  # z -> yx


def scroll1_z_to_gt_min_max_yx(downsample_factor):
    from scroll1_gp_bounds import bounds
    bounds = np.asarray(bounds).astype(np.float32)
    bounds /= downsample_factor / 4
    return scipy.interpolate.interp1d(bounds[:, 0], bounds[:, 1:], axis=0, fill_value='extrapolate')  # z -> min/max * yx


def interp1d(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor, dim: int=-1, extrapolate: str='const') -> torch.Tensor:
    # See https://github.com/pytorch/pytorch/issues/50334
    m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
    b = fp[:-1] - (m * xp[:-1])
    # indices = torch.sum(x[None, ...] >= xp.view(-1, *[1] * x.ndim), dim=0) - 1
    indices = torch.searchsorted(xp.squeeze(-1), x) - 1
    indices = torch.clamp(indices, 0, len(m) - 1)
    return m[indices] * x[..., None] + b[indices]


def pairwise_line_segment_intersections(first_yx_pairs, second_yx_pairs, return_yxs=False):

    # Based on https://stackoverflow.com/a/565282
    # first_yx_pairs :: first-line, start/end, yx
    # second_yx_pairs :: second-line, start/end, yx

    p = first_yx_pairs[:, None, 0]  # first-line, 1, yx
    r = first_yx_pairs[:, None, 1] - first_yx_pairs[:, None, 0]
    q = second_yx_pairs[None, :, 0]  # 1, second-line, yx
    s = second_yx_pairs[None, :, 1] - second_yx_pairs[None, :, 0]

    def cross(v, w):
        return v[..., 1] * w[..., 0] - v[..., 0] * w[..., 1]

    r_cross_s = cross(r, s)  # first-line, second-line
    t = cross(q - p, s) / (r_cross_s + 1.e-5)
    u = cross(q - p, r) / (r_cross_s + 1.e-5)

    parallel = r_cross_s == 0
    intersecting = ~parallel & (0 <= t) & (t <= 1) & (0 <= u) & (u <= 1)

    if not return_yxs:
        return intersecting

    intersection_yxs = p + t[..., None] * r  # or q + u[..., None] * s
    return intersecting, intersection_yxs
