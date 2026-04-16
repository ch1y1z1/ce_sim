from typing import List

import numpy as np


def _linear_pos(ny: int, region_distance: int, npml: int, num: int) -> List[int]:
    center_list = np.linspace(
        region_distance + npml,
        ny - region_distance - npml,
        num + 2,
    )[1:-1]
    return np.round(center_list).astype(np.int64).tolist()


def init_layer(grid, input_cfg, output_cfg):
    nx, ny = grid["nx"], grid["ny"]
    region_distance = grid["region_distance"]
    npml = grid["npml"]
    wg_width = grid["wg_width"]
    src_pos = grid["src_pos"]
    src_margin = grid["source_margin"]
    n_bits_i = input_cfg["n_bits"]
    n_bits_o = output_cfg["n_bits"]

    bg_x = np.zeros((nx, ny))
    region = np.zeros((nx, ny))

    probe_list = []
    src_list = []

    bit_centers_i = _linear_pos(ny, region_distance, npml, n_bits_i)
    for wg_center in bit_centers_i:
        bg_x[
            0 : npml + region_distance,
            wg_center - wg_width // 2 : wg_center + wg_width // 2,
        ] = 1
        src_slice = {
            "x": np.array(npml + src_pos),
            "y": np.arange(
                wg_center - wg_width // 2 - src_margin,
                wg_center + wg_width // 2 + src_margin,
            ),
        }
        src_list.append(src_slice)

    bit_centers_o = _linear_pos(ny, region_distance, npml, n_bits_o)
    for wg_center in bit_centers_o:
        bg_x[
            nx - npml - region_distance :,
            wg_center - wg_width // 2 : wg_center + wg_width // 2,
        ] = 1
        probe_slice = {
            "x": np.array(nx - npml - src_pos),
            "y": np.arange(
                wg_center - wg_width // 2 - src_margin,
                wg_center + wg_width // 2 + src_margin,
            ),
        }
        probe_list.append(probe_slice)

    region[
        npml + region_distance : nx - npml - region_distance,
        npml + region_distance : ny - npml - region_distance,
    ] = 1

    x = region * 0.5
    return x, bg_x, region, src_list, probe_list


def epsr_parameterization(x, bg_x, region, ep_min, ep_max):
    ep_opt = ep_min + (ep_max - ep_min) * x
    ep_outer = 1.0 + (ep_max - 1.0) * bg_x
    ep = ep_opt * region + ep_outer * (region == 0).astype(float)
    return ep


def mask_combine_rho(rho, bg_rho, design_region):
    return rho * design_region + bg_rho * (design_region == 0).astype(np.float32)
