from typing import List
import toml
import numpy as np


def linear_pos(ny: int, region_distance: int, num: int) -> List[int]:
    center_list = np.linspace(region_distance, ny - region_distance, num + 2)[1:-1]
    return np.round(center_list).astype(np.int64).tolist()


def init_layer(grid, input, output):
    nx, ny = grid["nx"], grid["ny"]
    region_distance = grid["region_distance"]
    npml = grid["npml"]
    wg_width = grid["wg_width"]
    src_pos = grid["src_pos"]
    src_margin = grid["source_margin"]
    n_bits_i = input["n_bits"]
    n_bits_o = output["n_bits"]
    bg_x = np.zeros((nx, ny))
    region = np.zeros((nx, ny))

    probe_list = []
    src_list = []

    # input bit waveguide
    bit_centers_i = linear_pos(ny, region_distance, n_bits_i)
    for idx in range(n_bits_i):
        wg_center = bit_centers_i[idx]
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

    # output bit waveguide
    bit_centers_o = linear_pos(ny, region_distance, n_bits_o)
    for idx in range(n_bits_o):
        wg_center = bit_centers_o[idx]
        bg_x[
            nx - npml - region_distance :,
            wg_center - wg_width // 2 : wg_center + wg_width // 2,
        ] = 1
        probe_slice = {
            "x": np.array(nx - npml - src_pos),
            "y": np.arange(wg_center - wg_width // 2, wg_center + wg_width // 2),
        }
        probe_list.append(probe_slice)

    # region
    region[
        npml + region_distance : nx - npml - region_distance,
        npml + region_distance : ny - npml - region_distance,
    ] = 1
    # Const init
    x = region * 0.5  # 0.5

    return x, bg_x, region, src_list, probe_list


def init_domain(config: str):
    """
    Input:
        config: str, the path to the configuration file
            or raw string of the configuration
    """
    config = toml.load(config)
    # print(config["layers"])

    layer_list = []
    for layer in config["layers"]:
        l = init_layer(config["grid"], layer["input"], layer["output"])
        layer_list.append(l)

    return layer_list


def epsr_parameterization(x, bg_x, region, ep_min, ep_max):
    """
    Defines the parameterization steps for constructing rho
    """

    # Final masking undoes the blurring of the waveguides
    x = x * region + bg_x * (region == 0).astype(float)

    return ep_min + (ep_max - ep_min) * x


if __name__ == "__main__":
    config = "./Configuration/chi_config.toml"
    init_domain(config)
