# shadow.py -- correlated shadow field generator and sampler
import numpy as np
import math
from typing import Tuple

def generate_shadow_field(minx, maxx, miny, maxy,
                          grid_res_m=50.0, sigma_m=50.0, sigma_db=7.0):
    width = maxx - minx; height = maxy - miny
    nx_grid = max(4, int(math.ceil(width / grid_res_m)) + 1)
    ny_grid = max(4, int(math.ceil(height / grid_res_m)) + 1)
    noise = np.random.normal(loc=0.0, scale=sigma_db, size=(ny_grid, nx_grid))
    sigma_grid = max(0.5, sigma_m / grid_res_m)
    kx = int(max(3, math.ceil(sigma_grid * 6))); ky = int(max(3, math.ceil(sigma_grid * 6)))
    x = np.arange(-kx, kx+1); y = np.arange(-ky, ky+1)
    xx, yy = np.meshgrid(x, y)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma_grid**2))
    kernel /= kernel.sum()
    out_shape = (ny_grid + kernel.shape[0], nx_grid + kernel.shape[1])
    def next_pow2(x): return 1 << (int(x-1).bit_length())
    fshape = (next_pow2(out_shape[0]), next_pow2(out_shape[1]))
    f_noise = np.fft.rfftn(noise, s=fshape)
    pad_kernel = np.zeros(fshape); pad_kernel[:kernel.shape[0], :kernel.shape[1]] = kernel
    f_kernel = np.fft.rfftn(pad_kernel, s=fshape)
    f_filtered = f_noise * f_kernel
    filtered = np.fft.irfftn(f_filtered, s=fshape)
    filtered = filtered[:ny_grid, :nx_grid]
    filtered = filtered - filtered.mean()
    if filtered.std() > 0:
        filtered = filtered * (sigma_db / filtered.std())
    gx = np.linspace(minx, maxx, nx_grid)
    gy = np.linspace(miny, maxy, ny_grid)
    return filtered, gx, gy

def shadow_at_pos_from_field(field, gx, gy, clip_db=12.0):
    def f(x: float, y: float) -> float:
        if x < gx[0]: x = gx[0]
        if x > gx[-1]: x = gx[-1]
        if y < gy[0]: y = gy[0]
        if y > gy[-1]: y = gy[-1]
        fx = (x - gx[0]) / (gx[-1] - gx[0]) * (len(gx)-1)
        fy = (y - gy[0]) / (gy[-1] - gy[0]) * (len(gy)-1)
        ix = int(math.floor(fx)); iy = int(math.floor(fy))
        ix1 = min(ix+1, len(gx)-1); iy1 = min(iy+1, len(gy)-1)
        wx = fx - ix; wy = fy - iy
        v00 = field[iy, ix]; v10 = field[iy, ix1]
        v01 = field[iy1, ix]; v11 = field[iy1, ix1]
        v = (1-wx)*(1-wy)*v00 + wx*(1-wy)*v10 + (1-wx)*wy*v01 + wx*wy*v11
        return max(-clip_db, min(clip_db, float(v)))
    return f
