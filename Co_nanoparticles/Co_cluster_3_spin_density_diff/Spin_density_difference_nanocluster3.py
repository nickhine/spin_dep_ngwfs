#!/usr/bin/env python3
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

def read_cube(cube_path):
    """
    Read a Gaussian cube file and return:
      - data: NumPy array (nx, ny, nz)
      - extent: [xmin', xmax', ymin', ymax'] with half-voxel shift
      - atoms: list of (x,y) positions from header
    """
    with open(cube_path, 'r') as f:
        f.readline(); f.readline()
        parts = f.readline().split()
        natoms = int(parts[0])
        origin = np.array(parts[1:4], dtype=float)
        # grid specs
        nx_line = f.readline().split(); nx, vx = int(nx_line[0]), np.array(nx_line[1:4], dtype=float)
        ny_line = f.readline().split(); ny, vy = int(ny_line[0]), np.array(ny_line[1:4], dtype=float)
        nz_line = f.readline().split(); nz, vz = int(nz_line[0]), np.array(nz_line[1:4], dtype=float)
        # atom coords
        atoms = []
        for _ in range(natoms):
            a = f.readline().split()
            atoms.append((float(a[2]), float(a[3])))
        # volumetric data
        vals = np.fromstring(f.read(), sep=' ')
    if vals.size != nx*ny*nz:
        raise ValueError(f"Expected {nx*ny*nz} values, got {vals.size}")
    data = vals.reshape((nx, ny, nz), order='F')


    dx = np.linalg.norm(vx)
    dy = np.linalg.norm(vy)

    xmin, ymin = origin[0], origin[1]
    extent = [xmin - dx/2,
              xmin + nx*dx - dx/2,
              ymin - dy/2,
              ymin + ny*dy - dy/2]
    return data, extent, atoms

def plot_difference(data1, data2, extent, atoms, outname, vmax=None):
    diff2d = np.sum(data1 - data2, axis=2)
    if vmax is None:
        vmax = np.max(np.abs(diff2d))
    masked = np.ma.masked_inside(diff2d, -1e-4, 1e-4)

    txt_out = os.path.splitext(outname)[0] + ".txt"
    np.savetxt(txt_out, diff2d, fmt="%.6e")
    print(f"Saved 2D data matrix: {txt_out}")

    cmap = cm.get_cmap('seismic').copy()
    cmap.set_bad('white')

    plt.figure(figsize=(6,5), dpi=1000, facecolor='white')
    # <-- DROP THE .T HERE!
    im = plt.imshow(
        masked,           # no .T
        origin='lower',
        extent=extent,
        cmap=cmap,
        vmin=-vmax, vmax=vmax,
        interpolation='none'
    )

    cbar = plt.colorbar(im, pad=0.02)
    cbar.set_label(r"Spin Density Difference (e/bohr$^2$)", fontsize=15)
    cbar.ax.tick_params(labelsize=15)  

    xs, ys = zip(*atoms)
    plt.scatter(xs, ys, s=15, c='k', edgecolors='white', linewidth=0.6, zorder=5)

    plt.xlabel("x (bohr)", fontsize=15)
    plt.ylabel("y (bohr)", fontsize=15)
    plt.tick_params(labelsize=15)
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    os.makedirs(os.path.dirname(outname) or '.', exist_ok=True)
    plt.savefig(outname, dpi=1000, facecolor='white')
    plt.close()
    print(f"Saved: {outname}")

if __name__ == "__main__":
    cube_T_N_13 = "Co_cluster3_sp_T_N_13_ae_spindensity.cube"
    cube_F_N_13 = "Co_cluster3_sp_F_N_13_ae_spindensity.cube"
    cube_T_N_10 = "Co_cluster3_sp_T_N_10_ae_spindensity.cube"
    cube_F_N_10 = "Co_cluster3_sp_F_N_10_ae_spindensity.cube"

    out_dir = "figures"
    os.makedirs(out_dir, exist_ok=True)
    # N_13
    data_T_N_13, extent_N_13, atoms_N_13 = read_cube(cube_T_N_13)
    data_F_N_13, _, _                = read_cube(cube_F_N_13)
    # N_10
    data_T_N_10, extent_N_10, atoms_N_10 = read_cube(cube_T_N_10)
    data_F_N_10, _, _                = read_cube(cube_F_N_10)


    proj_N_13   = np.sum(data_T_N_13 - data_F_N_13, axis=2)
    proj_N_10_TF = np.sum(data_T_N_10 - data_F_N_10, axis=2)
    vmax1 = max(np.max(np.abs(proj_N_13)), np.max(np.abs(proj_N_10_TF)))

    print('ploting')

    plot_difference(data_T_N_13, data_F_N_13, extent_N_13, atoms_N_13,
                    os.path.join(out_dir, "N_13_T_minus_F_atoms.pdf"), vmax1)
    
    plot_difference(data_T_N_10, data_F_N_10, extent_N_10, atoms_N_10,
                    os.path.join(out_dir, "N_10_T_minus_F_atoms.pdf"), vmax1)

