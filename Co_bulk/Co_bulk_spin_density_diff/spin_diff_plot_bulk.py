import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

def read_cube_manual(filepath):
    with open(filepath) as f:
        lines = f.readlines()

    natoms = int(lines[2].split()[0])
    nx, vx = int(lines[3].split()[0]), [float(x) for x in lines[3].split()[1:]]
    ny, vy = int(lines[4].split()[0]), [float(x) for x in lines[4].split()[1:]]
    nz, vz = int(lines[5].split()[0]), [float(x) for x in lines[5].split()[1:]]

    # Voxel lengths in Bohr
    dx = np.linalg.norm(vx)
    dy = np.linalg.norm(vy)
    dz = np.linalg.norm(vz)

    # Volumetric data starts after: 2 comment + 1 (natoms) + 3 (grid) + natoms atom lines
    data_start = 6 + natoms
    raw_data = " ".join(lines[data_start:]).split()
    data = np.array(raw_data, dtype=float).reshape((nx, ny, nz))

    return data, dx, dy, dz

# ---- Inputs/outputs
sd_cube  = "Co_ae_spindensity_sd.cube"
nsd_cube = "Co_ae_spindensity_nsd.cube"
fig_out  = "Density_Difference_SD_minus_nSD.pdf"

# Additional text outputs
vol_txt   = "volume_sd_minus_nsd_xyzv.txt"   # whole 3D: x y z value
slice_txt = "slice_sd_minus_nsd_xyv.txt"     # central slice: x y value
slice_mat = "slice_sd_minus_nsd_matrix.txt"  # central slice as matrix (rows=y, cols=x)

# ---- Load data
print("Loading spin-dependent cube...")
data_sd, dx, dy, dz = read_cube_manual(sd_cube)
print("Loading non-spin-dependent cube...")
data_nsd, _, _, _   = read_cube_manual(nsd_cube)

# ---- Difference and central z-slice
print("Computing spin density difference...")
diff = data_sd - data_nsd
k_center = diff.shape[2] // 2
slice_2d = diff[:, :, k_center]

# ---- Real-space axes (Bohr)
nx, ny, nz = diff.shape
x = np.arange(nx) * dx
y = np.arange(ny) * dy
z = np.arange(nz) * dz

# ---- Save whole 3D volume as x y z value
print(f"Saving full volume to: {vol_txt}")
X3, Y3, Z3 = np.meshgrid(x, y, z, indexing="ij")  # shapes (nx, ny, nz)
vol_out = np.column_stack([X3.ravel(), Y3.ravel(), Z3.ravel(), diff.ravel()])
np.savetxt(
    vol_txt, vol_out, fmt="%.6f %.6f %.6f %.6e",
    header=f"x y z value (Bohr); shape=({nx},{ny},{nz}); order=ijk (C-order ravel)"
)

# ---- Save central z-slice as x y value
print(f"Saving central slice (x y value) to: {slice_txt}")
X2, Y2 = np.meshgrid(x, y, indexing="ij")  # shapes (nx, ny)
slice_out = np.column_stack([X2.ravel(), Y2.ravel(), slice_2d.ravel()])
np.savetxt(
    slice_txt, slice_out, fmt="%.6f %.6f %.6e",
    header=f"x y value (Bohr); slice k={k_center}; shape=({nx},{ny}); order=ij"
)

# ---- Save central z-slice as a matrix (rows=y, cols=x)
print(f"Saving central slice (matrix) to: {slice_mat}")
np.savetxt(
    slice_mat, slice_2d, fmt="%.6e",
    header=f"matrix slice k={k_center}; rows=y(0..{ny-1}), cols=x(0..{nx-1})"
)



print("Generating figure ...")
fig, ax = plt.subplots(figsize=(3.5, 3.5))  # ~9 cm single-column width
cmap = ax.pcolormesh(
    x, y, slice_2d.T, cmap="seismic",
    shading="auto",
    vmin= -0.1, #np.max(np.abs(slice_2d)),
    vmax= 0.1 # np.max(np.abs(slice_2d))
)
cbar = plt.colorbar(cmap, ax=ax, pad=0.02, fraction=0.046)
cbar.set_label(r"Spin Density Difference ($e$/bohr$^3$)", fontsize=10)
cbar.ax.tick_params(labelsize=9)

ax.set_xlabel("x (bohr)", fontsize=10)
ax.set_ylabel("y (bohr)", fontsize=10)
ax.set_aspect("equal")
ax.tick_params(labelsize=9)

plt.tight_layout()
plt.savefig(fig_out, dpi=1200)
plt.close()
print(f"Saved figure to: {fig_out}")





