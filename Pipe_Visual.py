# %%
import matplotlib.pyplot as plt
import numpy as np

N_theta = 0
prefix = ""
X_data = np.load(prefix + "Pipe_X.npy")
Y_data = np.load(prefix + "Pipe_Y.npy")
Q_data = np.load(prefix + "Pipe_Q.npy")[:, 0]
res_data = np.load(prefix + "Pipe_res.npy")

print(X_data.shape, Q_data.shape)

# %%
import torch

X = torch.tensor(X_data)
print(torch.sum(torch.abs(X[1:] - X[:-1])))
# %%
print(X_data[0, 1, :])

# %%
X_new = np.zeros((1200, 129, 257))
Y_new = np.zeros((1200, 129, 257))
Q_new = np.zeros((1200, 129, 257))

for i in range(1200):
    for x in range(129):
        X_new[i, x, :] = X_data[i, x, 0]

        Y_new[i, x, :64] = np.linspace(-2, Y_data[i, x, 0], 65)[:-1]
        Y_new[i, x, -64:] = np.linspace(Y_data[i, x, -1], 2, 65)[1:]
        Y_new[i, x, 64 : 257 - 64] = Y_data[i, x, :]

        Q_new[i, x, 64 : 257 - 64] = Q_data[i, x, :]
# %%
x = 1
plt.plot(Y_data[i, x, :])
plt.plot(Y_new[i, x, :])
# %%
fig, ax = plt.subplots(1, figsize=(12, 4), dpi=80)
r = 8
i = 2
ax.pcolormesh(
    X_new[i, 0::r, 0::r],
    Y_new[i, 0::r, 0::r],
    np.zeros(X_new[i, 0::r, 0::r].shape),
    facecolor="none",
    edgecolor="black",
)
ax.axis("equal")
fig.tight_layout()

# %%
x = np.linspace(0, 10, 129)
y = np.linspace(-2, 2, 129)
# full coorindate arrays
xx, yy = np.meshgrid(x, y, indexing="ij")

print(yy[:, 0])
# %%
from scipy import interpolate
from scipy.interpolate import griddata

Q_interp = np.zeros((1200, 129, 129))
points = np.stack((X_new, Y_new), axis=-1)
print(points.shape)


for i in range(1200):
    print(i)
    Q_interp[i] = griddata(
        points[i].reshape(-1, 2),
        Q_new[i].reshape(
            -1,
        ),
        (xx, yy),
        method="cubic",
    )

np.save(prefix + "Pipe_Q_interp_x.npy", Q_interp)
# %%
ind = 99

fig, ax = plt.subplots(3, 1, figsize=(12, 9), dpi=80)
im0 = ax[0].pcolormesh(xx, yy, Q_interp[ind], shading="gouraud")
fig.colorbar(im0, ax=ax[0])
ax[0].axis("equal")

im1 = ax[1].pcolormesh(X_data[ind], Y_data[ind], Q_data[ind], shading="gouraud")
fig.colorbar(im1, ax=ax[1])
ax[1].axis("equal")
# %%
plt.figure(figsize=(16, 8))
plt.plot(res_data)

# %%
ind = 10
X, Y = X_data[ind, :, :], Y_data[ind, :, :]
ux, uy, p = Q_data[ind, 0, :, :], Q_data[ind, 1, :, :], Q_data[ind, 2, :, :]

fig, ax = plt.subplots(3, 1, figsize=(12, 9), dpi=80)
im0 = ax[0].pcolormesh(X, Y, ux, shading="gouraud")
fig.colorbar(im0, ax=ax[0])
ax[0].set_title("u")
ax[0].axis("equal")


im1 = ax[1].pcolormesh(X, Y, uy, shading="gouraud")
fig.colorbar(im1, ax=ax[1])
ax[1].set_title("v")
ax[1].axis("equal")


im2 = ax[2].pcolormesh(X[0::2, 0::2], Y[0::2, 0::2], p[0::2, 0::2], shading="gouraud")
fig.colorbar(im2, ax=ax[2])
ax[2].set_title("p")
ax[2].axis("equal")

fig.tight_layout()
fig.show()
fig.savefig("Pipe.pdf")

# %%
fig, ax = plt.subplots(1, figsize=(12, 4), dpi=80)
xnskip = 2
ynskip = 8
ax.pcolormesh(
    X[0::xnskip, 0::ynskip],
    Y[0::xnskip, 0::ynskip],
    np.zeros(X[0::xnskip, 0::ynskip].shape),
    facecolor="none",
    edgecolor="black",
)
ax.axis("equal")
fig.tight_layout()
fig.savefig("Mesh.pdf")
