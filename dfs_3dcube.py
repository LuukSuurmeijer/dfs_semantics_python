import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import distance_matrix
from sklearn.manifold import MDS
from mpl_toolkits.mplot3d import Axes3D
import sys

m = pd.read_csv("compositional_cube.vectors", header=None, sep=" ")
print(m)
m = m.to_numpy()

# multidimensional scaling (manual)
#dm = distance_matrix(m[:,1:], m[:,1:])
#mds = MDS(n_components=3, dissimilarity="precomputed")
#em = mds.fit_transform(dm)

# multidimensional scaling (automatic)
mds = MDS(n_components=3, random_state=41020) # fixed seed
em = mds.fit_transform(m[:,1:])

# plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# plot fixed points
for i in range(10):
    ax.scatter(
            em[i,0],                                # x
            em[i,1],                                # y
            em[i,2],                                # z
            linewidths=2,
            color="grey")
    ax.text(em[i,0]+.3,                                # x
            em[i,1],                                # y
            em[i,2]+.07,                                 # z + offset
            '%s' % m[i,0],
            ha="center",
            color="grey",
            fontsize=8)
    ax.plot([em[i,0],em[i,0]],                      # (x, x)
            [em[i,1],em[i,1]],                      # (y, y)
            [np.min(em[:,2]),em[i,2]],              # (z, z)
            linewidth=.5,
            color="grey")

# plot comprehension trajectory
for i in range(10,len(m)):
    # mike entered bar ordered
    c = "#1f77b4"
    # ... cola
    if (i == 14):
        c = "#9467bd"
    # ... fries
    if (i == 15):
        c = "#ff7f0e"
    ax.scatter(
            em[i,0],                                # x
            em[i,1],                                # y
            em[i,2],                                # z
            linewidths=2,
            color=c)
    ax.text(em[i,0]+.3,                                # x
            em[i,1],                          # y
            em[i,2]-.1,                                # z + offset
            '%s' % m[i,0],
            ha="left",
            color=c,
            fontsize=10)
    ax.plot([em[i,0],em[i,0]],                      # (x, x)
            [em[i,1],em[i,1]],                      # (y, y)
            [np.min(em[:,2]),em[i,2]],              # (z, z)
            linewidth=.5,
            color=c)
    # connect the dots ...
    if (i > 10 and i < 15):
        ax.plot([em[i-1,0],em[i,0]],                # (px, x)
                [em[i-1,1],em[i,1]],                # (py, y)
                [em[i-1,2],em[i,2]],                # (pz, z)
                linewidth=2,
                color=c)
    if (i == 15):
        ax.plot([em[i-2,0],em[i,0]],                # (ppx, x)
                [em[i-2,1],em[i,1]],                # (ppy, y)
                [em[i-2,2],em[i,2]],                # (ppz, z)
                linewidth=2,
                color=c)
    # surprisal values ...
    if (i == 14):
        ax.text((em[i-1,0] + em[i,0]) / 2 + 0.0,   # x + offset
                (em[i-1,1] + em[i,1]) / 2 + 0.7,   # y + offset
                (em[i-1,2] + em[i,2]) / 2 + 0.0,   # z + offset
                '0.80',
                ha="center",
                color=c,
                fontsize=13)
    if (i == 15):
        ax.text((em[i-2,0] + em[i,0]) / 2 + 0.0,   # x + offset
                (em[i-2,1] + em[i,1]) / 2 + 0.6,   # y + offset
                (em[i-2,2] + em[i,2]) / 2 - 0.1,   # z + offset
                '0.96',
                ha="center",
                color=c,
                fontsize=13)

# add title
#ax.set_title("Meaning Space Navigation", va="top", x=.5, y=1.05)

# remove axis labels
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

# rotation
ax.view_init(15,-140)

# reduce whitespace around cube
ax.dist = 8

fig.set_size_inches(10, 10)
fig.savefig("plots/cube_comp.pdf", bbox_inches="tight")
fig.show()
plt.show()
