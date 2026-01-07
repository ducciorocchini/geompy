# Geometric fern (Barnsley IFS) with optional "outline" look
# Requires: numpy, matplotlib

import numpy as np
import matplotlib.pyplot as plt

def barnsley_fern(n=200_000, seed=0):
    rng = np.random.default_rng(seed)
    x, y = 0.0, 0.0
    xs = np.empty(n)
    ys = np.empty(n)

    for i in range(n):
        r = rng.random()
        if r < 0.01:
            # stem
            x, y = 0.0, 0.16 * y
        elif r < 0.86:
            # successively smaller leaflets
            x, y = 0.85 * x + 0.04 * y, -0.04 * x + 0.85 * y + 1.6
        elif r < 0.93:
            # left leaflet
            x, y = 0.20 * x - 0.26 * y, 0.23 * x + 0.22 * y + 1.6
        else:
            # right leaflet
            x, y = -0.15 * x + 0.28 * y, 0.26 * x + 0.24 * y + 0.44

        xs[i] = x
        ys[i] = y

    return xs, ys

# --- generate points ---
x, y = barnsley_fern(n=180_000, seed=42)

# --- plot ---
plt.figure(figsize=(6, 10))
plt.scatter(x, y, s=0.2, marker='.', linewidths=0)  # geometric point-cloud fern
plt.axis('equal')
plt.axis('off')
plt.tight_layout()
plt.show()

# Save (optional):
# plt.figure(figsize=(6, 10))
# plt.scatter(x, y, s=0.2, marker='.', linewidths=0)
# plt.axis('equal'); plt.axis('off'); plt.tight_layout()
# plt.savefig("geometric_fern.png", dpi=300, bbox_inches="tight", pad_inches=0)
