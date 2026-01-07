import numpy as np
import matplotlib.pyplot as plt


def ink_ring_organic(
    seed=0,
    radius=1.0,
    linearity=0.92,     # 0..1 (più alto = più regolare)
    ink_gap=0.08,       # 0..1 (più alto = più aperto)
    streaks=0.35,       # 0..1 (tracce articolate)
    complexity=0.12,    # 0..1 (spray extra / micro-roughness)
    figsize=(6, 6),
    show=True,
    save=None,
    bg="white",
):
    rng = np.random.default_rng(seed)

    lin = float(np.clip(linearity, 0.0, 1.0))
    gap = float(np.clip(ink_gap, 0.0, 1.0))
    st = float(np.clip(streaks, 0.0, 1.0))
    c = float(np.clip(complexity, 0.0, 1.0))

    n = 2600
    theta = np.linspace(0, 2*np.pi, n, endpoint=True)

    # --- smoother ring (controlled by linearity) ---
    rough_amp = (0.09 * (1 - lin)) + (0.012 * c)
    noise = np.zeros_like(theta)
    for k in range(1, 5):
        noise += (1.0 / k) * np.sin(k * theta + rng.uniform(0, 2*np.pi))
    noise /= np.max(np.abs(noise)) + 1e-9

    r = radius * (1 + rough_amp * noise)
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    # --- gaps ---
    gap_count = int(1 + 9 * gap)
    width_lo = 0.02 + 0.05 * gap
    width_hi = 0.05 + 0.50 * gap

    mask = np.ones_like(theta, dtype=bool)
    for _ in range(gap_count):
        center = rng.uniform(0, 2*np.pi)
        width = rng.uniform(width_lo, width_hi)
        d = np.angle(np.exp(1j*(theta - center)))
        mask &= (np.abs(d) > width)

    # split segments
    segments = []
    idx = np.where(mask)[0]
    if idx.size > 0:
        splits = np.where(np.diff(idx) > 1)[0]
        start = 0
        for s in splits:
            segments.append(idx[start:s+1])
            start = s+1
        segments.append(idx[start:])

    # --- plot setup ---
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.set_aspect("equal")
    ax.axis("off")

    # ring stroke
    base_lw = 7 + 4 * (1 - lin)
    var_lw = 0.20 + 0.90 * (1 - lin)

    for seg in segments:
        t = np.linspace(0, 1, seg.size)
        lw = base_lw * (1 - var_lw + var_lw * (0.70 + 0.30*np.sin(2*np.pi*(t + rng.random()))))
        for i in range(seg.size - 1):
            ax.plot(
                x[seg[i]:seg[i]+2], y[seg[i]:seg[i]+2],
                color="black", lw=float(lw[i]),
                solid_capstyle="round", alpha=0.98
            )

    # --------- articulated ink streaks (NEW) ----------
    # Each streak = short random-walk that starts at the ring and moves outward/inward
    n_streaks = int(1 + 22 * (st ** 1.1))

    for _ in range(n_streaks):
        th0 = rng.uniform(0, 2*np.pi)

        # starting point near ring
        rr0 = radius + rng.normal(0.01, 0.03)
        px = rr0 * np.cos(th0)
        py = rr0 * np.sin(th0)

        # initial direction: mostly outward, with slight tangential component
        outward = 1 if rng.random() < 0.8 else -1
        dir_ang = th0 + outward * rng.normal(0.0, 0.25) + rng.normal(0.0, 0.18)

        steps = int(14 + 80 * st)
        step = (0.010 + 0.030 * st) * radius

        xs, ys = [px], [py]
        ang = dir_ang

        # random walk with drift + occasional branching spray
        for j in range(steps):
            # drift keeps it roughly radial, noise makes it organic
            ang += rng.normal(0, 0.20 + 0.35 * st)
            drift = 0.05 * outward * (th0 - ang)  # gently pull toward radial
            ang += drift

            xs.append(xs[-1] + step * np.cos(ang) + rng.normal(0, step * 0.25))
            ys.append(ys[-1] + step * np.sin(ang) + rng.normal(0, step * 0.25))

            # tiny side spray sometimes
            if rng.random() < (0.06 + 0.10 * st):
                spr = int(5 + 30 * st)
                s_ang = ang + rng.normal(0, 0.9)
                sx = xs[-1] + rng.normal(0, step * 0.8, size=spr)
                sy = ys[-1] + rng.normal(0, step * 0.8, size=spr)
                ss = rng.gamma(1.2, 10 + 25 * st, size=spr)
                ax.scatter(sx, sy, s=ss, c="black", alpha=0.06 + 0.10 * st, linewidths=0)

        # variable thickness along the streak
        t = np.linspace(0, 1, len(xs))
        lw = (0.7 + 3.2 * st) * (0.45 + 0.75 * (1 - t)) * (0.7 + 0.6 * rng.random())
        # draw as tiny segments for varying width
        for i in range(len(xs) - 1):
            ax.plot(xs[i:i+2], ys[i:i+2], color="black",
                    lw=float(lw[i]), alpha=0.85, solid_capstyle="round")

    # optional micro-spray
    n_spray = int(120 * c * (1 - lin))
    if n_spray > 0:
        ths = rng.uniform(0, 2*np.pi, size=n_spray)
        rr = radius + rng.normal(0.12, 0.10, size=n_spray)
        sizes = rng.gamma(1.1, 8, size=n_spray)
        ax.scatter(rr*np.cos(ths), rr*np.sin(ths), s=sizes, c="black", alpha=0.08, linewidths=0)

    pad = 1.55
    ax.set_xlim(-pad*radius, pad*radius)
    ax.set_ylim(-pad*radius, pad*radius)

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight", pad_inches=0, facecolor=bg)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


# Preset: anello pulito + tracce articolate moderate
ink_ring_organic(seed=4, linearity=0.95, ink_gap=0.06, streaks=0.4, complexity=0.99)
