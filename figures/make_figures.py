"""Figures of "Hopf Invariants of Polynomial Maps and Villarceau Spheres".
Every curve is computed from the formulas proved in the text.
Run:  python3 make_figures.py   (writes PNG files next to this script)
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import hsv_to_rgb

OUT = os.path.dirname(os.path.abspath(__file__))
DPI = 200
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                     "font.size": 10})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=DPI, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def stereo(z, w):
    """Stereographic projection of S^3 from (0, i) to R^3."""
    d = 1 - w.imag
    return z.real / d, z.imag / d, w.real / d


def hopf_fibre(a, alpha0, n=400):
    """Fibre through (a e^{i alpha0}, b): {(a e^{i(alpha0+t)}, b e^{it})}."""
    b = math.sqrt(1 - a * a)
    t = np.linspace(0, 2 * math.pi, n)
    return stereo(a * np.exp(1j * (alpha0 + t)), b * np.exp(1j * t))


def anti_fibre(a, alpha0, n=400):
    b = math.sqrt(1 - a * a)
    t = np.linspace(0, 2 * math.pi, n)
    return stereo(a * np.exp(1j * (alpha0 - t)), b * np.exp(1j * t))


def torus_surface(a, nu=80, nv=40):
    b = math.sqrt(1 - a * a)
    al, be = np.meshgrid(np.linspace(0, 2 * math.pi, nu),
                         np.linspace(0, 2 * math.pi, nv))
    return stereo(a * np.exp(1j * al), b * np.exp(1j * be))


def clean3d(ax, lim):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()


# ------------------------------------------------ 1. nested tori and fibres
def fig_hopf_tori():
    fig = plt.figure(figsize=(7.5, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    cols = {0.55: "#c0392b", 0.75: "#2471a3", 0.92: "#1e8449"}
    for a, col in cols.items():
        X, Y, Z = torus_surface(a)
        ax.plot_surface(X, Y, Z, color=col, alpha=0.06, linewidth=0,
                        shade=False)
        for k in range(10 if a < 0.9 else 8):
            x, y, z = hopf_fibre(a, 2 * math.pi * k / (10 if a < 0.9 else 8))
            ax.plot(x, y, z, color=col, lw=0.9)
    t = np.linspace(0, 2 * math.pi, 300)
    ax.plot(np.cos(t), np.sin(t), 0 * t, color="k", lw=1.6)
    ax.plot([0, 0], [0, 0], [-3.4, 3.4], color="k", lw=1.6)
    ax.text(0.6, -1.25, -0.1, r"$h^{-1}(\infty)$", fontsize=9)
    ax.text(0.1, 0.1, 3.2, r"$h^{-1}(0)$", fontsize=9)
    ax.view_init(elev=22, azim=-58)
    clean3d(ax, 2.6)
    save(fig, "fig_hopf_tori.png")


# ------------------------------------------- 2. Villarceau bitangent plane
def fig_villarceau():
    a = 0.6
    b = math.sqrt(1 - a * a)
    R, r = 1 / a, b / a
    fig = plt.figure(figsize=(7.5, 5.6))
    ax = fig.add_subplot(111, projection="3d")
    X, Y, Z = torus_surface(a, 90, 45)
    ax.plot_surface(X, Y, Z, color="#aab7c4", alpha=0.22, linewidth=0,
                    shade=True)
    # bitangent plane x3 = (b/a) x1
    u, v = np.meshgrid(np.linspace(-2.6, 2.6, 2), np.linspace(-2.6, 2.6, 2))
    ax.plot_surface(u, v, (b / a) * u, color="#f5cba7", alpha=0.25,
                    linewidth=0)
    x, y, z = hopf_fibre(a, 0.0)
    ax.plot(x, y, z, color="#c0392b", lw=2.2)
    x, y, z = anti_fibre(a, 0.0)
    ax.plot(x, y, z, color="#2471a3", lw=2.2)
    # tangency points: (+-R cos^2 phi?...) computed as the two common points
    phi = math.asin(r / R)
    for s in (1, -1):
        p = s * np.array([R * math.cos(phi) * math.cos(phi) * 0 + math.sqrt(R * R - r * r) * math.cos(phi), 0,
                          math.sqrt(R * R - r * r) * math.sin(phi)])
        ax.scatter(*p, color="k", s=18)
    ax.scatter(0, r, 0, color="#c0392b", s=14)
    ax.scatter(0, -r, 0, color="#2471a3", s=14)
    ax.view_init(elev=18, azim=-35)
    clean3d(ax, 2.1)
    save(fig, "fig_villarceau.png")


# ---------------------------------------- 3. base points and linked fibres
def fig_base_and_fibres():
    fig = plt.figure(figsize=(9.2, 4.4))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")
    uu, vv = np.meshgrid(np.linspace(0, 2 * math.pi, 60),
                         np.linspace(0, math.pi, 30))
    ax1.plot_surface(np.cos(uu) * np.sin(vv), np.sin(uu) * np.sin(vv),
                     np.cos(vv), color="#d5dbdb", alpha=0.25, linewidth=0)
    lats = [0.55, 0.0, -0.6]
    for li, h3 in enumerate(lats):
        n = 7
        for k in range(n):
            ang = 2 * math.pi * k / n
            col = hsv_to_rgb((k / n, 0.75, 0.85 - 0.18 * li))
            s = math.sqrt(1 - h3 * h3)
            ax1.scatter(s * math.cos(ang), s * math.sin(ang), h3, color=col,
                        s=22)
            # point p = (2 z conj w, |z|^2-|w|^2): |z|^2 = (1+h3)/2
            a = math.sqrt((1 + h3) / 2)
            x, y, z = hopf_fibre(a, ang)
            ax2.plot(x, y, z, color=col, lw=1.1)
    ax1.view_init(elev=20, azim=-60)
    clean3d(ax1, 1.0)
    ax2.view_init(elev=24, azim=-60)
    clean3d(ax2, 2.3)
    ax1.set_title(r"points of $S^2$", fontsize=10)
    ax2.set_title(r"their fibres in $\mathbb{R}^3\cup\{\infty\}$", fontsize=10)
    save(fig, "fig_base_and_fibres.png")


# ----------------------------------- 4. preimages of [z^p : w^q]: torus links
def preimage(p, q, c, n=3000):
    """Components of {z^p = c w^q} on S^3, c > 0."""
    from scipy.optimize import brentq
    a = brentq(lambda s: s**p - c * (1 - s * s)**(q / 2), 1e-9, 1 - 1e-12)
    b = math.sqrt(1 - a * a)
    g = math.gcd(p, q)
    comps = []
    s = np.linspace(0, 2 * math.pi, n)
    for k in range(g):
        z = a * np.exp(1j * q * s / g)
        w = b * np.exp(1j * (p * s / g + 2 * math.pi * k / q))
        comps.append(stereo(z, w))
    return comps


def fig_torus_links():
    fig = plt.figure(figsize=(9.6, 4.6))
    for i, (p, q, title) in enumerate([(3, 1, r"$[z^3:w]$, $H=3$"),
                                      (2, 3, r"$[z^2:w^3]$, $H=6$")]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        for c, col in [(0.45, "#c0392b"), (2.2, "#2471a3")]:
            for x, y, z in preimage(p, q, c):
                ax.plot(x, y, z, color=col, lw=1.6)
        ax.view_init(elev=78, azim=-90)
        clean3d(ax, 2.0)
        ax.set_title(title, fontsize=10)
    save(fig, "fig_torus_links.png")


# ------------------------------------ 5. meridian sections of the image tori
def fig_meridians():
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    t = np.linspace(0, 2 * math.pi, 400)
    for a in [0.3, 0.45, 0.6, 0.75, 0.9, 0.97]:
        b = math.sqrt(1 - a * a)
        R, r = 1 / a, b / a
        ax.plot(R + r * np.cos(t), r * np.sin(t), color=cm.viridis(a), lw=1.2)
        ax.plot(-R - r * np.cos(t), r * np.sin(t), color=cm.viridis(a),
                lw=1.2)
    a = 0.6
    b = math.sqrt(1 - a * a)
    xs = np.linspace(-3.4, 3.4, 2)
    ax.plot(xs, (b / a) * xs, color="#c0392b", lw=1.0, ls="--")
    ax.plot([1], [0], "ko", ms=3)
    ax.plot([-1], [0], "ko", ms=3)
    ax.axvline(0, color="k", lw=0.8)
    ax.axhline(0, color="0.6", lw=0.5)
    ax.set_xlim(-3.4, 3.4)
    ax.set_ylim(-2.4, 2.4)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$ (the plane $x_2=0$)")
    ax.set_ylabel(r"$x_3$")
    ax.text(1.05, -0.28, r"$1$", fontsize=9)
    ax.text(0.9, 2.05, r"bitangent line for $a=0.6$", color="#c0392b",
            fontsize=9)
    save(fig, "fig_meridians.png")


# ------------------------------------------------ 6. table of pi_{n+k}(S^n)
TABLE = {
    0: ["Z"] * 10,
    1: ["0", "Z", "2", "2", "2", "2", "2", "2", "2", "2"],
    2: ["0", "2", "2", "2", "2", "2", "2", "2", "2", "2"],
    3: ["0", "2", "12", "Z+12", "24", "24", "24", "24", "24", "24"],
    4: ["0", "12", "2", "2+2", "2", "0", "0", "0", "0", "0"],
    5: ["0", "2", "2", "2+2", "2", "Z", "0", "0", "0", "0"],
    6: ["0", "2", "3", "24+3", "2", "2", "2", "2", "2", "2"],
    7: ["0", "3", "15", "15", "30", "60", "120", "Z+120", "240", "240"],
}


def fig_table():
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    colours = {"hopf": "#f5b7b1", "stable": "#d6eaf8", "cited": "#fdfefe",
               "trivial": "#eaeded"}
    for k in range(8):
        for n in range(1, 11):
            entry = TABLE[k][n - 1]
            if k == 0 or n == 1:
                kind = "trivial"
            elif n in (2, 4, 8) and k >= 1:
                kind = "hopf"
            elif n >= k + 2:
                kind = "stable"
            else:
                kind = "cited"
            ax.add_patch(plt.Rectangle((n - 0.5, 7.5 - k - 0.5), 1, 1,
                                       facecolor=colours[kind],
                                       edgecolor="0.55", lw=0.6))
            txt = entry.replace("+", r"\oplus ")
            if txt == "Z":
                s = r"$\mathbb{Z}$"
            elif txt == "0":
                s = r"$0$"
            else:
                s = "$" + txt.replace("Z", r"\mathbb{Z}") + "$"
            ax.text(n, 7.5 - k, s, ha="center", va="center", fontsize=8.5)
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(-0.5, 8.5)
    ax.set_xticks(range(1, 11))
    ax.set_yticks([7.5 - k for k in range(8)])
    ax.set_yticklabels([str(k) for k in range(8)])
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$k$")
    ax.set_title(r"$\pi_{n+k}(S^n)$; cyclic groups written by their orders",
                 fontsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=colours[c],
                             edgecolor="0.55") for c in
               ("hopf", "stable", "cited")]
    ax.legend(handles, ["derived from columns $n=3,7$ by the Hopf bundles",
                        r"stable range $n\geq k+2$",
                        "taken from Toda's tables"],
              loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3,
              fontsize=8, frameon=False)
    save(fig, "fig_table.png")


# -------------------------------------------- 7. orders of the image of J
def fig_imageJ():
    from fractions import Fraction
    import sympy as sp
    ks = list(range(1, 11))
    orders = [int(sp.fraction(sp.nsimplify(sp.bernoulli(2 * k) / (4 * k)))[1])
              for k in ks]
    orders = [abs(o) for o in orders]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.bar([4 * k - 1 for k in ks], [math.log2(o) for o in orders],
           width=1.6, color="#5d6d7e")
    for k, o in zip(ks, orders):
        ax.text(4 * k - 1, math.log2(o) + 0.25, str(o), ha="center",
                fontsize=8)
    ax.set_xticks([4 * k - 1 for k in ks])
    ax.set_xlabel(r"stem $4k-1$")
    ax.set_ylabel(r"$\log_2 |\mathrm{im}\,J|$")
    ax.set_ylim(0, 18.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    save(fig, "fig_imageJ.png")


# --------------------------------- 8. the Clifford torus and the two families
def fig_clifford():
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    s = np.linspace(0, 2 * math.pi, 200)
    for c in np.linspace(0, 2 * math.pi, 9)[:-1]:
        ax.plot(np.mod(s + c, 2 * math.pi), s, ".", color="#c0392b", ms=0.8)
        ax.plot(np.mod(-s + c, 2 * math.pi), s, ".", color="#2471a3", ms=0.8)
    ax.set_xlim(0, 2 * math.pi)
    ax.set_ylim(0, 2 * math.pi)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\alpha=\arg z$")
    ax.set_ylabel(r"$\beta=\arg w$")
    ax.set_xticks([0, math.pi, 2 * math.pi])
    ax.set_xticklabels(["0", r"$\pi$", r"$2\pi$"])
    ax.set_yticks([0, math.pi, 2 * math.pi])
    ax.set_yticklabels(["0", r"$\pi$", r"$2\pi$"])
    save(fig, "fig_clifford.png")


# ------------------------------- 9. zero spheres and the split degree formula
def fig_split():
    fig = plt.figure(figsize=(9.6, 4.4))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    u, v = np.meshgrid(np.linspace(0, 2 * math.pi, 60), np.linspace(0, math.pi, 30))
    R0 = 0.5
    ax.plot_surface(R0 * np.cos(u) * np.sin(v), R0 * np.sin(u) * np.sin(v), R0 * np.cos(v),
                    color="tab:blue", alpha=0.18, linewidth=0, shade=True)
    J = np.array([0.6, 0.0, 0.8])
    t = np.linspace(0, 2 * math.pi, 300)
    e2 = np.array([0.0, 1.0, 0.0])
    circ = R0 * (np.outer(np.cos(t), J) + np.outer(np.sin(t), e2))
    ax.plot(*circ.T, color="tab:blue", lw=1.0)
    # zeros of a^2 + 1/4 - w for w = 0.02 + 0.03 J: the square roots of w - 1/4 in C_J
    z = np.sqrt(complex(0.02 - 0.25, 0.03))
    for sgn in (1, -1):
        pt = sgn * z.imag * J
        ax.scatter(*pt, color="tab:red", s=28, depthshade=False)
    ax.plot(*np.array([-0.8 * J, 0.8 * J]).T, color="0.35", lw=0.8, ls="--")
    ax.text(*(0.84 * J), r"$J$", fontsize=10)
    ax.text(0.05, -0.75, -0.62, r"zeros of $a^2+\frac{1}{4}$: the sphere $|\mathrm{Im}\,a|=\frac{1}{2}$", fontsize=8.5)
    clean3d(ax, 0.62)
    ax.view_init(elev=18, azim=-58)
    ax.set_title(r"Imaginary part of $\mathbb{H}$", fontsize=10)

    ax2 = fig.add_subplot(1, 2, 2)
    th = np.linspace(0, math.pi / 2, 200)
    ax2.fill_between(np.cos(th), 0, np.sin(th), color="tab:blue", alpha=0.08)
    ax2.plot(np.cos(th), np.sin(th), color="tab:blue", lw=1.2)
    r = [0.4, math.sqrt(0.6), math.sqrt(0.97)]
    s_ = [0.25, 0.5, 0.75]
    for ra in r:
        for sb in s_:
            inside = ra ** 2 + sb ** 2 < 1
            ax2.scatter(ra, sb, s=40, color="tab:red" if inside else "white",
                        edgecolors="tab:red", zorder=3)
    for ra in r:
        ax2.axvline(ra, color="0.8", lw=0.6, zorder=0)
    for sb in s_:
        ax2.axhline(sb, color="0.8", lw=0.6, zorder=0)
    ax2.set_xlim(0, 1.08)
    ax2.set_ylim(0, 1.08)
    ax2.set_aspect("equal")
    ax2.set_xlabel(r"radius $r_\alpha$ of a zero of $P$")
    ax2.set_ylabel(r"radius $s_\beta$ of a zero of $Q$")
    ax2.text(0.6, 0.98, r"five pairs inside: degree $5$", fontsize=8.5)
    ax2.set_title(r"$\deg(F/|F|)=\sum_{r_\alpha^2+s_\beta^2<1}\mu_\alpha\nu_\beta$", fontsize=10)
    save(fig, "fig_split.png")


if __name__ == "__main__":
    fig_hopf_tori()
    fig_villarceau()
    fig_base_and_fibres()
    fig_torus_links()
    fig_meridians()
    fig_table()
    fig_imageJ()
    fig_clifford()
    fig_split()
    print("figures written to", OUT)
