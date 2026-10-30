"""Independent numerical and symbolic checks of the finite identities used in
"Hopf Invariants of Polynomial Maps and Villarceau Spheres".

No statement of the paper rests on this program: every identity checked here
is proved by hand in the text.  Run:  python3 checks.py
"""
import itertools
import math
import sys
from fractions import Fraction

import numpy as np
import sympy as sp

RNG = np.random.default_rng(20261030)
PASSED = 0
FAILED = 0


def check(name, ok):
    global PASSED, FAILED
    if ok:
        PASSED += 1
    else:
        FAILED += 1
    print(("PASS " if ok else "FAIL ") + name)


# ---------------------------------------------------------------- algebras
def q_mul(p, q):
    a1, b1, c1, d1 = p
    a2, b2, c2, d2 = q
    return np.array([a1*a2 - b1*b2 - c1*c2 - d1*d2,
                     a1*b2 + b1*a2 + c1*d2 - d1*c2,
                     a1*c2 - b1*d2 + c1*a2 + d1*b2,
                     a1*d2 + b1*c2 - c1*b2 + d1*a2])


def q_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def o_mul(x, y):
    """Cayley-Dickson: (a,b)(c,d) = (ac - conj(d) b, d a + b conj(c))."""
    a, b = x[:4], x[4:]
    c, d = y[:4], y[4:]
    return np.concatenate([q_mul(a, c) - q_mul(q_conj(d), b),
                           q_mul(d, a) + q_mul(b, q_conj(c))])


def o_conj(x):
    return np.concatenate([q_conj(x[:4]), -x[4:]])


def o_inv(x):
    return o_conj(x) / np.dot(x, x)


ALG = {
    2: (lambda x, y: np.array([x[0]*y[0] - x[1]*y[1], x[0]*y[1] + x[1]*y[0]]),
        lambda x: np.array([x[0], -x[1]])),
    4: (q_mul, q_conj),
    8: (o_mul, o_conj),
}


def unit(d):
    v = RNG.normal(size=d)
    return v / np.linalg.norm(v)


# ------------------------------------------------- 1. octonion identities
x, y, z = (RNG.normal(size=8) for _ in range(3))
check("octonions are not associative",
      np.linalg.norm(o_mul(o_mul(x, y), z) - o_mul(x, o_mul(y, z))) > 1e-3)
check("octonions are alternative: (xx)y = x(xy), (yx)x = y(xx)",
      np.allclose(o_mul(o_mul(x, x), y), o_mul(x, o_mul(x, y))) and
      np.allclose(o_mul(o_mul(y, x), x), o_mul(y, o_mul(x, x))))
check("octonion norm is multiplicative",
      math.isclose(np.linalg.norm(o_mul(x, y)),
                   np.linalg.norm(x) * np.linalg.norm(y), rel_tol=1e-12))
check("Artin: (x b^-1) b = x and (x b) b^-1 = x",
      np.allclose(o_mul(o_mul(x, o_inv(y)), y), x) and
      np.allclose(o_mul(o_mul(x, y), o_inv(y)), x))
check("Moufang (xy)^-1 = y^-1 x^-1",
      np.allclose(o_inv(o_mul(x, y)), o_mul(o_inv(y), o_inv(x))))


# ------------------------------------------- 2. the Hopf maps h(a,b)
def hopf(a, b, d):
    mul, conj = ALG[d]
    return np.concatenate([2 * mul(a, conj(b)), [np.dot(a, a) - np.dot(b, b)]])


for d in (2, 4, 8):
    mul, conj = ALG[d]
    ok_norm = ok_fibre = ok_chart = True
    for _ in range(200):
        u = unit(2 * d)
        a, b = u[:d], u[d:]
        p = hopf(a, b, d)
        ok_norm &= math.isclose(np.dot(p, p), 1.0, rel_tol=1e-12)
        # the fibre over x = a b^{-1} is {(x c, c)}: test with a random c
        xx = mul(a, conj(b)) / np.dot(b, b)
        c = RNG.normal(size=d)
        v = np.concatenate([mul(xx, c), c])
        v /= np.linalg.norm(v)
        ok_fibre &= np.allclose(hopf(v[:d], v[d:], d), p, atol=1e-10)
        # stereographic chart: h = sigma(a b^{-1})
        s = (np.concatenate([2 * xx, [np.dot(xx, xx) - 1]]) /
             (np.dot(xx, xx) + 1))
        ok_chart &= np.allclose(s, p, atol=1e-10)
    check(f"d={d}: |h(a,b)| = 1 on the unit sphere", ok_norm)
    check(f"d={d}: h is constant on the subspace L_x = {{(x c, c)}}", ok_fibre)
    check(f"d={d}: h(a,b) = stereographic image of a b^-1", ok_chart)

# Riemannian submersion identity  h(u).h(v) = 2|<u,v>|^2 - 1  (complex case)
ok = True
for _ in range(200):
    u = unit(4)
    v = unit(4)
    uc = u[0::2] + 1j * u[1::2]
    vc = v[0::2] + 1j * v[1::2]
    lhs = np.dot(hopf(u[:2], u[2:], 2), hopf(v[:2], v[2:], 2))
    ok &= math.isclose(lhs, 2 * abs(np.vdot(uc, vc))**2 - 1, abs_tol=1e-12)
check("complex Hopf map: h(u).h(v) = 2|<u,v>|^2 - 1", ok)


# --------------------------- 3. degree of the power map x -> x^m on S^{d-1}
def power(xv, m, d):
    mul, _ = ALG[d]
    r = np.zeros(d)
    r[0] = 1.0
    for _ in range(m):
        r = mul(r, xv)
    return r


def local_sign(xv, m, d, h=1e-6):
    """Sign of the Jacobian of x -> x^m at x in S^{d-1}, w.r.t. bases
    x*e_k and y*e_k (k=1..d-1) of the tangent spaces at x and y = x^m."""
    mul, conj = ALG[d]
    y = power(xv, m, d)
    J = np.zeros((d - 1, d - 1))
    for k in range(1, d):
        e = np.zeros(d)
        e[k] = 1.0
        t = mul(xv, e)
        xp = xv + h * t
        xp /= np.linalg.norm(xp)
        xm = xv - h * t
        xm /= np.linalg.norm(xm)
        dv = (power(xp, m, d) - power(xm, m, d)) / (2 * h)
        w = mul(conj(y), dv)          # pull back to T_1 = Im
        J[:, k - 1] = w[1:]
    return np.sign(np.linalg.det(J))


for d in (4, 8):
    for m in (2, 3, 5):
        psi = 1.1
        yv = np.zeros(d)
        yv[0] = math.cos(psi)
        yv[1:] = unit(d - 1) * math.sin(psi)
        u = yv[1:] / np.linalg.norm(yv[1:])
        roots = []
        for k in range(m):
            th = (psi + 2 * math.pi * k) / m
            xv = np.zeros(d)
            xv[0] = math.cos(th)
            xv[1:] = u * math.sin(th)
            roots.append(xv)
        ok_roots = all(np.allclose(power(r, m, d), yv, atol=1e-10) for r in roots)
        deg = sum(local_sign(r, m, d) for r in roots)
        check(f"d={d}, m={m}: x^m = y has the m roots in C_y, degree {int(deg)}",
              ok_roots and deg == m)


# ------------------------------------------- 4. Gauss linking integrals
def gauss_linking(c1, c2):
    d1 = np.roll(c1, -1, axis=0) - c1
    d2 = np.roll(c2, -1, axis=0) - c2
    m1 = c1 + d1 / 2
    m2 = c2 + d2 / 2
    tot = 0.0
    for i in range(len(c1)):
        r = m1[i] - m2
        n = np.linalg.norm(r, axis=1)**3
        tot += np.sum(np.einsum('j,kj->k', d1[i], np.cross(d2, r)) / n)
    return tot / (4 * math.pi)


def stereo(P):
    return P[:, :3] / (1 - P[:, 3:4])


def fibre(z0, w0, n=1200):
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    z = z0 * np.exp(1j * t)
    w = w0 * np.exp(1j * t)
    return stereo(np.stack([z.real, z.imag, w.real, w.imag], axis=1))


ok = True
for _ in range(4):
    u, v = unit(4), unit(4)
    lk = gauss_linking(fibre(u[0] + 1j*u[1], u[2] + 1j*u[3]),
                       fibre(v[0] + 1j*v[1], v[2] + 1j*v[3]))
    ok &= abs(lk - 1) < 0.02
check("two random Hopf fibres have linking number 1", ok)


def preimage(p, q, c):
    """All components of f^{-1}([c:1]) for f = [z^p : w^q], c > 0."""
    g = math.gcd(p, q)
    comps = []
    a0 = float(sp.nsolve(sp.Symbol('a')**p - c * (1 - sp.Symbol('a')**2)**(q / 2), 0.7))
    b0 = math.sqrt(1 - a0 * a0)
    n = 4000
    for k in range(g):
        s = np.linspace(0, 2 * math.pi, n, endpoint=False)
        # z = a0 e^{i q s/g}, w = b0 e^{i (p s/g + 2 pi k / q)}  solves z^p = c w^q
        z = a0 * np.exp(1j * q * s / g)
        w = b0 * np.exp(1j * (p * s / g + 2 * math.pi * k / q))
        comps.append(np.stack([z.real, z.imag, w.real, w.imag], axis=1))
    return comps


for (p, q) in [(1, 1), (2, 1), (2, 3), (3, 2), (2, 2), (3, 4)]:
    A = preimage(p, q, 0.6)
    B = preimage(p, q, 1.9)
    ok_eq = all(np.allclose((C[:, 0] + 1j*C[:, 1])**p,
                            0.6 * (C[:, 2] + 1j*C[:, 3])**q, atol=1e-9) for C in A)
    lk = sum(gauss_linking(stereo(X), stereo(Y)) for X in A for Y in B)
    check(f"f=[z^{p}:w^{q}]: preimages lie on z^p = c w^q and link {round(lk, 3)} = pq",
          ok_eq and abs(lk - p * q) < 0.05)


# ------------------------------------------------ 5. Villarceau circles
a, t = sp.symbols('a t', positive=True)
b = sp.sqrt(1 - a**2)
D = 1 - b * sp.sin(t)
X = sp.Matrix([a * sp.cos(t), a * sp.sin(t), b * sp.cos(t)]) / D
c = sp.Matrix([0, b / a, 0])
rad = sp.simplify(sp.expand(((X - c).T * (X - c))[0] * D**2 * a**2 - D**2))
check("stereographic image of a Hopf fibre: |X - (0,b/a,0)|^2 = 1/a^2",
      sp.simplify(sp.expand_trig(rad).rewrite(sp.cos).subs(sp.sin(t)**2, 1 - sp.cos(t)**2)) == 0
      or all(abs(float(rad.subs({a: av, t: tv}))) < 1e-12
             for av in (0.2, 0.5, 0.9) for tv in (0.3, 1.7, 4.0)))
check("the image circle lies in the plane x3 = (b/a) x1",
      sp.simplify(X[2] - b / a * X[0]) == 0)
R, r = 1 / a, b / a
check("image tori: R = 1/a, r = b/a, R^2 - r^2 = 1",
      sp.simplify(R**2 - r**2 - 1) == 0)
rho2 = X[0]**2 + X[1]**2
torus = sp.simplify(((X.T * X)[0] + R**2 - r**2)**2 - 4 * R**2 * rho2)
check("the image circle lies on the torus (|X|^2+R^2-r^2)^2 = 4R^2 rho^2",
      all(abs(float(torus.subs({a: av, t: tv}))) < 1e-10
          for av in (0.2, 0.5, 0.9) for tv in (0.3, 1.7, 4.0)))

# Villarceau factorisation: in the plane x3 = tan(phi) x1, sin(phi) = r/R,
# the torus quartic is a constant times the product of the two circle equations.
Rs, rs, u_, v_ = sp.symbols('R r u v', positive=True)
cphi = sp.sqrt(Rs**2 - rs**2) / Rs
sphi = rs / Rs
x1, x2, x3 = u_ * cphi, v_, u_ * sphi           # u along the tilted line, v = x2
F = (x1**2 + x2**2 + x3**2 + Rs**2 - rs**2)**2 - 4 * Rs**2 * (x1**2 + x2**2)
Cp = u_**2 + (v_ - rs)**2 - Rs**2
Cm = u_**2 + (v_ + rs)**2 - Rs**2
check("Villarceau: torus quartic = (circle centred +r)(circle centred -r) in the bitangent plane",
      sp.expand(F - Cp * Cm) == 0)


# ------------------------------------- 6. order of the image of J in 4k-1
def bern(n):
    return sp.bernoulli(n)


expected = {1: 24, 2: 240, 3: 504, 4: 480, 5: 264, 6: 65520, 7: 24, 8: 16320,
            9: 28728, 10: 13200}
ok = True
for k, e in expected.items():
    den = sp.fraction(sp.nsimplify(bern(2 * k) / (4 * k)))[1]
    ok &= abs(den) == e
check("denominators of B_2k/4k, k=1..10: 24,240,504,480,264,65520,24,16320,28728,13200", ok)


# --------------------------- 6b. the formula for the denominators of B_2k/4k
def m_formula(n):
    """2^{2+v_2(n)} * prod_{p odd, (p-1)|n} p^{1+v_p(n)}, n = 2k."""
    val = 2**(2 + sp.multiplicity(2, n))
    for p in sp.primerange(3, n + 2):
        if n % (p - 1) == 0:
            val *= p**(1 + sp.multiplicity(p, n))
    return val


ok = all(sp.fraction(sp.nsimplify(bern(2 * k) / (4 * k)))[1] == m_formula(2 * k)
         for k in range(1, 61))
check("denominator of B_2k/4k = 2^(2+v2(2k)) prod_{(p-1)|2k} p^(1+vp(2k)), k <= 60", ok)


# --------------------------- 6c. further identities of Chapters 3, 4, 6, 7, 8
# quaternionic Hopf map: h(u).h(v) = 2|<u,v>|^2 - 1 with <u,v> = conj(a) a' + conj(b) b'
ok = True
for _ in range(200):
    u, v = unit(8), unit(8)
    a1, b1, a2, b2 = u[:4], u[4:], v[:4], v[4:]
    ip = q_mul(q_conj(a1), a2) + q_mul(q_conj(b1), b2)
    lhs = np.dot(hopf(a1, b1, 4), hopf(a2, b2, 4))
    ok &= math.isclose(lhs, 2 * np.dot(ip, ip) - 1, abs_tol=1e-12)
check("quaternionic Hopf map: h(u).h(v) = 2|<u,v>|^2 - 1", ok)

# octonions: (a,b) and (ac,bc) lie in different fibres for a = e1, b = e2, c = e4
E8 = np.eye(8)
lhs = o_mul(o_mul(E8[1], E8[4]), o_inv(o_mul(E8[2], E8[4])))
rhs = o_mul(E8[1], o_inv(E8[2]))
check("octonions: (e1 e4)(e2 e4)^-1 = e3 but e1 e2^-1 = -e3, so (ac,bc) is not in the fibre of (a,b)",
      np.allclose(lhs, E8[3]) and np.allclose(rhs, -E8[3]))

# derivative of x -> x^m on W_u is multiplication by sin(m th)/sin(th)
ok = True
for d in (4, 8):
    mul, conj = ALG[d]
    for m in (2, 3, 4, 7):
        th = 0.83
        uu = np.zeros(d); uu[1:] = unit(d - 1)
        xv = np.zeros(d); xv[0] = math.cos(th); xv += math.sin(th) * uu
        w = unit(d); w[0] = 0; w -= np.dot(w, uu) * uu; w /= np.linalg.norm(w)
        h = 1e-6
        dv = (power(xv + h * w, m, d) - power(xv - h * w, m, d)) / (2 * h)
        ok &= np.allclose(dv, math.sin(m * th) / math.sin(th) * w, atol=1e-6)
check("d=4,8: D(x^m)(w) = (sin m th / sin th) w for w orthogonal to C_u", ok)

# degree of F/|F| on S^3 computed by integrating the pull-back of the volume form
def degree_S3(F, N=64):
    th = np.linspace(0, math.pi / 2, N + 1)
    th = (th[:-1] + th[1:]) / 2
    al = np.linspace(0, 2 * math.pi, 2 * N, endpoint=False) + math.pi / (2 * N)
    tot = 0.0
    dth, dal = (math.pi / 2) / N, math.pi / N
    eps = 1e-6
    for t in th:
        A, B = np.meshgrid(al, al, indexing='ij')

        def G(t_, A_, B_):
            z = math.cos(t_) * np.exp(1j * A_)
            w = math.sin(t_) * np.exp(1j * B_)
            p, q = F(z, w)
            v = np.stack([p.real, p.imag, q.real, q.imag])
            return v / np.linalg.norm(v, axis=0)
        g0 = G(t, A, B)
        gt = (G(t + eps, A, B) - G(t - eps, A, B)) / (2 * eps)
        ga = (G(t, A + eps, B) - G(t, A - eps, B)) / (2 * eps)
        gb = (G(t, A, B + eps) - G(t, A, B - eps)) / (2 * eps)
        M = np.stack([g0, gt, ga, gb])           # 4 x 4 x grid
        det = np.linalg.det(np.moveaxis(M, (0, 1), (-2, -1)))
        tot += det.sum() * dth * dal * dal
    return tot / (2 * math.pi**2)


ORIENT = round(degree_S3(lambda z, w: (z, w)))   # the chart (t, alpha, beta) has this orientation
cases = [
    ("[z^2 - 1/4 : w]", lambda z, w: (z**2 - 0.25, w), 2),
    ("[z^2 - 4 : w]", lambda z, w: (z**2 - 4.0, w), 0),
    ("[z^2 : zw + 1]", lambda z, w: (z**2, z * w + 1), 0),
    ("[z^2 + w^2 : zw]", lambda z, w: (z**2 + w**2, z * w), 4),
    ("[z^2 : w^3]", lambda z, w: (z**2, w**3), 6),
]
for name, F, expect in cases:
    dg = degree_S3(F) / ORIENT
    check(f"Hopf invariant of {name} = degree of (P,Q)/|(P,Q)| = {round(dg, 3)} = {expect}",
          abs(dg - expect) < 0.02)


# Villarceau spheres for d = 4, 8 (Theorem E and the proposition on two families)
ok_sphere = ok_plane = ok_mirror = True
for d in (4, 8):
    mul, conj = ALG[d]
    eps_ = np.zeros(d); eps_[1] = 1.0          # a unit vector orthogonal to 1
    for av in (0.35, 0.8):
        bv = math.sqrt(1 - av**2)
        Rv, rv = 1 / av, bv / av

        def sig(A, B):
            be = np.dot(B, eps_)
            return np.concatenate([A, np.delete(B - be * eps_, 1)]) / (1 - be)

        def proj(x):
            return np.delete(x - np.dot(x, eps_) * eps_, 1)
        for _ in range(3):
            uu = unit(d)
            for _ in range(20):
                c = unit(d)
                P = sig(av * mul(uu, c), bv * c)
                x, y = P[:d], P[d:]
                centre = np.concatenate([rv * mul(uu, eps_), np.zeros(d - 1)])
                ok_sphere &= math.isclose(np.linalg.norm(P - centre), Rv, rel_tol=1e-10)
                # P lies in Pi_u = lambda_u(Pi_1): (u^-1 x, y) in Pi_1
                xi = mul(conj(uu), x)
                ok_plane &= np.allclose(y, rv * proj(xi), atol=1e-10)
                ok_plane &= math.isclose((np.linalg.norm(x) - Rv)**2 + np.dot(y, y), rv**2,
                                         rel_tol=1e-9)
                # mirror family: (rho(a c), b c) maps to the sphere centred at -r eps
                ac = av * c
                Pm = sig(ac - 2 * np.dot(ac, eps_) * eps_, bv * c)
                cm = np.concatenate([-rv * eps_, np.zeros(d - 1)])
                ok_mirror &= math.isclose(np.linalg.norm(Pm - cm), Rv, rel_tol=1e-10)
check("d=4,8: sigma(F_u) is the sphere of radius R about (u r eps, 0) in Pi_u, on the tube", ok_sphere and ok_plane)
check("d=4,8: the mirror family gives the spheres about (-r eps, 0)", ok_mirror)

xi_, eta_, r_ = sp.symbols('xi eta r', real=True)
Acal = xi_**2 + (1 + r_**2) * eta_ - 1
Gp = (xi_ - r_)**2 + (1 + r_**2) * eta_ - (1 + r_**2)
Gm = (xi_ + r_)**2 + (1 + r_**2) * eta_ - (1 + r_**2)
check("Pi_1 meets the tube in S^+ u S^-: (A+2)^2 - 4(1+r^2)(xi^2+eta) = G+ G-",
      sp.expand((Acal + 2)**2 - 4 * (1 + r_**2) * (xi_**2 + eta_) - Gp * Gm) == 0)

# general ring torus: the bitangent identity with sin(phi) free
s_ = sp.Symbol('s')
c_ = sp.Symbol('c')
F2 = ((u_**2 + v_**2) + Rs**2 - rs**2)**2 - 4 * Rs**2 * (u_**2 * c_**2 + v_**2)
check("bitangent identity: quartic = C+ C- + 4u^2 (R^2 sin^2 phi - r^2)",
      sp.expand(F2 - Cp * Cm - 4 * u_**2 * (Rs**2 * (1 - c_**2) - rs**2)) == 0)


# ------------------------- 7. consistency of the table with the splittings
# pi_{n+k}(S^n) for k = 0..7, n = 1..10, as abelian groups (lists of orders; 0 = Z)
T = {
    (1, 0): [0], (2, 0): [0], (3, 0): [0], (4, 0): [0], (5, 0): [0], (6, 0): [0], (7, 0): [0], (8, 0): [0], (9, 0): [0], (10, 0): [0],
    (2, 1): [0], (3, 1): [2], (4, 1): [2], (5, 1): [2], (6, 1): [2], (7, 1): [2], (8, 1): [2], (9, 1): [2], (10, 1): [2],
    (2, 2): [2], (3, 2): [2], (4, 2): [2], (5, 2): [2], (6, 2): [2], (7, 2): [2], (8, 2): [2], (9, 2): [2], (10, 2): [2],
    (2, 3): [2], (3, 3): [12], (4, 3): [0, 12], (5, 3): [24], (6, 3): [24], (7, 3): [24], (8, 3): [24], (9, 3): [24], (10, 3): [24],
    (2, 4): [12], (3, 4): [2], (4, 4): [2, 2], (5, 4): [2], (6, 4): [], (7, 4): [], (8, 4): [], (9, 4): [], (10, 4): [],
    (2, 5): [2], (3, 5): [2], (4, 5): [2, 2], (5, 5): [2], (6, 5): [0], (7, 5): [], (8, 5): [], (9, 5): [], (10, 5): [],
    (2, 6): [2], (3, 6): [3], (4, 6): [24, 3], (5, 6): [2], (6, 6): [2], (7, 6): [2], (8, 6): [2], (9, 6): [2], (10, 6): [2],
    (2, 7): [3], (3, 7): [15], (4, 7): [15], (5, 7): [30], (6, 7): [60], (7, 7): [120], (8, 7): [0, 120], (9, 7): [240], (10, 7): [240],
}
for n in range(1, 11):
    for k in range(0, 8):
        T.setdefault((n, k), [])
for k in range(1, 8):
    T[(1, k)] = []


def pi(i, n):
    """pi_i(S^n) from the table (None if outside the table)."""
    k = i - n
    if k < 0:
        return []
    if (n, k) in T:
        return sorted(T[(n, k)])
    if n > 10 and k <= 7:
        return sorted(T[(10, k)])
    return None


def same(g1, g2):
    """Compare finitely generated abelian groups given as cyclic orders
    (0 = Z) by their invariants: rank and the multiset of prime powers."""
    def inv(g):
        rank = g.count(0)
        pp = []
        for m in g:
            if m in (0, 1):
                continue
            for pr, e in sp.factorint(m).items():
                pp.append(pr**e)
        return rank, sorted(pp)
    return inv(g1) == inv(g2)


ok2 = all(same(pi(i, 2), pi(i, 3)) for i in range(3, 10))
check("pi_i(S^2) = pi_i(S^3) for 3 <= i <= 9 (complex Hopf bundle)", ok2)
ok4 = all(same(pi(i, 4), pi(i, 7) + pi(i - 1, 3)) for i in range(4, 11))
check("pi_i(S^4) = pi_i(S^7) + pi_{i-1}(S^3) for 4 <= i <= 10 (quaternionic)", ok4)
ok8 = all(same(pi(i, 8), (pi(i, 15) or []) + pi(i - 1, 7)) for i in range(8, 16))
check("pi_i(S^8) = pi_i(S^15) + pi_{i-1}(S^7) for 8 <= i <= 15 (octonionic)", ok8)
okF = all(same(pi(n + k, n), pi(n + 1 + k, n + 1))
          for k in range(0, 8) for n in range(k + 2, 10))
check("Freudenthal: pi_{n+k}(S^n) is constant for n >= k+2 in the table", okF)


# ------------------------- Riemannian submersion for every K (Proposition octsubmersion)
def tangent_svd(u, d, h=1e-6):
    J = np.zeros((d + 1, 2 * d))
    for k in range(2 * d):
        e = np.zeros(2 * d)
        e[k] = h
        J[:, k] = (hopf((u + e)[:d], (u + e)[d:], d)
                   - hopf((u - e)[:d], (u - e)[d:], d)) / (2 * h)
    q, _ = np.linalg.qr(np.column_stack([u, RNG.normal(size=(2 * d, 2 * d - 1))]))
    return np.sort(np.linalg.svd(J @ q[:, 1:], compute_uv=False))


ok = True
for d in (2, 4, 8):
    for _ in range(50):
        sv = tangent_svd(unit(2 * d), d)
        ok &= np.allclose(sv[-d:], 2, atol=1e-6) and np.allclose(sv[:-d], 0, atol=1e-6)
check("d=2,4,8: Dh has d singular values 2 and d-1 zeros on the tangent space", ok)

ok = True
for d in (2, 4, 8):
    mul, conj = ALG[d]
    for _ in range(50):
        x, y = RNG.normal(size=d), RNG.normal(size=d)
        u = y / (1 + x @ x)
        c = mul(conj(x), u)
        ok &= np.allclose(mul(x, c) + u, y) and np.allclose(
            [(mul(x, g) @ u - g @ mul(conj(x), u)) for g in np.eye(d)], 0)
check("Lemma lineperp: (y,0) = (x c, c) + (u, -conj(x) u), and (u, -conj(x) u) is orthogonal to L_x", ok)


# ------------------------- split quaternionic maps (Theorem split)
def q_root(c, k):
    """The k-th roots of a non-real quaternion c: those in the slice of c."""
    v = c[1:]
    nv = np.linalg.norm(v)
    roots = np.roots([1] + [0] * (k - 1) + [-complex(c[0], nv)])
    return [np.concatenate([[z.real], z.imag * v / nv]) for z in roots]


def q_pow(a, k):
    r = np.array([1.0, 0, 0, 0])
    for _ in range(k):
        r = q_mul(r, a)
    return r


def split_degree(m, n, c1, c2, y):
    """Signed count of preimages of y under F/|F|, F(a,b) = (a^m + c1, b^n + c2)."""
    def G(v):
        F = np.concatenate([q_pow(v[:4], m) + c1, q_pow(v[4:], n) + c2])
        return F / np.linalg.norm(F)
    y1, y2 = y[:4], y[4:]
    g = lambda lam: (np.linalg.norm(lam * y1 - c1) ** (2 / m)
                     + np.linalg.norm(lam * y2 - c2) ** (2 / n) - 1)
    grid = np.linspace(1e-9, 50, 200001)
    vals = np.array([g(t) for t in grid])
    total = 0
    for i in np.nonzero(np.sign(vals[:-1]) != np.sign(vals[1:]))[0]:
        lo, hi = grid[i], grid[i + 1]
        for _ in range(80):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if np.sign(g(mid)) == np.sign(g(lo)) else (lo, mid)
        lam = (lo + hi) / 2
        for a in q_root(lam * y1 - c1, m):
            for b in q_root(lam * y2 - c2, n):
                v = np.concatenate([a, b])
                v = v / np.linalg.norm(v)
                q, _ = np.linalg.qr(np.column_stack([v, RNG.normal(size=(8, 7))]))
                q[:, 0] = v
                if np.linalg.det(q) < 0:
                    q[:, 1] = -q[:, 1]
                h = 1e-6
                DG = np.column_stack([(G(v + h * e) - G(v - h * e)) / (2 * h)
                                      for e in q[:, 1:].T])
                total += int(np.sign(np.linalg.det(np.column_stack([G(v), DG]))))
    return total


y = unit(8)
c_small = np.array([0.05, 0.02, -0.03, 0.04])
c_big = np.array([3.0, 1.0, -2.0, 1.5])
degs = (split_degree(2, 1, c_small, np.zeros(4), y),
        split_degree(2, 3, c_small, 0.5 * c_small, y),
        split_degree(2, 3, c_big, c_big, y))
check("split quaternionic maps: deg F/|F| = 2, 6, 0 for zeros inside / inside / outside", degs == (2, 6, 0))


# ------------------ one-sided polynomials (Lemma qdiff, Props. qjac, normmult)
def a_pow(a, k, d):
    mul = ALG[d][0]
    r = np.zeros(d); r[0] = 1
    for _ in range(k):
        r = mul(r, a)
    return r


def onesided(coef, d):
    mul = ALG[d][0]
    return lambda q: sum(mul(a_pow(q, k, d), p) for k, p in enumerate(coef))


ok = True
for d in (4, 8):
    mul = ALG[d][0]
    for _ in range(40):
        coef = [RNG.normal(size=d) for _ in range(4)]
        P = onesided(coef, d)
        q = RNG.normal(size=d)
        x, y = q[0], np.linalg.norm(q[1:])
        I = np.concatenate([[0], q[1:] / y])
        A = sum(k * mul(a_pow(q, k - 1, d), p) for k, p in enumerate(coef) if k)
        B = sum(((x + 1j * y) ** k).imag / y * p for k, p in enumerate(coef))
        s, t = A @ B, mul(I, A) @ B
        h = 1e-6
        J = np.column_stack([(P(q + h * e) - P(q - h * e)) / (2 * h) for e in np.eye(d)])
        ok &= abs(np.linalg.det(J) - np.linalg.norm(B) ** (d - 4) * (s * s + t * t)) < 1e-4 * (1 + abs(np.linalg.det(J)))
check("d=4,8: det DP = |B|^(d-4) (<A,B>^2 + <IA,B>^2) for random one-sided cubics", ok)

ok = True
for d in (4, 8):
    mul = ALG[d][0]
    for _ in range(20):
        coef = [RNG.normal(size=d) for _ in range(4)]
        S = np.zeros(7)
        for i, pi in enumerate(coef):
            for j, pj in enumerate(coef):
                S[i + j] += pi @ pj
        P = onesided(coef, d)
        roots = [z for z in np.roots(S[::-1]) if z.imag > 1e-9]
        for z in roots:
            U = sum((z ** k).real * p for k, p in enumerate(coef))
            V = sum((z ** k).imag * p for k, p in enumerate(coef))
            J = -mul(U, ALG[d][1](V)) / (V @ V)
            ok &= abs(J[0]) < 1e-8 and abs(np.linalg.norm(J) - 1) < 1e-8
            ok &= np.linalg.norm(P(np.concatenate([[z.real], z.imag * J[1:]]))) < 1e-8
        ok &= len(roots) == 3
check("d=4,8: the roots of S_P in the upper half-plane give the zeros x + yJ, J = -U V^-1", ok)


def regular_preimages(w1, w2, d):
    """Preimages of (w1, w2) under F = (a^2 + 1/4, b^2 - 9/4 b + 1/2) in the unit ball."""
    out = []
    for c, poly in ((w1 - np.eye(d)[0] / 4, [1, 0]), (w2, [1, -9 / 4, 1 / 2])):
        v = c[1:] / np.linalg.norm(c[1:])
        zs = np.roots([1, poly[1], poly[2] - complex(c[0], np.linalg.norm(c[1:]))]
                      if len(poly) == 3 else [1, 0, -complex(c[0], np.linalg.norm(c[1:]))])
        out.append([np.concatenate([[z.real], z.imag * v]) for z in zs])
    return [(a, b) for a in out[0] for b in out[1] if a @ a + b @ b < 1]


ok = True
for d in (4, 8):
    mul = ALG[d][0]
    F = lambda v: np.concatenate([mul(v[:d], v[:d]) + np.eye(d)[0] / 4,
                                  mul(v[d:], v[d:]) - 9 / 4 * v[d:] + np.eye(d)[0] / 2])
    w1, w2 = 1e-3 * unit(d), 1e-3 * unit(d)
    pre = regular_preimages(w1, w2, d)
    ok &= len(pre) == 2
    for a, b in pre:
        v = np.concatenate([a, b]); h = 1e-6
        ok &= np.linalg.norm(F(v) - np.concatenate([w1, w2])) < 1e-9
        J = np.column_stack([(F(v + h * e) - F(v - h * e)) / (2 * h) for e in np.eye(2 * d)])
        ok &= np.linalg.det(J) > 0
check("Example split, d=4,8: a regular value near 0 has 2 preimages in the ball, both positive", ok)

ok = True
for d in (4, 8):
    mul, conj = ALG[d]
    E = np.eye(d)
    for _ in range(20):
        a = RNG.normal(size=d)
        word = a + sum(mul(mul(E[j], a), E[j]) for j in range(1, d))
        ok &= np.allclose(-word / (d - 2), conj(a))
        ok &= np.isclose(((d - 3) * a - (word - a))[0] / (2 * (d - 2)), a[0]) and np.allclose(
            ((d - 3) * a - (word - a))[1:] / (2 * (d - 2)), 0)
    M = np.column_stack([conj(e) for e in E])
    ok &= np.isclose(np.linalg.det(M), -1)
check("d=4,8: conj(a) = -(a + sum (e_j a) e_j)/(d-2), and conjugation has determinant -1", ok)

print(f"\n{PASSED} checks passed, {FAILED} failed")
sys.exit(1 if FAILED else 0)
