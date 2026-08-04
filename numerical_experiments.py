"""
Numerical experiments for IPRM in fractional Fourier series.
Includes: reconstruction plots, error decay, alpha comparison,
and IPRM vs Direct Gegenbauer comparison.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_gegenbauer, gammaln
import matplotlib
from matplotlib.ticker import MultipleLocator
matplotlib.rcParams['mathtext.fontset'] = 'cm'
# matplotlib.rcParams['font.size'] = 14

# ============================================================
# Test Functions
# ============================================================
# def f1(x):
#     """sgn(x): single jump at x=0, size 2"""
#     return np.sign(x)

# def f2(x):
#     """Piecewise exponential: jump at x=0, size 2"""
#     return np.where(x < 0, np.exp(x), 2 + np.exp(-x))

# def f3(x):
#     """Square wave: two jumps at x=±0.5, size 2"""
#     return np.where(np.abs(x) < 0.5, 1.0, -1.0)

# def f4(x):
#     """Piecewise polynomial: jump at x=0, size 2"""
#     return np.where(x < 0, x**2 - 1, x**2 + 1)

# def f5(x):
#     """Piecewise cos/exp: jump at x=0.5, size 2"""
#     return np.where(x < 0.5, np.cos(np.pi * x), 2 * np.exp(-(x - 0.5)))

# def f6(x):
#     """Three pieces: jumps at x=±0.5"""
#     x = np.atleast_1d(np.asarray(x, dtype=float))
#     result = np.zeros_like(x)
#     m1 = x < -0.5
#     m2 = (x >= -0.5) & (x < 0.5)
#     m3 = x >= 0.5
#     result[m1] = np.sin(2 * np.pi * x[m1])
#     result[m2] = x[m2]**2 + 1
#     result[m3] = -np.exp(-(x[m3] - 0.5))
#     return result

def f1(x):
    """Piecewise Runge (steep), poles at ±0.2i"""
    return np.where(x < 0, 1/(1+25*x**2) - 1, 1/(1+25*x**2) + 1)

# def f2(x):
#     """Piecewise rational, poles at ±i"""
#     return np.where(x < 0, x / (1 + x**2) - 1, x / (1 + x**2) + 1)

def f2(x):
    """Mild Runge, asymmetric jump at x=0.3"""
    return np.where(x < 0.3, 1/(1+4*x**2), 1/(1+4*(x-0.3)**2) + 1)

def f3(x):
    """Two jumps, Runge-type pieces"""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    result = np.zeros_like(x)
    result[x < -0.5] = 1/(1+16*x[x < -0.5]**2)
    m2 = (x >= -0.5) & (x < 0.5)
    result[m2] = 1/(1+9*x[m2]**2) + 1
    result[x >= 0.5] = 1/(1+16*x[x >= 0.5]**2)
    return result

def f4(x):
    """Piecewise tanh, poles at ±iπ/20"""
    return np.where(x < 0, np.tanh(10*x), np.tanh(10*x) + 2)

def f5(x):
    """Two jumps, piecewise tanh"""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    result = np.zeros_like(x)
    m1 = x < -0.5
    m2 = (x >= -0.5) & (x < 0.5)
    m3 = x >= 0.5
    result[m1] = np.tanh(6*(x[m1]+0.5)) - 1
    result[m2] = np.tanh(4*x[m2]) + 1
    result[m3] = np.tanh(6*(x[m3]-0.5)) + 1
    return result

def f6(x):
    """Three pieces, mixed types"""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    result = np.zeros_like(x)
    m1 = x < -0.5; m2 = (x >= -0.5) & (x < 0.5); m3 = x >= 0.5
    result[m1] = np.tanh(8*(x[m1]+0.5))
    result[m2] = 1/(1+16*x[m2]**2)
    result[m3] = np.exp(-5*(x[m3]-0.5))
    return result

def f8(x):
    """finitely gated LFM signal."""
    x = np.asarray(x, dtype=float)

    a = -0.55
    b = 0.35
    alpha0 = np.pi / 6
    k0 = 1
    phi0 = np.pi / 4

    signal = np.zeros_like(x, dtype=complex)
    mask = (x >= a) & (x < b)

    cot_alpha0 = np.cos(alpha0) / np.sin(alpha0)

    signal[mask] = (np.exp(1j * phi0) * np.exp(-0.5j * cot_alpha0 * x[mask]**2) * np.exp(1j * k0 * np.pi * x[mask]))

    return signal

def f7(x):
    """hard-edged optical aperture field."""
    x = np.asarray(x, dtype=float)

    alpha0 = np.pi / 5
    cot_alpha0 = np.cos(alpha0) / np.sin(alpha0)

    amplitude = np.zeros_like(x, dtype=complex)

    left = (x >= -0.8) & (x < -0.2)
    right = (x >= 0.15) & (x < 0.75)

    amplitude[left] = 1.0
    amplitude[right] = 0.45 * np.exp(1j * np.pi / 3)

    field = (amplitude * np.exp(-0.5j * cot_alpha0 * x**2))

    return field

# Discontinuity locations for each function
DISC = {
    'f1': [0.0],
    'f2': [0.3],
    'f3': [-0.5, 0.5],
    'f4': [0.0],
    'f5': [-0.5, 0.5],
    'f6': [-0.5, 0.5],
    'f8': [-0.55, 0.35],
    'f7': [-0.8, -0.2, 0.15, 0.75],
}

ALL_FUNCS = {
    'f1': (f1, DISC['f1'], r'$f_1(x)=\mathrm{sgn}(x)$'),
    'f2': (f2, DISC['f2'], r'$f_2(x)$: piecewise exponential'),
    'f3': (f3, DISC['f3'], r'$f_3(x)$: square wave'),
    'f4': (f4, DISC['f4'], r'$f_4(x)$: piecewise polynomial'),
    'f5': (f5, DISC['f5'], r'$f_5(x)$: asymmetric jump'),
    'f6': (f6, DISC['f6'], r'$f_6(x)$: three pieces'),
}

# ============================================================
# Core Computations
# ============================================================

def compute_frac_fourier_coeffs(f, alpha, N, M=2000):
    """Compute fractional Fourier coefficients c_{k,alpha}
    c_{k,alpha} = (1/2) int_{-1}^{1} f(x) e^{i/2 x^2 cot(alpha) - ikpi x} dx
    """
    nodes, weights = np.polynomial.legendre.leggauss(M)
    cot_a = np.cos(alpha) / np.sin(alpha)
    fvals = f(nodes)
    k_values = np.arange(-N, N + 1)
    
    # Phase: e^{i/2 x^2 cot(alpha) - ikpi x}
    chirp = np.exp(0.5j * cot_a * nodes**2)
    C_alpha = np.zeros(2 * N + 1, dtype=complex)
    for ik, k in enumerate(k_values):
        integrand = fvals * chirp * np.exp(-1j * k * np.pi * nodes)
        C_alpha[ik] = 0.5 * np.sum(weights * integrand)
    return C_alpha, k_values


def compute_W_alpha(k_values, m, alpha, lam, M=None):
    """Compute IPRM transformation matrix W_alpha.
    W_{k,l,alpha} = (1/2) int_{-1}^{1} C_l^lambda(x) e^{i/2 x^2 cot(alpha) - ikpi x} dx
    """
    cot_a = np.cos(alpha) / np.sin(alpha)
    k_max = int(np.max(np.abs(k_values)))
    if M is None:
        M = int(2 * (m + 1) + 2 * k_max + 2 * abs(cot_a) + 50)

    nodes, weights = np.polynomial.legendre.leggauss(M)

    # Gegenbauer values via three-term recurrence
    C = np.zeros((m + 1, M))
    C[0, :] = 1.0
    if m >= 1:
        C[1, :] = 2 * lam * nodes
    for l in range(1, m):
        C[l + 1, :] = (2 * (l + lam) * nodes * C[l, :]
                        - (l + 2 * lam - 1) * C[l - 1, :]) / (l + 1)

    chirp = np.exp(0.5j * cot_a * nodes**2)
    fourier = np.exp(-1j * np.pi * np.outer(k_values, nodes))
    E = fourier * chirp[np.newaxis, :]

    W = 0.5 * (E * weights[np.newaxis, :]) @ C.T
    return W


def compute_W_direct(k_values, m, alpha, lam, M=None):
    """Compute Direct Gegenbauer transformation matrix W_bar_{k,l,alpha}.
    W_bar_{k,l,alpha} = (1/h_l^lambda) int_{-1}^{1} (1-x^2)^{lambda-1/2} 
                         C_l^lambda(x) e^{i/2 x^2 cot(alpha) - ikpi x} dx
    """
    cot_a = np.cos(alpha) / np.sin(alpha)
    k_max = int(np.max(np.abs(k_values)))
    if M is None:
        M = int(2 * (m + 1) + 2 * k_max + 2 * abs(cot_a) + 50)

    nodes, weights = np.polynomial.legendre.leggauss(M)

    C = np.zeros((m + 1, M))
    C[0, :] = 1.0
    if m >= 1:
        C[1, :] = 2 * lam * nodes
    for l in range(1, m):
        C[l + 1, :] = (2 * (l + lam) * nodes * C[l, :]
                        - (l + 2 * lam - 1) * C[l - 1, :]) / (l + 1)

    # Weight function (1-x^2)^{lambda-1/2}
    w_geg = (1 - nodes**2)**(lam - 0.5)

    chirp = np.exp(0.5j * cot_a * nodes**2)
    fourier = np.exp(-1j * np.pi * np.outer(k_values, nodes))
    E = fourier * chirp[np.newaxis, :]

    # h_l^lambda
    def h_l(l):
        return np.exp(np.log(np.pi) + gammaln(l + 2 * lam) - (2 * lam - 1) * np.log(2)
                      - np.log(l + lam) - 2 * gammaln(lam) - gammaln(l + 1))

    W_bar = np.zeros((len(k_values), m + 1), dtype=complex)
    for l_idx in range(m + 1):
        integrand = w_geg * C[l_idx, :]
        W_bar[:, l_idx] = (E * weights[np.newaxis, :]) @ integrand / h_l(l_idx)

    return W_bar


def fractional_partial_sum(f, alpha, N, x_eval, M=2000):
    """Compute the fractional Fourier partial sum f_{N,alpha}(x)."""
    C_alpha, k_values = compute_frac_fourier_coeffs(f, alpha, N, M)
    cot_a = np.cos(alpha) / np.sin(alpha)

    result = np.zeros_like(x_eval, dtype=complex)
    for ik, k in enumerate(k_values):
        phi_k = np.exp(-0.5j * cot_a * x_eval**2) * np.exp(1j * k * np.pi * x_eval)
        result += C_alpha[ik] * phi_k
    return result


def iprm_reconstruct(f, alpha, m, N, lam, x_eval, M_quad=2000):
    """IPRM reconstruction on [-1,1] (single interval).
    m: Gegenbauer truncation degree
    N: Fourier truncation (2N+1 equations), requires N >= m/2
    """
    C_alpha, k_values = compute_frac_fourier_coeffs(f, alpha, N, M_quad)
    W = compute_W_alpha(k_values, m, alpha, lam)
    if m + 1 == len(k_values):
        G_hat = np.linalg.solve(W, C_alpha)
    else:
        G_hat, _, _, _ = np.linalg.lstsq(W, C_alpha, rcond=None)

    # Evaluate Gegenbauer expansion
    result = np.zeros_like(x_eval, dtype=complex)
    for l in range(m + 1):
        result += G_hat[l] * eval_gegenbauer(l, lam, x_eval)
    return np.real(result), G_hat


def direct_gegenbauer_reconstruct(f, alpha, m, N, lam, x_eval, M_quad=2000):
    """Direct Gegenbauer reconstruction.
    m: Gegenbauer truncation degree
    N: Fourier truncation
    """
    C_alpha, k_values = compute_frac_fourier_coeffs(f, alpha, N, M_quad)
    W_bar = compute_W_direct(k_values, m, alpha, lam)

    # g_bar_l = sum_k c_{k,alpha} * W_bar_{k,l}
    G_bar = W_bar.T @ C_alpha

    result = np.zeros_like(x_eval, dtype=complex)
    for l in range(m + 1):
        result += G_bar[l] * eval_gegenbauer(l, lam, x_eval)
    return np.real(result), G_bar


def iprm_piecewise(f, disc_pts, alpha, m, N, lam, x_eval, M_quad=2000):
    """IPRM reconstruction for piecewise functions.
    Apply IPRM on each smooth subinterval separately.
    m: Gegenbauer truncation degree
    N: Fourier truncation
    """
    # Build list of subintervals
    breakpoints = [-1.0] + sorted(disc_pts) + [1.0]
    result = np.zeros_like(x_eval)

    for i in range(len(breakpoints) - 1):
        a, b = breakpoints[i], breakpoints[i + 1]
        mask = (x_eval >= a) & (x_eval < b) if i < len(breakpoints) - 2 \
               else (x_eval >= a) & (x_eval <= b)
        if not np.any(mask):
            continue
        x_sub = x_eval[mask]

        # Map [a,b] to [-1,1]: t = 2(x-a)/(b-a) - 1
        t_sub = 2 * (x_sub - a) / (b - a) - 1

        # Define f on [-1,1] via the mapping
        def f_mapped(t, _a=a, _b=b):
            x_orig = _a + (t + 1) * (_b - _a) / 2
            return f(x_orig)

        recon, _ = iprm_reconstruct(f_mapped, alpha, m, N, lam, t_sub, M_quad)
        result[mask] = recon

    return result


def direct_gegenbauer_piecewise(f, disc_pts, alpha, m, N, lam, x_eval, M_quad=2000):
    """Direct Gegenbauer reconstruction for piecewise functions.
    m: Gegenbauer truncation degree
    N: Fourier truncation
    """
    breakpoints = [-1.0] + sorted(disc_pts) + [1.0]
    result = np.zeros_like(x_eval)

    for i in range(len(breakpoints) - 1):
        a, b = breakpoints[i], breakpoints[i + 1]
        mask = (x_eval >= a) & (x_eval < b) if i < len(breakpoints) - 2 \
               else (x_eval >= a) & (x_eval <= b)
        if not np.any(mask):
            continue
        x_sub = x_eval[mask]
        t_sub = 2 * (x_sub - a) / (b - a) - 1

        def f_mapped(t, _a=a, _b=b):
            x_orig = _a + (t + 1) * (_b - _a) / 2
            return f(x_orig)

        recon, _ = direct_gegenbauer_reconstruct(f_mapped, alpha, m, N, lam, t_sub, M_quad)
        result[mask] = recon

    return result


# ============================================================
# Experiment 1: Reconstruction plots (Gibbs vs IPRM vs Direct)
# ============================================================
def experiment1_reconstruction():
    """Show Gibbs phenomenon and IPRM reconstruction for all 6 functions."""
    alpha = np.pi / 4
    m = 16            # Gegenbauer truncation
    N = 10 * m        # Fourier truncation, N >> m
    lam = 0.75
    x_eval = np.linspace(-0.999, 0.999, 2000)

    # fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # for idx, (key, (func, discs, label)) in enumerate(ALL_FUNCS.items()):
    #     ax = axes[idx // 3, idx % 3]

    #     # Exact
    #     f_exact = func(x_eval)

    #     # Fractional Fourier partial sum (Gibbs)
    #     f_frac = fractional_partial_sum(func, alpha, N, x_eval)

    #     # IPRM
    #     f_iprm = iprm_piecewise(func, discs, alpha, m, N, lam, x_eval)

    #     ax.plot(x_eval, f_exact, 'k-', lw=2, label='Exact')
    #     ax.plot(x_eval, np.real(f_frac), 'b--', lw=1, alpha=0.7, label=f'FrFS $N={N}$')
    #     ax.plot(x_eval, f_iprm, 'r-', lw=1.5, label='IPRM')

    #     ax.set_title(label, fontsize=13)
    #     ax.set_xlim(-1, 1)
    #     ax.grid(True, alpha=0.3)
    #     if idx == 0:
    #         ax.legend(fontsize=10, loc='best')

    # plt.suptitle(rf'Reconstruction: $\alpha=\pi/4$, $m={m}$, $\lambda={lam}$',
    #              fontsize=15, y=1.01)
    # plt.tight_layout()
    # plt.show()

    for idx, (key, (func, discs, label)) in enumerate(ALL_FUNCS.items()):
        fig, ax = plt.subplots(figsize=(12, 8))

        # Exact
        f_exact = func(x_eval)

        # Fractional Fourier partial sum (Gibbs)
        f_frac = fractional_partial_sum(func, alpha, N, x_eval)

        # Direct
        f_direct = direct_gegenbauer_piecewise(func, discs, alpha, m, N, lam, x_eval)

        # IPRM
        f_iprm = iprm_piecewise(func, discs, alpha, m, N, lam, x_eval)
        
        
        # ax.plot(x_eval, f_exact, 'k--', lw=4, label='Exact')
        # ax.plot(x_eval, np.real(f_frac), 'b--', lw=3.5, alpha=0.7, label=f'FrFS')
        # ax.plot(x_eval, f_direct, 'g--', lw=3.5, alpha=0.7, label='Direct')
        # ax.plot(x_eval, f_iprm, 'r--', lw=3.5, label='IPRM')
        
        
        # # ax.set_title(label, fontsize=13)
        # ax.set_xlim(-1, 1)
        # ax.xaxis.set_major_locator(MultipleLocator(0.5))
        # ax.yaxis.set_major_locator(MultipleLocator(1))
        # ax.tick_params(axis='both', labelsize=25)
        # ax.set_xlabel('$x$', fontsize=30)
        # ax.grid(True, alpha=0.3)
        # ax.legend(fontsize=25, loc='upper left', framealpha=0.5)
        # plt.tight_layout()
        # plt.savefig(f'reconstruction_{key}.png', bbox_inches='tight')
        # plt.show()
        # plt.close()
        # print(f"Saved reconstruction_{key}.png")

        norm_exact_L2 = np.linalg.norm(f_exact) + 1e-16
        e_frfs_L2 = np.linalg.norm(f_exact - f_frac) / norm_exact_L2
        e_dir_L2  = np.linalg.norm(f_exact - f_direct) / norm_exact_L2
        e_iprm_L2 = np.linalg.norm(f_exact - f_iprm) / norm_exact_L2

        e_frfs_inf = np.max(np.abs(f_exact - f_frac))
        e_dir_inf  = np.max(np.abs(f_exact - f_direct))
        e_iprm_inf = np.max(np.abs(f_exact - f_iprm))

        print(f'{key} m={m}:')
        print(f'[L-inf ] FrFS={e_frfs_inf:.2e}  Direct={e_dir_inf:.2e}  IPRM={e_iprm_inf:.2e}')
        print(f'[Rel L2] FrFS={e_frfs_L2:.2e}  Direct={e_dir_L2:.2e}  IPRM={e_iprm_L2:.2e}')


# ============================================================
# Experiment 2: Error decay curves (error vs N)
# ============================================================
def experiment2_error_decay():
    """L^inf error vs m for IPRM on each test function."""
    alpha = np.pi / 4
    lam = 0.75
    m_list = [4, 8, 12, 16, 20, 24, 28, 32]
    x_eval = np.linspace(-0.999, 0.999, 2000)

    for idx, (key, (func, discs, label)) in enumerate(ALL_FUNCS.items()):
        fig, ax = plt.subplots(figsize=(12, 8))
        f_exact = func(x_eval)

        errors = []
        for m in m_list:
            N = 10 * m  # N >> m
            f_iprm = iprm_piecewise(func, discs, alpha, m, N, lam, x_eval)
            err = np.max(np.abs(f_exact - f_iprm))
            errors.append(err)
            print(f"  {key}, m={m}: error = {err:.4e}")

        ax.semilogy(m_list, errors, 'ro-', lw=3.5, ms=6)
        ax.tick_params(axis='both', labelsize=30)
        ax.set_xlabel('$m$', fontsize=40)
        ax.set_ylabel(r'$\|f - f_{m,\alpha}\|_{L^\infty}$', fontsize=40)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'error_decay_{key}.png', bbox_inches='tight')
        plt.show()
        plt.close()
        print(f"Saved error_decay_{key}.png")

def experiment3_alpha_comparison():
    lam = 0.75
    m_list = [4, 8, 12, 16, 20, 24, 28, 32]
    alpha_list = [np.pi/16, np.pi/8, 3*np.pi/16, np.pi/4,
                  5*np.pi/16, 3*np.pi/8, 7*np.pi/16]
    alpha_labels = [r'$\pi/16$', r'$\pi/8$', r'$3\pi/16$', r'$\pi/4$',
                    r'$5\pi/16$', r'$3\pi/8$', r'$7\pi/16$']
    colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:purple',
              'tab:orange', 'tab:brown', 'tab:pink']
    x_eval = np.linspace(-0.999, 0.999, 2000)

    for key, (func, discs, label) in ALL_FUNCS.items():

        f_exact = func(x_eval)

        # 先收集所有误差数据
        all_errors = []  # shape: (n_alpha, n_m)
        for alpha in alpha_list:
            errors = []
            for m in m_list:
                N = 10 * m
                f_iprm = iprm_piecewise(func, discs, alpha, m, N, lam, x_eval)
                errors.append(np.max(np.abs(f_exact - f_iprm)))
            all_errors.append(errors)
        all_errors = np.array(all_errors)  # (7, 8)

        mean_errors = np.mean(all_errors, axis=0)       # (8,)
        rel_dev = np.abs(all_errors - mean_errors) / mean_errors  # (7, 8)

        fig, ax = plt.subplots(figsize=(12, 8))

        # ── 下图：相对偏差（越小越"无关"）──────────────────
        for ai, (alabel, color) in enumerate(zip(alpha_labels, colors)):
            ax.semilogy(m_list, rel_dev[ai] + 1e-16, 'o-',
                         color=color, lw=3.5, ms=6, alpha=0.7,
                         label=rf'$\alpha={alabel[1:-1]}$')

        ax.set_xlabel('$m$', fontsize=40)
        ax.set_ylabel(
            r'$\frac{|e_\alpha - \bar{e}|}{\bar{e}}$', fontsize=40
        )
        ax.tick_params(axis='both', labelsize=30)
        ax.legend(fontsize=25, ncol=2)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'error_alpha_{key}.png', bbox_inches='tight')
        plt.show()
        plt.close()
        print(f"Saved error_alpha_{key}.png")


        # # ── 双联图 ──────────────────────────────────────────
        # fig, (ax1, ax2) = plt.subplots(
        #     2, 1, figsize=(12, 12),
        #     gridspec_kw={'height_ratios': [3, 2]},
        #     sharex=True
        # )

        # # ── 上图：所有 α 线 + 加粗均值线 ──────────────────
        # for ai, (alpha, alabel, color) in enumerate(
        #         zip(alpha_list, alpha_labels, colors)):
        #     ax1.semilogy(m_list, all_errors[ai], 'o-',
        #                  color=color, lw=2, ms=6, alpha=0.55,
        #                  label=rf'$\alpha={alabel[1:-1]}$')

        # # 均值线加粗加黑，突出"它们都是同一条线"
        # ax1.semilogy(m_list, mean_errors, 'k--', lw=3, ms=0,
        #              label=r'mean over $\alpha$', zorder=5)

        # ax1.set_ylabel(r'$\|f - f_{m,\alpha}\|_{L^\infty}$', fontsize=36)
        # ax1.legend(fontsize=22, ncol=2)
        # ax1.tick_params(axis='both', labelsize=26)
        # ax1.grid(True, alpha=0.3)
        # # ax1.set_title(
        # #     rf'{label} — curves coincide $\Rightarrow$ result is $\alpha$-independent',
        # #     fontsize=28, pad=10
        # # )

        # # ── 下图：相对偏差（越小越"无关"）──────────────────
        # for ai, (alabel, color) in enumerate(zip(alpha_labels, colors)):
        #     ax2.semilogy(m_list, rel_dev[ai] + 1e-16, 'o-',
        #                  color=color, lw=2, ms=6, alpha=0.7,
        #                  label=rf'$\alpha={alabel[1:-1]}$')

        # # ax2.axhline(1e-14, color='gray', ls=':', lw=1.5,
        # #             label='machine $\epsilon$ level')
        # ax2.set_xlabel('$m$', fontsize=36)
        # ax2.set_ylabel(
        #     r'$\frac{|e_\alpha - \bar{e}|}{\bar{e}}$', fontsize=32
        # )
        # ax2.tick_params(axis='both', labelsize=26)
        # ax2.legend(fontsize=20, ncol=2)
        # ax2.grid(True, alpha=0.3)
        # # ax2.set_title(
        # #     r'Relative deviation from mean $\approx$ 0 (numerical noise only)',
        # #     fontsize=24
        # # )

        # plt.tight_layout()
        # plt.savefig(f'error_decay_{key}.png', bbox_inches='tight')
        # plt.show()
        # plt.close()
        # print(f"Saved error_decay_{key}.png")


# ============================================================
# Experiment 4: Practical FrFT examples
# ============================================================



def experiment4_practical_examples():
    """Imaginary components of LFM and optical aperture signals."""

    lam = 0.75
    m = 8
    N = 10 * m
    x_eval = np.linspace(-0.999, 0.999, 2000)

    examples = [('f8', f8, DISC['f8'], np.pi / 6, r'Gated LFM signal'),
        ('f7', f7, DISC['f7'], np.pi / 5, r'Optical aperture field')]

    components = [('real', np.real, r'$\operatorname{Re} f(x)$', 'Real part'),
                  ('imaginary', np.imag, r'$\operatorname{Im} f(x)$', 'Imaginary part')]

    for key, func, discs, alpha, title in examples:

        for (component_name, component, ylabel, component_title) in components:

            fig, ax = plt.subplots(figsize=(12, 8))

            def component_function(x, _func=func, _component=component):
                return _component(_func(x))

            f_exact = component_function(x_eval)
            f_frac = fractional_partial_sum(component_function, alpha, N, x_eval)
            f_direct = direct_gegenbauer_piecewise(component_function, discs, alpha, m, N, lam, x_eval)
            f_iprm = iprm_piecewise(component_function, discs, alpha, m, N, lam, x_eval)

            ax.plot(x_eval, f_exact, 'k--', lw=3, label='Exact')
            ax.plot(x_eval, np.real(f_frac), 'b--', lw=2.5, alpha=0.7, label='FrFS')
            ax.plot(x_eval, np.real(f_direct), 'g--.', lw=2.5, alpha=0.7, label='Direct')
            ax.plot(x_eval, np.real(f_iprm), 'r--', lw=2.5, label='IPRM')

            # for point in discs:
            #     ax.axvline(point, color='gray', linestyle=':', linewidth=4.0)

            ax.set_xlim(-1, 1)
            ax.xaxis.set_major_locator(MultipleLocator(0.5))
            ax.set_xlabel(r'$x$', fontsize=30)
            ax.set_ylabel(ylabel, fontsize=30)
            # ax.set_title(subtitle, fontsize=19)
            ax.tick_params(axis='both', labelsize=25)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=25, loc='upper left', framealpha=0.5)

            norm_exact = (np.linalg.norm(f_exact) + 1.0e-16)
            error_frac_inf = np.max(np.abs(f_exact - np.real(f_frac)))
            error_direct_inf = np.max(np.abs(f_exact - np.real(f_direct)))
            error_iprm_inf = np.max(np.abs(f_exact - np.real(f_iprm)))

            error_frac_l2 = (np.linalg.norm(f_exact - np.real(f_frac)) / norm_exact)
            error_direct_l2 = (np.linalg.norm(f_exact - np.real(f_direct)) / norm_exact)
            error_iprm_l2 = (np.linalg.norm(f_exact - np.real(f_iprm)) / norm_exact)

            print("\n" + "=" * 65)

            print(f'{key}: {component_title}, '
                  f'm={m}, N={N}, lambda={lam}')

            print("-" * 65)

            print(f'[L-inf ] '
                  f'FrFS={error_frac_inf:.3e}, '
                  f'Direct={error_direct_inf:.3e}, '
                  f'IPRM={error_iprm_inf:.3e}')

            print(f'[Rel L2] '
                  f'FrFS={error_frac_l2:.3e}, '
                  f'Direct={error_direct_l2:.3e}, '
                  f'IPRM={error_iprm_l2:.3e}')

            plt.tight_layout()

            filename = (f'{key}_{component_name}')
            plt.savefig(f'{filename}.png', bbox_inches='tight')
            print(f'Saved {filename}.png ')

            plt.show()
            plt.close()


# ============================================================
# Run all experiments
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Experiment 1: Reconstruction plots")
    print("=" * 60)
    experiment1_reconstruction()

    print("\n" + "=" * 60)
    print("Experiment 2: Error decay curves")
    print("=" * 60)
    experiment2_error_decay()

    print("\n" + "=" * 60)
    print("Experiment 3: Effect of alpha")
    print("=" * 60)
    experiment3_alpha_comparison()

    print("\n" + "=" * 60)
    print("Experiment 4: Practical plots")
    print("=" * 60)
    experiment4_practical_examples()

    print("\n\nAll experiments completed!")
