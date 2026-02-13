import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUIを使わないバックエンド
import matplotlib.pyplot as plt

def simulate_kuramoto_transition(N=1000, K_values=None, Delta=1.0, T=100.0, dt=0.01, transient=50.0, n_runs=5):
    if K_values is None:
        K_values = np.linspace(0.0, 3.0, 31)
    rs_mean = []
    rs_std = []

    for K in K_values:
        r_runs = []
        for seed in range(n_runs):
            np.random.seed(seed)
            # Lorentzian frequency distribution: g(ω) = Δ / [π(ω² + Δ²)]
            u = np.random.rand(N)
            omega = Delta * np.tan(np.pi * (u - 0.5))
            theta = np.random.uniform(0, 2*np.pi, N)

            steps = int(T / dt)
            trans_steps = int(transient / dt)

            for step in range(steps):
                z = np.mean(np.exp(1j * theta))  # complex order parameter
                r = np.abs(z)
                psi = np.angle(z)
                dtheta = omega + K * r * np.sin(psi - theta)  # mean-field form
                theta = (theta + dtheta * dt) % (2 * np.pi)

            # 最終状態のr
            z_final = np.mean(np.exp(1j * theta))
            r_final = np.abs(z_final)
            r_runs.append(r_final)

        rs_mean.append(np.mean(r_runs))
        rs_std.append(np.std(r_runs))

    return np.array(K_values), np.array(rs_mean), np.array(rs_std)

# 実行例(論文再現)
Ks, rs, stds = simulate_kuramoto_transition()
plt.figure(figsize=(8, 5))
plt.errorbar(Ks, rs, yerr=stds, fmt='o-', capsize=3, label='Simulation (N=1000)')
plt.axvline(x=2.0, color='r', linestyle='--', label='Theory: Kc = 2Δ = 2')
plt.xlabel('Coupling K')
plt.ylabel('Order parameter r')
plt.title('Kuramoto Transition (Lorentzian Δ=1)')
plt.legend()
plt.grid(True)
plt.savefig('kuramoto_transition.png', dpi=150, bbox_inches='tight')
print("グラフを kuramoto_transition.png に保存しました")

# 近傍スケーリング確認(β≈0.5)
mask = Ks > 2.05
if np.any(mask):
    logK = np.log(Ks[mask] - 2.0)
    logR = np.log(rs[mask])
    beta, _ = np.polyfit(logK, logR, 1)
    print(f"Estimated exponent β ≈ {beta:.3f} (theory: 0.5)")
