import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUIを使わないバックエンド
import matplotlib.pyplot as plt

def simulate_hmf_to_kuramoto(N=500, gamma=10.0, K=2.5, Delta=1.0, T=50, dt=0.001):
    np.random.seed(42)
    u = np.random.rand(N)
    omega = Delta * np.tan(np.pi * (u - 0.5))
    theta = np.random.uniform(0, 2*np.pi, N)
    p = np.zeros(N)  # momentum

    steps = int(T / dt)
    for _ in range(steps):
        # HMF force
        z = np.mean(np.exp(1j * theta))
        r = np.abs(z)
        psi = np.angle(z)
        F = K * r * np.sin(psi - theta)
        dp = F - gamma * p + omega
        p += dp * dt
        theta += (p / 1.0) * dt   # mass=1
        theta %= 2*np.pi

    r_final = np.abs(np.mean(np.exp(1j * theta)))
    print(f"HMF (γ={gamma}) final r = {r_final:.4f}")
    return r_final

# γ→∞ でKuramotoに収束(Kc preserved)の検証
gamma_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
r_finals = []

print("HMF→Kuramoto収束の検証")
print("=" * 50)

for gamma in gamma_values:
    r = simulate_hmf_to_kuramoto(gamma=gamma)
    r_finals.append(r)

# グラフ作成
plt.figure(figsize=(10, 6))
plt.semilogx(gamma_values, r_finals, 'o-', linewidth=2, markersize=8)
plt.xlabel('Damping coefficient γ', fontsize=12)
plt.ylabel('Order parameter r', fontsize=12)
plt.title('HMF to Kuramoto Transition (K=2.5, Δ=1.0)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.axhline(y=r_finals[-1], color='r', linestyle='--',
            label=f'Kuramoto limit (γ→∞): r ≈ {r_finals[-1]:.3f}')
plt.legend()
plt.tight_layout()
plt.savefig('hmf_kuramoto_transition.png', dpi=150, bbox_inches='tight')
print("\nグラフを hmf_kuramoto_transition.png に保存しました")
print(f"\n結論: γ={gamma_values[-1]}で秩序パラメータ r ≈ {r_finals[-1]:.4f} (Kuramoto極限)")
