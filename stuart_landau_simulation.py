import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUIを使わないバックエンド
import matplotlib.pyplot as plt

def stuart_landau(alpha=0.5, beta=1.0, omega=1.0, T=50, dt=0.01):
    """
    Stuart-Landau方程式のシミュレーション
    dA/dt = (α + iω)A - β|A|²A

    α > 0: 超臨界Hopf分岐後の極限サイクル振動
    α < 0: 固定点への減衰
    """
    A = 0.01 + 0j
    As = []
    A_complex = []

    for _ in range(int(T/dt)):
        dA = (alpha + 1j*omega) * A - beta * np.abs(A)**2 * A
        A += dA * dt
        As.append(np.abs(A))
        A_complex.append(A)

    return np.array(As), np.array(A_complex)

print("Stuart-Landau方程式シミュレーション")
print("=" * 60)

# 1. αを変化させて超臨界Hopf分岐を観察
alphas = [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

results = {}

for idx, alpha in enumerate(alphas):
    As, A_complex = stuart_landau(alpha=alpha, T=50, dt=0.01)
    results[alpha] = (As, A_complex)

    time = np.arange(len(As)) * 0.01
    ax = axes[idx]
    ax.plot(time, As, linewidth=2)
    ax.set_xlabel('Time t', fontsize=10)
    ax.set_ylabel('Amplitude |A|', fontsize=10)
    ax.set_title(f'α = {alpha:.1f}', fontsize=12)
    ax.grid(True, alpha=0.3)

    # 理論値との比較
    if alpha > 0:
        theoretical_amplitude = np.sqrt(alpha / 1.0)  # β=1の場合
        ax.axhline(y=theoretical_amplitude, color='r', linestyle='--',
                  label=f'Theory: |A|=√(α/β)={theoretical_amplitude:.3f}')
        ax.legend(fontsize=8)

    final_amplitude = As[-1]
    print(f"α = {alpha:5.1f}: 最終振幅 |A| = {final_amplitude:.4f}")

plt.tight_layout()
plt.savefig('stuart_landau_hopf_bifurcation.png', dpi=150, bbox_inches='tight')
print("\nHopf分岐図を stuart_landau_hopf_bifurcation.png に保存しました")

# 2. 位相空間プロット（複素平面）
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, alpha in enumerate(alphas):
    As, A_complex = results[alpha]

    ax = axes[idx]
    ax.plot(A_complex.real, A_complex.imag, linewidth=1.5, alpha=0.7)
    ax.scatter([A_complex[0].real], [A_complex[0].imag],
              color='green', s=100, marker='o', label='Start', zorder=5)
    ax.scatter([A_complex[-1].real], [A_complex[-1].imag],
              color='red', s=100, marker='x', label='End', zorder=5)
    ax.set_xlabel('Re(A)', fontsize=10)
    ax.set_ylabel('Im(A)', fontsize=10)
    ax.set_title(f'Phase Space: α = {alpha:.1f}', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.legend(fontsize=8)

    # 極限サイクルの理論半径
    if alpha > 0:
        theta = np.linspace(0, 2*np.pi, 100)
        r_theory = np.sqrt(alpha / 1.0)
        ax.plot(r_theory * np.cos(theta), r_theory * np.sin(theta),
               'r--', linewidth=2, alpha=0.5, label=f'Limit cycle r={r_theory:.3f}')

plt.tight_layout()
plt.savefig('stuart_landau_phase_space.png', dpi=150, bbox_inches='tight')
print("位相空間図を stuart_landau_phase_space.png に保存しました")

# 3. 分岐図（α vs 最終振幅）
alpha_range = np.linspace(-2.0, 3.0, 51)
final_amplitudes = []

for alpha in alpha_range:
    As, _ = stuart_landau(alpha=alpha, T=100, dt=0.01)
    final_amplitudes.append(As[-1])

final_amplitudes = np.array(final_amplitudes)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(alpha_range, final_amplitudes, 'b-', linewidth=2, label='Simulation')

# 理論曲線
alpha_positive = alpha_range[alpha_range > 0]
theory_amplitudes = np.sqrt(alpha_positive / 1.0)
ax.plot(alpha_positive, theory_amplitudes, 'r--', linewidth=2,
       label='Theory: |A|=√(α/β)')

ax.axvline(x=0, color='gray', linestyle=':', linewidth=2, label='Hopf bifurcation (α=0)')
ax.set_xlabel('Parameter α', fontsize=12)
ax.set_ylabel('Steady-state amplitude |A|', fontsize=12)
ax.set_title('Supercritical Hopf Bifurcation Diagram', fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
ax.set_ylim(-0.1, 2.0)

plt.tight_layout()
plt.savefig('stuart_landau_bifurcation_diagram.png', dpi=150, bbox_inches='tight')
print("分岐図を stuart_landau_bifurcation_diagram.png に保存しました")

# 4. ωを変化させた場合の周波数依存性
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

omegas = [0.5, 1.0, 2.0, 5.0]
colors = plt.cm.viridis(np.linspace(0, 1, len(omegas)))

for omega, color in zip(omegas, colors):
    As, A_complex = stuart_landau(alpha=1.0, omega=omega, T=30, dt=0.01)
    time = np.arange(len(As)) * 0.01

    ax1.plot(time[:1000], As[:1000], linewidth=2, color=color, label=f'ω={omega}')
    ax2.plot(A_complex[:3000].real, A_complex[:3000].imag,
            linewidth=1, color=color, alpha=0.7, label=f'ω={omega}')

ax1.set_xlabel('Time t', fontsize=12)
ax1.set_ylabel('Amplitude |A|', fontsize=12)
ax1.set_title('Frequency dependence (α=1.0)', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend()

ax2.set_xlabel('Re(A)', fontsize=12)
ax2.set_ylabel('Im(A)', fontsize=12)
ax2.set_title('Phase space for different ω', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')
ax2.legend()

plt.tight_layout()
plt.savefig('stuart_landau_frequency.png', dpi=150, bbox_inches='tight')
print("周波数依存性を stuart_landau_frequency.png に保存しました")

print("\n" + "=" * 60)
print("シミュレーション完了!")
print("\nStuart-Landau方程式の特徴:")
print("- α < 0: 固定点 A=0 に減衰")
print("- α = 0: Hopf分岐点")
print("- α > 0: 半径 r=√(α/β) の極限サイクル振動")
print("- ω: 振動角周波数を決定")
