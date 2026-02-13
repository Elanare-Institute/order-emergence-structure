import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUIを使わないバックエンド
import matplotlib.pyplot as plt

def vicsek_mips(N=2000, L=20.0, v0=0.5, eta=0.3, R=1.0, T=200, dt=0.1, rho_max=4.0, save_snapshots=True):
    np.random.seed(42)
    pos = np.random.rand(N, 2) * L
    theta = np.random.uniform(0, 2*np.pi, N)

    # スナップショット保存用
    snapshots = []
    snapshot_times = [0, int(T/dt*0.25), int(T/dt*0.5), int(T/dt*0.75), int(T/dt)-1]

    r_history = []  # polarization履歴

    for t in range(int(T/dt)):
        # 近傍平均
        for i in range(N):
            dist = np.linalg.norm(pos - pos[i], axis=1)
            neighbors = dist < R
            avg_theta = np.mean(theta[neighbors])
            theta[i] = avg_theta + eta * (np.random.rand() - 0.5) * 2 * np.pi

        # density-dependent speed for MIPS
        rho_local = np.array([np.sum(np.linalg.norm(pos - pos[i], axis=1) < R) / (np.pi * R**2) for i in range(N)])
        v = v0 * (1 - rho_local / rho_max)
        v = np.clip(v, 0.01, v0)

        dx = v * np.cos(theta) * dt
        dy = v * np.sin(theta) * dt
        pos[:,0] = (pos[:,0] + dx) % L
        pos[:,1] = (pos[:,1] + dy) % L

        # polarization計算
        r = np.abs(np.mean(np.exp(1j * theta)))
        r_history.append(r)

        # スナップショット保存
        if save_snapshots and t in snapshot_times:
            snapshots.append((pos.copy(), theta.copy(), rho_local.copy(), t*dt))

    # 最終polarization r
    r_final = np.abs(np.mean(np.exp(1j * theta)))
    print(f"Vicsek+MIPS final r = {r_final:.4f}")

    return pos, theta, rho_local, r_history, snapshots

# シミュレーション実行
print("Vicsek+MIPS シミュレーション開始")
print("=" * 50)
pos, theta, rho_local, r_history, snapshots = vicsek_mips()

# 可視化1: 時間発展のスナップショット
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.flatten()

for idx, (pos_snap, theta_snap, rho_snap, time) in enumerate(snapshots[:4]):
    ax = axes[idx]
    # 密度を色で表現
    scatter = ax.scatter(pos_snap[:, 0], pos_snap[:, 1],
                        c=rho_snap, cmap='viridis', s=10, alpha=0.6)
    # 方向を矢印で表現（サンプリング）
    sample_idx = np.random.choice(len(pos_snap), 200, replace=False)
    ax.quiver(pos_snap[sample_idx, 0], pos_snap[sample_idx, 1],
             np.cos(theta_snap[sample_idx]), np.sin(theta_snap[sample_idx]),
             alpha=0.5, scale=30, width=0.003, color='red')
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.set_aspect('equal')
    ax.set_title(f't = {time:.1f}', fontsize=12)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(scatter, ax=ax, label='Local density')

plt.tight_layout()
plt.savefig('vicsek_mips_snapshots.png', dpi=150, bbox_inches='tight')
print("\nスナップショットを vicsek_mips_snapshots.png に保存しました")

# 可視化2: Polarization時間発展
fig, ax = plt.subplots(figsize=(10, 6))
time_array = np.arange(len(r_history)) * 0.1
ax.plot(time_array, r_history, linewidth=2)
ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('Polarization r', fontsize=12)
ax.set_title('Vicsek+MIPS: Polarization over time', fontsize=14)
ax.grid(True, alpha=0.3)
ax.axhline(y=r_history[-1], color='r', linestyle='--',
          label=f'Final r = {r_history[-1]:.4f}')
ax.legend()
plt.tight_layout()
plt.savefig('vicsek_mips_polarization.png', dpi=150, bbox_inches='tight')
print("Polarization時間発展を vicsek_mips_polarization.png に保存しました")

# 可視化3: 最終状態の詳細
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 粒子位置と密度
scatter = ax1.scatter(pos[:, 0], pos[:, 1], c=rho_local, cmap='viridis', s=15, alpha=0.7)
sample_idx = np.random.choice(len(pos), 300, replace=False)
ax1.quiver(pos[sample_idx, 0], pos[sample_idx, 1],
          np.cos(theta[sample_idx]), np.sin(theta[sample_idx]),
          alpha=0.6, scale=30, width=0.004, color='red')
ax1.set_xlim(0, 20)
ax1.set_ylim(0, 20)
ax1.set_aspect('equal')
ax1.set_title('Final State: Particle positions & density', fontsize=12)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
plt.colorbar(scatter, ax=ax1, label='Local density ρ')

# 密度分布のヒストグラム
ax2.hist(rho_local, bins=30, edgecolor='black', alpha=0.7)
ax2.set_xlabel('Local density ρ', fontsize=12)
ax2.set_ylabel('Count', fontsize=12)
ax2.set_title('Density Distribution (MIPS signature)', fontsize=12)
ax2.axvline(x=np.mean(rho_local), color='r', linestyle='--',
           label=f'Mean ρ = {np.mean(rho_local):.2f}')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('vicsek_mips_final_state.png', dpi=150, bbox_inches='tight')
print("最終状態を vicsek_mips_final_state.png に保存しました")

print("\n" + "=" * 50)
print("シミュレーション完了!")
print(f"最終 polarization: r = {r_history[-1]:.4f}")
print(f"平均局所密度: ρ = {np.mean(rho_local):.4f}")
print(f"密度の標準偏差: σ_ρ = {np.std(rho_local):.4f}")
print("MIPSによるクラスタリングは密度分布のばらつきで確認できます")
