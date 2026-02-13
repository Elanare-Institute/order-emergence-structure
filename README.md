# Order Emergence Structure

蔵本モデル(Kuramoto Model)による同期現象のシミュレーション

## 概要

このプロジェクトは、蔵本モデルを用いて振動子集団の同期転移を再現します。
ローレンツ分布の固有周波数を持つ振動子系において、結合強度Kの増加に伴う秩序パラメータrの変化を観察できます。

## セットアップ

このプロジェクトでは[uv](https://docs.astral.sh/uv/)を使用して仮想環境を管理しています。

### 1. uvのインストール

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. 仮想環境の作成

```bash
uv venv
```

### 3. 仮想環境の有効化

```bash
source .venv/bin/activate
```

### 4. 依存パッケージのインストール

```bash
uv pip install numpy matplotlib
```

## 実行方法

```bash
source .venv/bin/activate
python kuramoto_simulation.py
```

## 理論的背景

- **臨界結合強度**: Kc = 2Δ (Δ=1の場合、Kc=2)
- **臨界指数**: β ≈ 0.5
- **平均場近似**: dθᵢ/dt = ωᵢ + K·r·sin(ψ - θᵢ)

## 参考文献

Kuramoto, Y. (1975). Self-entrainment of a population of coupled non-linear oscillators.