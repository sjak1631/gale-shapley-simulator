# Gale-Shapley マッチングシミュレーター

大学院入試研究室配属シミュレーション。Gale-Shapley アルゴリズム（学生提案型）を使用して、安定マッチングを計算します。

## セットアップ

### 前提条件
- `uv` がインストール済み（[uv のインストール](https://docs.astral.sh/uv/getting-started/installation/)）

### 環境構築

```bash
# uv で仮想環境を作成し、依存関係をインストール
uv sync

# （オプション）別環境から実行する場合
# uv run python3 simulation.py --help
```

## 使用方法

### 単純なシミュレーションモード（1回試行）

```bash
uv run python3 simulation.py --type simple
```

**オプション:**
- `--noise-std FLOAT` : ノイズ標準偏差（デフォルト: 1.6）
- `--seed INT` : 乱数シード（デフォルト: 42）
- `--applicants PATH` : applicants.csv のパス
- `--labs PATH` : labs.csv のパス

**例:**
```bash
uv run python3 simulation.py --type simple --noise-std 2.0 --seed 123
```

### 確率計算モード（複数試行）

スコア範囲を指定して、各スコアでの確率を計算します。

```bash
uv run python3 simulation.py --type probability
```

**オプション:**
- `--target-lab LAB` : 対象研究室（デフォルト: john）
- `--target-name NAME` : 対象受験者（デフォルト: me）
- `--score-range MIN MAX` : 能力スコア範囲（デフォルト: 0 10）
- `--trials INT` : 各スコアでの試行回数（デフォルト: 1000）
- `--noise-std FLOAT` : ノイズ標準偏差（デフォルト: 1.6）

**例:**
```bash
# john の確率を 10000 回試行で計算
uv run python3 simulation.py --type probability --target-lab john --trials 10000

# 特定の受験者の特定の研究室への確率
uv run python3 simulation.py --type probability --target-name me --target-lab john --score-range 15 20 --trials 500
```

## ヘルプ

```bash
uv run python3 simulation.py --help
```

## ファイル構成

```
matching-simulation/
├── pyproject.toml           # プロジェクト設定・依存関係定義
├── .python-version          # Python バージョン指定
├── README.md                # このファイル
├── simulation.py            # メインシミュレータ
├── applicants.csv           # 受験者データ
└── labs.csv                 # 研究室データ
```

## データフォーマット

### applicants.csv
```
氏名,ability_score,第1志望,第2志望,第3志望,第4志望,第5志望,第6志望
me,15,john,,,,,
jane,10,john,,,,,
...
```

### labs.csv
```
研究室名,定員
john,2
...
```

## アルゴリズム

Gale-Shapley マッチング（学生提案型）:
1. 各受験者の能力スコアにガウスノイズを加えてペーパーテストスコアを生成
2. 全研究室が同じテストスコアで受験者を評価（ランキング共通）
3. 受験者が志望順に提案、研究室が最高評価者を保持（不安定なら交換）
4. すべての受験者がマッチング完了または全志望先に拒否されるまで繰り返し

## ライセンス

（未指定）
