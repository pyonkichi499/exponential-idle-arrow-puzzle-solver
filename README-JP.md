# Exponential Idle 矢印パズルソルバー

モバイルゲーム[Exponential Idle](https://conicgames.github.io/exponentialidle/)に登場する矢印パズルを解くためのPythonライブラリおよびCLIツールです。

## 概要

Exponential Idleの矢印パズルは以下の特徴を持つ論理パズルです：
- ボードは0から4の値を持つセルで構成されます
- セルをタップすると、そのセルと上下左右の隣接セルの値が増加します（5で割った余り）
- 目標は全てのセルを1にすることです
- Propagationアルゴリズムとhard/expertモード戦略を使用して解くことができます

## インストール

### 前提条件

- Python 3.8以上
- [Rye](https://rye-up.com/)（Pythonプロジェクトマネージャー）

### セットアップ

1. リポジトリをクローン：
```bash
git clone https://github.com/yourusername/arrow-puzzle-solver.git
cd arrow-puzzle-solver
```

2. Ryeを使用して依存関係をインストール：
```bash
rye sync
```

## 使い方

### コマンドラインインターフェース

ソルバーは複数のコマンドを持つCLIを提供します：

#### パズルを解く

```bash
# 対話的入力
rye run python -m arrow_puzzle_solver solve

# ファイルから読み込み
rye run python -m arrow_puzzle_solver solve --input-file puzzle.txt

# 特定のモードで実行
rye run python -m arrow_puzzle_solver solve --mode expert
```

#### ランダムパズルを生成

```bash
# 7x7の中難易度パズルを生成
rye run python -m arrow_puzzle_solver generate --size 7 --difficulty medium
```

#### デモを実行

```bash
rye run python -m arrow_puzzle_solver demo
```

### Python API

プログラムから使用することも可能です：

```python
from arrow_puzzle_solver import Board, Solver

# ボードを作成
board = Board(size=7)

# 初期値を設定（0-4）
board.set_value(0, 0, 2)
board.set_value(0, 1, 3)
# ... 他の値を設定

# ソルバーを作成して解く
solver = Solver(board)
if solver.solve(mode='expert'):
    print("解けました！")
    print(solver.board)
    print(f"総移動数: {len(solver.moves)}")
else:
    print("パズルを解けませんでした")
```

## アルゴリズム

### Propagation手法

各行を処理する基本的な解法戦略：
1. 中央のタイルを解く
2. 中央の左側のタイルを解く（1、2、3マス）
3. 中央の右側のタイルを解く（1、2、3マス）
4. 最下行を除く全ての行で繰り返す

### Hard/Expertモード

難しいパズル用の高度な戦略：
1. まずPropagationを適用
2. 最下行の情報を最上行にエンコード
3. 最下行の値に基づいて特定のタップシーケンスを適用
4. 再度上からPropagationを実行

## 入力フォーマット

パズルはスペースまたはカンマ区切りの値で表現されます：

```
2 3 1 4 0 2 3
1 4 2 3 1 0 4
3 0 4 1 2 3 1
4 2 1 0 3 4 2
0 3 2 4 1 2 0
2 1 3 2 4 0 3
3 4 0 1 2 3 4
```

## 開発

### プロジェクト構造

```
arrow-puzzle-solver/
├── src/arrow_puzzle_solver/
│   ├── __init__.py
│   ├── board.py        # ボード表現
│   ├── solver.py       # 解法アルゴリズム
│   └── cli.py          # CLIインターフェース
├── tests/              # テストスイート
├── pyproject.toml      # プロジェクト設定
└── README.md
```

### テストの実行

```bash
rye run pytest tests/ -v
```

### コード品質

```bash
# コードフォーマット
rye run black src/ tests/

# リント
rye run ruff src/ tests/
```

## 制限事項

- 現在のアルゴリズムでは、ランダムに生成された全てのパズルが解けるわけではありません
- このアルゴリズムは、解けることが保証されているExponential Idleのパズル用に設計されています
- 現在は正方形のボード（デフォルト7x7）のみサポートしています

## 例

### 基本的な使用例

```python
from arrow_puzzle_solver import Board, Solver

# シンプルな3x3ボードを作成
board = Board(size=3)

# 解けるパターンを設定
board.tap(1, 1)  # 中央をタップ

# 解く
solver = Solver(board)
solver.solve()
```

### CLI例

```bash
# ファイルからパズルを解く
echo "2 3 1
      1 4 2
      3 0 4" > puzzle.txt
      
rye run python -m arrow_puzzle_solver solve --input-file puzzle.txt --size 3
```

## 参考資料

- [Exponential Idle Guide - Arrow Puzzles](https://exponential-idle-guides.netlify.app/guides/asd/)
- アルゴリズムの実装はガイドに記載されている解法に基づいています

## ライセンス

このプロジェクトはMITライセンスの下でライセンスされています - 詳細は[LICENSE](LICENSE)ファイルを参照してください。

## コントリビューション

コントリビューションを歓迎します！プルリクエストをお気軽に送信してください。