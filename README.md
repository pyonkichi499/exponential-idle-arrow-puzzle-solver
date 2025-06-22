# Puzzle Solver

様々なパズルを解くための統合フレームワークです。現在、以下のパズルに対応しています：
- Exponential Idleの矢印パズル
- 15パズル

## Overview

このプロジェクトは、拡張可能なパズルソルバーフレームワークを提供します。各パズルタイプは独自の実装を持ちながら、共通のインターフェースを通じて統一的に扱うことができます。

### 対応パズル

#### 矢印パズル (Arrow Puzzle)
Exponential Idleのゲーム内で出現する論理パズル：
- ボードは0から4の値を持つセルで構成
- セルをタップすると、そのセルと隣接する4方向のセルの値が+1される（mod 5）
- 目標：すべてのセルを1にする
- Propagationアルゴリズムとハード/エキスパートモード戦略を使用

#### 15パズル (15 Puzzle)
クラシックなスライディングパズル：
- 4x4のグリッドに1-15の数字と1つの空きスペース
- 空きスペースに隣接する数字をスライドして移動
- 目標：数字を昇順に並べる
- A*アルゴリズムを使用して最適解を探索

## Installation

### Prerequisites

- Python 3.8 or higher
- [Rye](https://rye-up.com/) (Python project manager)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/arrow-puzzle-solver.git
cd arrow-puzzle-solver
```

2. Install dependencies using Rye:
```bash
rye sync
```

## Usage

### Command Line Interface

#### 矢印パズル

```bash
# デモを実行
rye run python -m puzzle_solver arrow demo

# パズルを解く
rye run python -m puzzle_solver arrow solve --size 6 --board "0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0"

# ランダムなパズルを生成
rye run python -m puzzle_solver arrow generate --size 6

# 画像認識で自動解法（実験的機能）
rye run python -m puzzle_solver arrow auto-solve
```

#### 15パズル

```bash
# デモを実行
rye run python -m puzzle_solver fifteen demo

# パズルを解く
rye run python -m puzzle_solver fifteen solve --board "1,2,3,4,5,6,7,8,9,10,11,12,13,14,0,15"

# ランダムなパズルを生成
rye run python -m puzzle_solver fifteen generate
```

### Python API

```python
# 矢印パズル
from puzzle_solver.puzzles.arrow import ArrowBoard, ArrowSolver

board = ArrowBoard(size=7)
board.tap(0, 0)  # タップ操作
solver = ArrowSolver(board)
if solver.solve():
    print("Solved!")
    print(solver.get_solution())

# 15パズル
from puzzle_solver.puzzles.fifteen import FifteenBoard, FifteenSolver

board = FifteenBoard()
board.shuffle(100)  # 100回のランダム移動でシャッフル
solver = FifteenSolver(board)
if solver.solve():
    print("Solved!")
    for move in solver.get_solution():
        print(f"Move: {move}")
```

## Architecture

このプロジェクトは拡張可能なアーキテクチャを採用しています：

```
puzzle_solver/
├── core/               # 共通インターフェースとベースクラス
│   ├── base_board.py   # パズルボードの抽象基底クラス
│   ├── base_solver.py  # ソルバーの抽象基底クラス
│   └── base_vision.py  # 画像認識の抽象基底クラス
├── puzzles/            # 各パズルの実装
│   ├── arrow/          # 矢印パズル
│   └── fifteen/        # 15パズル
├── automation.py       # 自動化ユーティリティ
└── cli.py             # CLIインターフェース
```

### 新しいパズルの追加

新しいパズルタイプを追加するには：

1. `puzzles/`ディレクトリに新しいパズルのディレクトリを作成
2. `BaseBoard`、`BaseSolver`、`BaseVision`（必要に応じて）を継承して実装
3. `cli.py`に新しいコマンドグループを追加

## Automatic Screen Solving

矢印パズルは画面上のパズルを自動的に検出して解くことができます：

```bash
# インタラクティブモード - マウスで領域を選択
rye run python -m puzzle_solver arrow auto-solve

# 領域を指定 (x, y, width, height)
rye run python -m puzzle_solver arrow auto-solve --region 100 200 500 500

# 連続解法
rye run python -m puzzle_solver arrow auto-solve --continuous --max-puzzles 10
```

**⚠️ 重要：現在の画像認識実装は概念実証であり、実際のExponential Idleゲームプレイではテストされていません。** 詳細は[KNOWN_ISSUES.md](KNOWN_ISSUES.md)を参照してください。

## Development

### Running Tests

```bash
rye run pytest tests/ -v
```

### Code Quality

```bash
# Format code
rye run black src/ tests/

# Lint code
rye run ruff src/ tests/
```

## References

- [Exponential Idle Guide - Arrow Puzzles](https://exponential-idle-guides.netlify.app/guides/asd/)
- 矢印パズルのアルゴリズム実装は上記ガイドに記載された解法に基づいています

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.