# CLAUDE-JP.md

このファイルは、このリポジトリでコードを扱う際のClaude Code (claude.ai/code) への指針を提供します。

## プロジェクト概要

これは「exponential-idle-arrow-puzzle-solver」プロジェクトです - モバイルゲーム「Exponential Idle」に登場する矢印パズルを解くためのライブラリです。

## 現在の状態

プロジェクトは以下のコンポーネントで実装されています：
- 矢印パズルのボード表現
- Propagationアルゴリズムを実装したソルバー
- Hard/Expertモードの解法サポート
- パズルを解くためのCLIインターフェース
- pytestを使用したテストスイート

## 開発に関する注意事項

1. **言語**: Python（Ryeで管理）

2. **プロジェクト構造**:
   - `src/arrow_puzzle_solver/`: メインパッケージ
     - `board.py`: ボード表現
     - `solver.py`: 解法アルゴリズム
     - `cli.py`: コマンドラインインターフェース
   - `tests/`: テストスイート

3. **主要コマンド**:
   - 依存関係のインストール: `rye sync`
   - テストの実行: `rye run pytest tests/ -v`
   - CLIの実行: `rye run python -m arrow_puzzle_solver [command]`
   - デモ: `rye run python -m arrow_puzzle_solver demo`

4. **アルゴリズム実装**:
   - 行を順次解くPropagation手法
   - 高度なエンコーディング戦略を使用したHard/Expertモード
   - 解の正確性の検証

## 矢印パズルについて

Exponential Idleの矢印パズルは以下の特徴を持つ論理パズルです：
- ボードは0-4の値を持つセルで構成
- セルをタップすると、そのセルと上下左右の隣接セルの値が増加（5で割った余り）
- 目標は全てのセルを1にすること
- PropagationアルゴリズムとHard/Expertモード戦略を使用

## 重要な注意事項

- このファイル（CLAUDE-JP.md）とCLAUDE.mdの両方を常に同期して更新してください
- 新しい機能や変更を加える際は、両方のファイルに同じ情報を反映させてください