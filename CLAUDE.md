# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

MaAI（間合い）は、リアルタイム・連続的な非言語行動生成ソフトウェア。音声対話システムやロボット向けに、ターンテイキング予測・相槌予測・うなずき予測・VADを提供する。日本語・英語・中国語対応。CPUで動作可能。

## ビルド・インストール

```bash
# 依存関係のインストール
uv sync

# 新しいパッケージの追加
uv add <パッケージ名>

# パッケージビルド
hatch build
```

バージョン管理はhatch-vcsによりgitタグから自動生成。PyPI公開は`v*`タグのpushでGitHub Actionsが実行。

## トレーニング

```bash
cd train
uv add -r requirements-train.txt
uv run python train.py \
    --data_train_path sample/train.csv \
    --data_val_path sample/valid.csv \
    --data_test_path sample/test.csv \
    --vap_encoder_type cpc \
    --vap_cpc_model_pt ../asset/cpc/60k_epoch4-d0f474de.pt \
    --vap_freeze_encoder 1 \
    --opt_max_epochs 25 \
    --opt_saved_dir ./trained_data/

# 評価
uv run python evaluation.py --data_test_path sample/test.csv --checkpoint ./trained_data/
```

トレーニングはPyTorch Lightningベース。

## アーキテクチャ

### パイプライン

```
音声入力(Mic/Wav/TCP/Chunk) → CPCエンコーダ(2ch並列) → GPTトランスフォーマー → 確率出力 → 可視化/送信
```

- **入力** (`src/maai/input.py`): `Mic`, `Wav`, `TCPReceiver`, `Chunk`, `Zero` — Queueベースの非同期入力
- **エンコーダ** (`encoder.py`, `encoder_components.py`): Facebook Research CPCモデルで音声→特徴量変換（320倍ダウンサンプリング）
- **トランスフォーマー** (`modules.py`): `GPT`（単チャンネル）と`GPTStereo`（クロスチャンネル）。KVキャッシュによる効率的推論
- **メインクラス** (`model.py`): `Maai`クラスがスレッドベースのリアルタイム推論ループを管理
- **出力** (`output.py`): `ConsoleBar`, `GuiBar`, `GuiPlot`, `TCPTransmitter`

### モデルバリアント (`src/maai/models/`)

| ファイル | mode | 用途 |
|---------|------|------|
| `vap.py` | `vap`, `vap_mc` | ターンテイキング予測 |
| `vap_bc.py` | `bc` | 相槌タイミング予測 |
| `vap_bc_2type.py` | `bc_2type` | 反応型/感情型相槌 |
| `vap_nod.py` | `nod` | うなずき予測 |
| `vap_prompt.py` | `vap_prompt` | プロンプト条件付き（実験的） |

共通設定は`models/config.py`の`VapConfig`データクラス（dim=256, num_heads=4, context_limit等）。

### 音声パラメータ

- サンプリングレート: 16kHz固定
- フレームサイズ: 160サンプル（10ms）
- デフォルトコンテキスト: 20秒
- フレームレート: 10Hz（デフォルト）または50Hz

### モデル管理

事前学習モデルはHuggingFace Hub (`maai-kyoto`組織) から自動ダウンロード。`util.py`の`load_model_from_hf()`が管理。CPCモデルは`~/.cache/cpc/`にキャッシュ。

## 主な外部依存

- `torch>=2.2.0`, `einops>=0.7.0`: ニューラルネット
- `pyaudio`: マイク入力（システムにPortAudioが必要）
- `soundfile`: WAV読み書き
- `huggingface-hub`: モデルダウンロード
- `fastapi>=0.111.0`: TCP通信
- システム要件: `sox`（音声処理）

## 言語

コードのコメント・ドキュメントは日本語と英語が混在。README_JP.mdに日本語ドキュメントあり。`readme/`ディレクトリに各モデルの詳細ドキュメント（英語版と`_JP.md`日本語版）。
