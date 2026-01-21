# 次世代ローカルファースト音声入力エージェント

完全ローカルで動作する高速音声入力システム。

## 特徴

- **低レイテンシ**: 発話終了からテキスト注入まで ≤ 500ms
- **高精度**: WER < 5%
- **軽量**: アイドル時 CPU < 2%、RAM < 500MB
- **プライバシー保護**: 音声データは完全ローカル処理

## 技術スタック

- **音声取得**: WASAPI (pyaudiowpatch)
- **VAD**: Silero VAD
- **音声認識**: faster-whisper (INT8)
- **テキスト注入**: Win32 SendInput

## セットアップ

```bash
# 仮想環境作成
python -m venv venv
venv\Scripts\activate

# 依存関係インストール
pip install -r requirements.txt
```

## 使用方法

```bash
python -m src.main
```

## ライセンス

MIT License
