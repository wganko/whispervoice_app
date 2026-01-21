"""
エントリーポイント

メインアプリケーションの起動処理。
"""

import sys
from src.audio import WasapiCapture


def main():
    """メイン関数"""
    print("=" * 60)
    print("ローカルファースト音声入力エージェント v0.1.0")
    print("=" * 60)
    print()
    
    # デバイス一覧を表示
    print("利用可能なマイクデバイス:")
    with WasapiCapture() as capture:
        devices = capture.list_devices()
        for device in devices:
            default_mark = " [DEFAULT]" if device.is_default else ""
            print(f"  [{device.index}] {device.name}{default_mark}")
    print()
    
    print("※ 完全な機能は後続の Story で実装されます")
    return 0


if __name__ == "__main__":
    sys.exit(main())
