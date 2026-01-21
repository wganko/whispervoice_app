"""
ローカルファースト音声入力エージェント - メインエントリーポイント

F8 キーで録音を開始/停止し、音声認識結果をアクティブウィンドウに入力する。
"""

import sys
import logging
import time
import threading
import atexit
from typing import Optional

from src.audio import WasapiCapture
from src.vad import SileroVAD, SimpleEnergyVAD
from src.stt import WhisperStreamProcessor, AudioAccumulator
from src.input import TextInjector
from src.hotkey import GlobalHotkeyManager, RecordingToggle, HotkeyConfig, VK, DEFAULT_HOTKEY
from src.metrics import LatencyTimer, LatencyLogger, MeasurementPoint, get_latency_logger


# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


class VoiceInputAgent:
    """音声入力エージェント"""
    
    def __init__(
        self,
        hotkey_config: Optional[HotkeyConfig] = None,
        use_silero_vad: bool = True,
        whisper_model_size: str = "base"
    ):
        self.hotkey_config = hotkey_config or DEFAULT_HOTKEY
        self.use_silero_vad = use_silero_vad
        self.whisper_model_size = whisper_model_size
        
        # コンポーネント
        self._capture: Optional[WasapiCapture] = None
        self._vad = None
        self._stt: Optional[WhisperStreamProcessor] = None
        self._injector: Optional[TextInjector] = None
        self._accumulator: Optional[AudioAccumulator] = None
        self._hotkey_manager: Optional[GlobalHotkeyManager] = None
        self._recording_toggle: Optional[RecordingToggle] = None
        self._latency_logger: Optional[LatencyLogger] = None
        
        self._running = False
        
    def _init_components(self) -> None:
        """コンポーネントを初期化"""
        logger.info("コンポーネントを初期化中...")
        
        # 音声キャプチャ
        self._capture = WasapiCapture()
        
        # VAD
        if self.use_silero_vad:
            try:
                self._vad = SileroVAD(
                    min_speech_duration_ms=150,
                    min_silence_duration_ms=50
                )
                logger.info("Silero VAD を使用")
            except Exception as e:
                logger.warning(f"Silero VAD の初期化に失敗: {e}")
                self._vad = SimpleEnergyVAD(
                    min_speech_duration_ms=150,
                    min_silence_duration_ms=50
                )
        else:
            self._vad = SimpleEnergyVAD()
            logger.info("エネルギーベース VAD を使用")
            
        # STT
        self._stt = WhisperStreamProcessor(
            model_size=self.whisper_model_size,
            device="cpu",
            compute_type="int8",
            language="ja",
            beam_size=3
        )
        
        # モデルを事前ロード
        logger.info("音声認識モデルをプリロード中...")
        self._stt.preload()
        
        # テキスト注入
        self._injector = TextInjector(delay_between_chars_ms=5.0)
        
        # 音声アキュムレータ
        self._accumulator = AudioAccumulator()
        
        # レイテンシロガー
        self._latency_logger = get_latency_logger()
        
        # 録音トグル
        self._recording_toggle = RecordingToggle(
            on_start=self._on_recording_start,
            on_stop=self._on_recording_stop
        )
        
        # ホットキーマネージャー
        self._hotkey_manager = GlobalHotkeyManager()
        self._hotkey_manager.register(
            hotkey_id=1,
            config=self.hotkey_config,
            callback=self._recording_toggle.toggle
        )
        
        logger.info("初期化完了")
        
    def _on_recording_start(self) -> None:
        """録音開始"""
        logger.info("🎤 録音開始")
        self._vad.reset()
        self._accumulator.clear()
        self._capture.start(callback=self._on_audio_chunk)
        
    def _on_recording_stop(self) -> None:
        """録音停止"""
        logger.info("🛑 録音停止")
        self._capture.stop()
        
        # 残りのバッファを処理
        audio_data = self._accumulator.flush()
        if audio_data:
            self._process_audio(audio_data)
            
    def _on_audio_chunk(self, audio_data: bytes) -> None:
        """音声チャンク受信"""
        is_speech = self._vad.is_speech(audio_data)
        complete_audio = self._accumulator.add(audio_data, is_speech)
        
        if complete_audio:
            self._process_audio(complete_audio)
            
    def _process_audio(self, audio_data: bytes) -> None:
        """音声処理"""
        timer = LatencyTimer()
        timer.mark(MeasurementPoint.SPEECH_END)
        
        try:
            timer.mark(MeasurementPoint.STT_START)
            result = self._stt.transcribe(audio_data)
            timer.mark(MeasurementPoint.STT_END)
            
            if result.text:
                clean_text = result.text.strip()
                
                if self._is_valid_text(clean_text):
                    logger.info(f"認識結果: {clean_text}")
                    
                    time.sleep(0.05)
                    
                    timer.mark(MeasurementPoint.INJECTION_START)
                    injection_result = self._injector.inject_with_ime_workaround(clean_text)
                    timer.mark(MeasurementPoint.INJECTION_END)
                    
                    if injection_result.success:
                        logger.info(f"✓ 注入完了 ({len(clean_text)} 文字)")
                    
                    measurement = timer.get_measurement(text_length=len(clean_text))
                    self._latency_logger.log(measurement)
                    
                    if not self._latency_logger.check_target(500.0):
                        logger.warning(f"⚠ レイテンシ超過: {measurement.total_latency_ms:.0f}ms")
                        
        except Exception as e:
            logger.error(f"音声処理エラー: {e}")
            
    def _is_valid_text(self, text: str) -> bool:
        """有効なテキストか判定"""
        if not text or len(text) < 1:
            return False
            
        hallucination_patterns = [
            "ご視聴ありがとう", "チャンネル登録", "お願いします",
            "ご静聴", "♪", "..."
        ]
        
        for pattern in hallucination_patterns:
            if pattern in text:
                return False
        return True
            
    def run(self) -> None:
        """エージェント実行"""
        self._init_components()
        
        print()
        print("=" * 50)
        print(f"  ホットキー: {self.hotkey_config}")
        print(f"  終了: Ctrl+C を 2回 押す")
        print("=" * 50)
        print()
        
        self._running = True
        self._hotkey_manager.start()
        
        # 終了時のクリーンアップを登録
        atexit.register(self._cleanup)
        
        try:
            while self._running:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n終了中...")
        finally:
            self.stop()
            
    def _cleanup(self):
        """終了時クリーンアップ"""
        self.stop()
            
    def stop(self) -> None:
        """エージェント停止"""
        if not self._running:
            return
            
        self._running = False
        
        if self._recording_toggle and self._recording_toggle.is_recording:
            self._recording_toggle.stop()
            
        if self._hotkey_manager:
            self._hotkey_manager.stop()
            
        if self._capture:
            self._capture.stop()
            
        # 統計表示
        if self._latency_logger:
            stats = self._latency_logger.get_statistics()
            if stats.count > 0:
                print(f"\n--- 統計 ---")
                print(f"処理回数: {stats.count}")
                print(f"平均レイテンシ: {stats.mean_ms:.0f}ms")
                
        logger.info("停止完了")


def show_devices():
    """デバイス一覧表示"""
    print("利用可能なマイク:")
    with WasapiCapture() as capture:
        for device in capture.list_devices():
            mark = " [DEFAULT]" if device.is_default else ""
            print(f"  [{device.index}] {device.name}{mark}")


def main():
    """メイン"""
    print("=" * 50)
    print("ローカル音声入力エージェント v0.2.0")
    print("=" * 50)
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--model", default="base")
    parser.add_argument("--no-silero", action="store_true")
    args = parser.parse_args()
    
    if args.list_devices:
        show_devices()
        return 0
        
    agent = VoiceInputAgent(
        whisper_model_size=args.model,
        use_silero_vad=not args.no_silero
    )
    
    try:
        agent.run()
    except Exception as e:
        logger.error(f"エラー: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())
