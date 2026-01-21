"""
ローカルファースト音声入力エージェント - メインエントリーポイント

F8 キーで録音を開始/停止し、音声認識結果をアクティブウィンドウに入力する。
"""

import sys
import logging
import signal
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
    """
    音声入力エージェント
    
    各コンポーネントを統合し、ホットキーによる音声入力を実現する。
    """
    
    def __init__(
        self,
        hotkey_config: Optional[HotkeyConfig] = None,
        use_silero_vad: bool = True,
        whisper_model_size: str = "base"
    ):
        """
        Args:
            hotkey_config: ホットキー設定（None の場合は F8）
            use_silero_vad: Silero VAD を使用するかどうか
            whisper_model_size: Whisper モデルサイズ
        """
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
                self._vad = SileroVAD()
                logger.info("Silero VAD を使用")
            except Exception as e:
                logger.warning(f"Silero VAD の初期化に失敗: {e}")
                logger.info("エネルギーベース VAD にフォールバック")
                self._vad = SimpleEnergyVAD()
        else:
            self._vad = SimpleEnergyVAD()
            logger.info("エネルギーベース VAD を使用")
            
        # STT
        self._stt = WhisperStreamProcessor(
            model_size=self.whisper_model_size,
            device="cpu",
            compute_type="int8",
            language="ja"
        )
        
        # テキスト注入
        self._injector = TextInjector()
        
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
        
        logger.info("コンポーネントの初期化完了")
        
    def _on_recording_start(self) -> None:
        """録音開始時のコールバック"""
        logger.info("🎤 録音開始")
        self._vad.reset()
        self._accumulator.clear()
        
        # 音声キャプチャ開始
        self._capture.start(callback=self._on_audio_chunk)
        
    def _on_recording_stop(self) -> None:
        """録音停止時のコールバック"""
        logger.info("🛑 録音停止")
        
        # 音声キャプチャ停止
        self._capture.stop()
        
        # 残りのバッファを処理
        audio_data = self._accumulator.flush()
        if audio_data:
            self._process_audio(audio_data)
            
    def _on_audio_chunk(self, audio_data: bytes) -> None:
        """
        音声チャンク受信時のコールバック
        
        Args:
            audio_data: 16kHz/mono/16bit PCM のバイト列
        """
        # VAD で音声区間を判定
        is_speech = self._vad.is_speech(audio_data)
        
        # アキュムレータに追加
        complete_audio = self._accumulator.add(audio_data, is_speech)
        
        if complete_audio:
            # 音声区間が完了したら処理
            self._process_audio(complete_audio)
            
    def _process_audio(self, audio_data: bytes) -> None:
        """
        音声データを処理（STT → テキスト注入）
        
        Args:
            audio_data: 音声データ
        """
        timer = LatencyTimer()
        timer.mark(MeasurementPoint.SPEECH_END)
        
        try:
            # STT
            timer.mark(MeasurementPoint.STT_START)
            result = self._stt.transcribe(audio_data)
            timer.mark(MeasurementPoint.STT_END)
            
            if result.text:
                logger.info(f"認識結果: {result.text}")
                
                # テキスト注入
                timer.mark(MeasurementPoint.INJECTION_START)
                injection_result = self._injector.inject(result.text)
                timer.mark(MeasurementPoint.INJECTION_END)
                
                if injection_result.success:
                    logger.info(f"✓ テキスト注入完了 ({len(result.text)} 文字)")
                else:
                    logger.warning(f"テキスト注入に失敗: {injection_result.failed_characters}")
                    
                # レイテンシ記録
                measurement = timer.get_measurement(text_length=len(result.text))
                self._latency_logger.log(measurement)
                
                # 目標チェック
                if not self._latency_logger.check_target(500.0):
                    logger.warning(f"⚠ レイテンシが目標 (500ms) を超過: {measurement.total_latency_ms:.2f}ms")
                    
        except Exception as e:
            logger.error(f"音声処理エラー: {e}")
            
    def run(self) -> None:
        """エージェントを実行"""
        self._init_components()
        
        logger.info(f"音声入力エージェント起動")
        logger.info(f"ホットキー: {self.hotkey_config}")
        logger.info(f"終了: Ctrl+C")
        
        self._running = True
        self._hotkey_manager.start()
        
        try:
            # シグナルハンドラ
            signal.signal(signal.SIGINT, self._signal_handler)
            
            # メインループ
            while self._running:
                signal.pause() if hasattr(signal, 'pause') else __import__('time').sleep(0.1)
                
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
            
    def _signal_handler(self, signum, frame):
        """シグナルハンドラ"""
        self._running = False
        
    def stop(self) -> None:
        """エージェントを停止"""
        logger.info("エージェントを停止中...")
        
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
                logger.info(f"セッション統計:")
                logger.info(f"  処理回数: {stats.count}")
                logger.info(f"  平均レイテンシ: {stats.mean_ms:.2f}ms")
                logger.info(f"  中央値: {stats.median_ms:.2f}ms")
                logger.info(f"  最小/最大: {stats.min_ms:.2f}ms / {stats.max_ms:.2f}ms")
                
        logger.info("エージェント停止完了")


def show_devices():
    """利用可能なデバイスを表示"""
    print("=" * 60)
    print("利用可能なマイクデバイス:")
    print("-" * 60)
    
    with WasapiCapture() as capture:
        devices = capture.list_devices()
        for device in devices:
            default_mark = " [DEFAULT]" if device.is_default else ""
            print(f"  [{device.index}] {device.name}{default_mark}")
            print(f"      チャンネル数: {device.channels}, サンプルレート: {device.sample_rate} Hz")
            
    print("=" * 60)


def main():
    """メイン関数"""
    print("=" * 60)
    print("ローカルファースト音声入力エージェント v0.1.0")
    print("=" * 60)
    print()
    
    import argparse
    parser = argparse.ArgumentParser(description="ローカルファースト音声入力エージェント")
    parser.add_argument("--list-devices", action="store_true", help="デバイス一覧を表示")
    parser.add_argument("--model", default="base", help="Whisper モデルサイズ (tiny, base, small, medium)")
    parser.add_argument("--no-silero", action="store_true", help="Silero VAD を使用しない")
    args = parser.parse_args()
    
    if args.list_devices:
        show_devices()
        return 0
        
    # エージェント起動
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
