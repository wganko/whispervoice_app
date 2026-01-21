"""
VAD モジュールのテスト
"""

import pytest
import numpy as np

from src.vad.silero_vad import (
    SileroVAD, SimpleEnergyVAD, VADResult, VoiceState
)


class TestVADResult:
    """VADResult データクラスのテスト"""
    
    def test_create_result(self):
        """結果オブジェクト作成テスト"""
        result = VADResult(
            is_speech=True,
            confidence=0.85,
            state=VoiceState.SPEECH
        )
        assert result.is_speech is True
        assert result.confidence == 0.85
        assert result.state == VoiceState.SPEECH


class TestVoiceState:
    """VoiceState Enum のテスト"""
    
    def test_states(self):
        """状態値テスト"""
        assert VoiceState.SILENCE.value == "silence"
        assert VoiceState.SPEECH.value == "speech"


class TestSimpleEnergyVAD:
    """SimpleEnergyVAD クラスのテスト（Silero 不要）"""
    
    def test_init(self):
        """初期化テスト"""
        vad = SimpleEnergyVAD()
        assert vad.threshold_db == -40.0
        assert vad.current_state == VoiceState.SILENCE
        
    def test_init_custom_params(self):
        """カスタムパラメータでの初期化テスト"""
        vad = SimpleEnergyVAD(
            threshold_db=-30.0,
            min_speech_duration_ms=100
        )
        assert vad.threshold_db == -30.0
        assert vad.min_speech_duration_ms == 100
        
    def test_silence_detection(self):
        """無音検出テスト"""
        vad = SimpleEnergyVAD(threshold_db=-40.0)
        
        # 無音データ（ほぼゼロ）
        silence = np.zeros(512, dtype=np.int16)
        result = vad.process(silence.tobytes())
        
        assert result.is_speech is False
        assert result.state == VoiceState.SILENCE
        
    def test_speech_detection(self):
        """音声検出テスト"""
        vad = SimpleEnergyVAD(
            threshold_db=-40.0,
            min_speech_duration_ms=0  # 即座に検出
        )
        
        # 大きな音（サイン波）
        t = np.linspace(0, 0.032, 512, endpoint=False)  # 32ms
        frequency = 440
        loud_sound = (np.sin(2 * np.pi * frequency * t) * 20000).astype(np.int16)
        
        result = vad.process(loud_sound.tobytes())
        
        assert result.is_speech is True
        assert result.state == VoiceState.SPEECH
        
    def test_reset(self):
        """リセットテスト"""
        vad = SimpleEnergyVAD(min_speech_duration_ms=0)
        
        # 音声を検出させる
        loud = (np.ones(512) * 10000).astype(np.int16)
        vad.process(loud.tobytes())
        
        # リセット
        vad.reset()
        
        assert vad.current_state == VoiceState.SILENCE
        
    def test_is_speech_interface(self):
        """簡易インターフェーステスト"""
        vad = SimpleEnergyVAD()
        
        silence = np.zeros(512, dtype=np.int16)
        assert vad.is_speech(silence.tobytes()) is False


class TestSileroVAD:
    """SileroVAD クラスのテスト（モデルロードなし）"""
    
    def test_init(self):
        """初期化テスト（モデルロードなし）"""
        vad = SileroVAD()
        assert vad.threshold == 0.5
        assert vad.current_state == VoiceState.SILENCE
        assert vad._model is None
        
    def test_init_custom_params(self):
        """カスタムパラメータでの初期化テスト"""
        vad = SileroVAD(
            threshold=0.7,
            min_speech_duration_ms=300
        )
        assert vad.threshold == 0.7
        assert vad.min_speech_duration_ms == 300
        
    def test_bytes_to_float(self):
        """バイト列変換テスト"""
        vad = SileroVAD()
        
        # int16 の最大値
        samples = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16)
        result = vad._bytes_to_float(samples.tobytes())
        
        expected = np.array([0.0, 0.5, -0.5, 32767/32768, -1.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected, decimal=4)
        
    def test_reset(self):
        """リセットテスト"""
        vad = SileroVAD()
        vad._state = VoiceState.SPEECH
        vad._speech_frames = 10
        
        vad.reset()
        
        assert vad._state == VoiceState.SILENCE
        assert vad._speech_frames == 0
        assert len(vad._buffer) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
