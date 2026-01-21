"""
音声取得モジュールのテスト
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

# テスト対象のモジュール
from src.audio.wasapi_capture import WasapiCapture, AudioDevice
from src.audio.resampler import AudioResampler, PassthroughResampler, create_resampler


class TestWasapiCapture:
    """WasapiCapture クラスのテスト"""
    
    def test_init(self):
        """初期化テスト"""
        capture = WasapiCapture()
        assert capture._device_index is None
        assert capture._is_recording is False
        assert capture._stream is None
        
    def test_init_with_device_index(self):
        """デバイスインデックス指定での初期化テスト"""
        capture = WasapiCapture(device_index=1)
        assert capture._device_index == 1
        
    def test_target_format(self):
        """ターゲットフォーマットの確認"""
        capture = WasapiCapture()
        assert capture.TARGET_SAMPLE_RATE == 16000
        assert capture.TARGET_CHANNELS == 1
        
    def test_properties(self):
        """プロパティのテスト"""
        capture = WasapiCapture()
        assert capture.sample_rate == 16000
        assert capture.channels == 1
        assert capture.is_recording is False


class TestAudioResampler:
    """AudioResampler クラスのテスト"""
    
    def test_init(self):
        """初期化テスト"""
        resampler = AudioResampler(
            input_sample_rate=44100,
            input_channels=2,
            input_dtype=np.int16
        )
        assert resampler.input_sample_rate == 44100
        assert resampler.input_channels == 2
        assert resampler.input_dtype == np.int16
        
    def test_passthrough_mono_16khz(self):
        """16kHz/mono の場合変換不要"""
        resampler = AudioResampler(
            input_sample_rate=16000,
            input_channels=1,
            input_dtype=np.int16
        )
        
        # テストデータ作成
        samples = np.array([0, 1000, -1000, 2000, -2000], dtype=np.int16)
        input_data = samples.tobytes()
        
        # リサンプリング実行（入力と同じフォーマットなので変換なし）
        output_data = resampler.resample(input_data)
        output_samples = np.frombuffer(output_data, dtype=np.int16)
        
        np.testing.assert_array_equal(samples, output_samples)
        
    def test_stereo_to_mono(self):
        """ステレオからモノラルへの変換テスト"""
        resampler = AudioResampler(
            input_sample_rate=16000,
            input_channels=2,
            input_dtype=np.int16
        )
        
        # ステレオテストデータ (L, R, L, R, ...)
        stereo_samples = np.array([
            100, 200,   # L=100, R=200 -> avg=150
            300, 400,   # L=300, R=400 -> avg=350
            500, 600,   # L=500, R=600 -> avg=550
        ], dtype=np.int16)
        
        input_data = stereo_samples.tobytes()
        output_data = resampler.resample(input_data)
        output_samples = np.frombuffer(output_data, dtype=np.int16)
        
        expected = np.array([150, 350, 550], dtype=np.int16)
        np.testing.assert_array_equal(expected, output_samples)
        
    def test_resample_44100_to_16000(self):
        """44100Hz から 16000Hz へのリサンプリングテスト"""
        resampler = AudioResampler(
            input_sample_rate=44100,
            input_channels=1,
            input_dtype=np.int16
        )
        
        # 1秒分の 44100Hz サイン波
        t = np.linspace(0, 1, 44100, endpoint=False)
        frequency = 440  # 440Hz
        samples = (np.sin(2 * np.pi * frequency * t) * 16000).astype(np.int16)
        
        input_data = samples.tobytes()
        output_data = resampler.resample(input_data)
        output_samples = np.frombuffer(output_data, dtype=np.int16)
        
        # 16000 サンプル（1秒分）になっているはず
        assert len(output_samples) == 16000


class TestPassthroughResampler:
    """PassthroughResampler クラスのテスト"""
    
    def test_passthrough(self):
        """パススルー動作テスト"""
        resampler = PassthroughResampler()
        
        input_data = b'\x00\x01\x02\x03'
        output_data = resampler.resample(input_data)
        
        assert input_data == output_data


class TestCreateResampler:
    """create_resampler ファクトリ関数のテスト"""
    
    def test_create_passthrough_for_target_format(self):
        """ターゲットフォーマット時は PassthroughResampler を返す"""
        resampler = create_resampler(
            input_sample_rate=16000,
            input_channels=1,
            input_dtype=np.int16
        )
        assert isinstance(resampler, PassthroughResampler)
        
    def test_create_resampler_for_different_sample_rate(self):
        """異なるサンプルレート時は AudioResampler を返す"""
        resampler = create_resampler(
            input_sample_rate=44100,
            input_channels=1,
            input_dtype=np.int16
        )
        assert isinstance(resampler, AudioResampler)
        
    def test_create_resampler_for_stereo(self):
        """ステレオ時は AudioResampler を返す"""
        resampler = create_resampler(
            input_sample_rate=16000,
            input_channels=2,
            input_dtype=np.int16
        )
        assert isinstance(resampler, AudioResampler)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
