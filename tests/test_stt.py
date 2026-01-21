"""
STT モジュールのテスト
"""

import pytest
import numpy as np

from src.stt.whisper_stream import (
    WhisperStreamProcessor,
    AudioAccumulator,
    TranscriptionResult,
    TranscriptionSegment,
    TranscriptionState
)


class TestTranscriptionResult:
    """TranscriptionResult データクラスのテスト"""
    
    def test_create_result(self):
        """結果オブジェクト作成テスト"""
        result = TranscriptionResult(
            text="こんにちは",
            segments=[],
            language="ja",
            language_probability=0.99,
            processing_time_ms=100.0
        )
        assert result.text == "こんにちは"
        assert result.language == "ja"


class TestTranscriptionSegment:
    """TranscriptionSegment データクラスのテスト"""
    
    def test_create_segment(self):
        """セグメント作成テスト"""
        segment = TranscriptionSegment(
            text="テスト",
            start=0.0,
            end=1.0,
            confidence=0.95,
            is_final=True
        )
        assert segment.text == "テスト"
        assert segment.start == 0.0
        assert segment.end == 1.0


class TestWhisperStreamProcessor:
    """WhisperStreamProcessor クラスのテスト（モデルロードなし）"""
    
    def test_init(self):
        """初期化テスト（モデルロードなし）"""
        processor = WhisperStreamProcessor()
        assert processor.model_size == "base"
        assert processor.device == "cpu"
        assert processor.compute_type == "int8"
        assert processor.language == "ja"
        assert processor.state == TranscriptionState.IDLE
        assert processor.is_loaded is False
        
    def test_init_custom_params(self):
        """カスタムパラメータでの初期化テスト"""
        processor = WhisperStreamProcessor(
            model_size="small",
            language="en",
            beam_size=3
        )
        assert processor.model_size == "small"
        assert processor.language == "en"
        assert processor.beam_size == 3
        
    def test_bytes_to_float32(self):
        """バイト列変換テスト"""
        processor = WhisperStreamProcessor()
        
        samples = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16)
        result = processor._bytes_to_float32(samples.tobytes())
        
        expected = np.array([0.0, 0.5, -0.5, 32767/32768, -1.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected, decimal=4)


class TestAudioAccumulator:
    """AudioAccumulator クラスのテスト"""
    
    def test_init(self):
        """初期化テスト"""
        acc = AudioAccumulator()
        assert acc.max_duration_s == 30.0
        assert acc.is_accumulating is False
        assert acc.duration_s == 0.0
        
    def test_accumulate_speech(self):
        """音声蓄積テスト"""
        acc = AudioAccumulator()
        
        # 音声チャンクを追加
        chunk = b"\x00\x01" * 512  # 512 サンプル
        result = acc.add(chunk, is_speech=True)
        
        assert result is None  # まだ返さない
        assert acc.is_accumulating is True
        assert acc.duration_s > 0
        
    def test_return_on_silence(self):
        """無音検出時に返却するテスト"""
        acc = AudioAccumulator()
        
        # 音声を蓄積
        chunk = b"\x00\x01" * 512
        acc.add(chunk, is_speech=True)
        
        # 無音を検出
        result = acc.add(b"\x00\x00" * 512, is_speech=False)
        
        assert result is not None
        assert len(result) == len(chunk)
        assert acc.is_accumulating is False
        
    def test_no_return_without_accumulation(self):
        """蓄積前の無音では返却しないテスト"""
        acc = AudioAccumulator()
        
        result = acc.add(b"\x00\x00" * 512, is_speech=False)
        
        assert result is None
        assert acc.is_accumulating is False
        
    def test_flush(self):
        """強制フラッシュテスト"""
        acc = AudioAccumulator()
        
        chunk = b"\x00\x01" * 512
        acc.add(chunk, is_speech=True)
        
        result = acc.flush()
        
        assert result is not None
        assert len(result) == len(chunk)
        assert acc.is_accumulating is False
        
    def test_flush_empty(self):
        """空のフラッシュテスト"""
        acc = AudioAccumulator()
        
        result = acc.flush()
        
        assert result is None
        
    def test_clear(self):
        """クリアテスト"""
        acc = AudioAccumulator()
        
        acc.add(b"\x00\x01" * 512, is_speech=True)
        acc.clear()
        
        assert acc.is_accumulating is False
        assert acc.duration_s == 0.0
        
    def test_max_duration(self):
        """最大長制限テスト"""
        acc = AudioAccumulator(max_duration_s=0.1, sample_rate=16000)
        
        # 0.1秒 = 1600 サンプル = 3200 バイト
        chunk = b"\x00\x01" * 1600
        result = acc.add(chunk, is_speech=True)
        
        assert result is not None
        assert len(result) == len(chunk)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
