"""
faster-whisper ストリーミング音声認識モジュール

faster-whisper (CTranslate2 バックエンド) を使用した逐次文字起こし。
INT8 量子化モデルで高速かつ低リソースな推論を実現。
"""

import numpy as np
from typing import Optional, Iterator, List, Callable
from dataclasses import dataclass
from enum import Enum
import threading
import queue
import time


class TranscriptionState(Enum):
    """文字起こし状態"""
    IDLE = "idle"
    PROCESSING = "processing"
    COMPLETED = "completed"


@dataclass
class TranscriptionSegment:
    """文字起こしセグメント"""
    text: str
    start: float  # 開始時間（秒）
    end: float    # 終了時間（秒）
    confidence: float
    is_final: bool  # 確定かどうか


@dataclass
class TranscriptionResult:
    """文字起こし結果"""
    text: str
    segments: List[TranscriptionSegment]
    language: str
    language_probability: float
    processing_time_ms: float


class WhisperStreamProcessor:
    """
    faster-whisper を使用したストリーミング音声認識プロセッサ
    
    Usage:
        processor = WhisperStreamProcessor()
        for result in processor.transcribe_stream(audio_generator):
            print(result.text)
    """
    
    # 定数
    SAMPLE_RATE = 16000
    CHUNK_DURATION_S = 0.5  # チャンク長（秒）
    MIN_AUDIO_DURATION_S = 0.5  # 最小音声長（秒）
    
    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "ja",
        beam_size: int = 5
    ):
        """
        Args:
            model_size: モデルサイズ (tiny, base, small, medium, large-v2, large-v3)
            device: デバイス (cpu, cuda)
            compute_type: 計算精度 (int8, float16, float32)
            language: 言語コード
            beam_size: ビームサーチのビーム数
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        
        self._model = None
        self._state = TranscriptionState.IDLE
        
    def _load_model(self):
        """モデルをロード"""
        if self._model is None:
            from faster_whisper import WhisperModel
            import logging
            logger = logging.getLogger(__name__)
            
            logger.info(f"Whisper モデルをロード中: {self.model_size}")
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type
            )
            logger.info("Whisper モデルのロード完了")
            
    def preload(self) -> None:
        """モデルを事前ロード（起動時に呼び出す）"""
        self._load_model()
            
    def _bytes_to_float32(self, audio_bytes: bytes) -> np.ndarray:
        """
        int16 バイト列を float32 配列に変換
        
        Args:
            audio_bytes: 16bit PCM バイト列
            
        Returns:
            -1.0 から 1.0 の範囲の float32 配列
        """
        samples = np.frombuffer(audio_bytes, dtype=np.int16)
        return samples.astype(np.float32) / 32768.0
        
    def transcribe(self, audio_data: bytes) -> TranscriptionResult:
        """
        音声データを文字起こし
        
        Args:
            audio_data: 16kHz/mono/16bit PCM のバイト列
            
        Returns:
            文字起こし結果
        """
        self._load_model()
        self._state = TranscriptionState.PROCESSING
        
        start_time = time.perf_counter()
        
        # バイト列を float32 に変換
        audio_float = self._bytes_to_float32(audio_data)
        
        # 最小長チェック
        if len(audio_float) < self.SAMPLE_RATE * self.MIN_AUDIO_DURATION_S:
            self._state = TranscriptionState.IDLE
            return TranscriptionResult(
                text="",
                segments=[],
                language=self.language,
                language_probability=0.0,
                processing_time_ms=0.0
            )
            
        # 推論実行
        segments_gen, info = self._model.transcribe(
            audio_float,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=False,  # 外部 VAD を使用
            without_timestamps=False
        )
        
        # セグメントを収集
        segments = []
        full_text = []
        
        for segment in segments_gen:
            seg = TranscriptionSegment(
                text=segment.text.strip(),
                start=segment.start,
                end=segment.end,
                confidence=segment.avg_logprob,
                is_final=True
            )
            segments.append(seg)
            full_text.append(segment.text.strip())
            
        end_time = time.perf_counter()
        processing_time_ms = (end_time - start_time) * 1000
        
        self._state = TranscriptionState.COMPLETED
        
        return TranscriptionResult(
            text=" ".join(full_text),
            segments=segments,
            language=info.language,
            language_probability=info.language_probability,
            processing_time_ms=processing_time_ms
        )
        
    def transcribe_stream(
        self,
        audio_iterator: Iterator[bytes],
        on_intermediate: Optional[Callable[[str], None]] = None
    ) -> Iterator[TranscriptionResult]:
        """
        音声ストリームを逐次文字起こし
        
        Args:
            audio_iterator: 音声チャンクのイテレータ
            on_intermediate: 中間結果のコールバック
            
        Yields:
            文字起こし結果
        """
        self._load_model()
        
        buffer = b""
        chunk_size = int(self.SAMPLE_RATE * self.CHUNK_DURATION_S * 2)  # 16bit = 2 bytes
        
        for audio_chunk in audio_iterator:
            buffer += audio_chunk
            
            # チャンクサイズに達したら処理
            while len(buffer) >= chunk_size:
                chunk = buffer[:chunk_size]
                buffer = buffer[chunk_size:]
                
                result = self.transcribe(chunk)
                
                if result.text and on_intermediate:
                    on_intermediate(result.text)
                    
                if result.text:
                    yield result
                    
        # 残りのバッファを処理
        if len(buffer) > 0:
            result = self.transcribe(buffer)
            if result.text:
                yield result
                
    @property
    def state(self) -> TranscriptionState:
        """現在の状態"""
        return self._state
        
    @property
    def is_loaded(self) -> bool:
        """モデルがロード済みかどうか"""
        return self._model is not None


class AudioAccumulator:
    """
    音声データアキュムレータ
    
    VAD の結果に基づいて音声区間を蓄積し、
    無音が検出されたら蓄積した音声を返す。
    """
    
    def __init__(
        self,
        max_duration_s: float = 30.0,
        sample_rate: int = 16000
    ):
        """
        Args:
            max_duration_s: 最大蓄積時間（秒）
            sample_rate: サンプルレート
        """
        self.max_duration_s = max_duration_s
        self.sample_rate = sample_rate
        
        self._buffer: bytearray = bytearray()
        self._is_accumulating = False
        
    def add(self, audio_chunk: bytes, is_speech: bool) -> Optional[bytes]:
        """
        音声チャンクを追加
        
        Args:
            audio_chunk: 音声データ
            is_speech: VAD による音声判定
            
        Returns:
            蓄積完了した音声データ、または None
        """
        if is_speech:
            self._buffer.extend(audio_chunk)
            self._is_accumulating = True
            
            # 最大長チェック
            max_bytes = int(self.max_duration_s * self.sample_rate * 2)
            if len(self._buffer) >= max_bytes:
                result = bytes(self._buffer)
                self._buffer.clear()
                self._is_accumulating = False
                return result
                
        else:
            if self._is_accumulating and len(self._buffer) > 0:
                # 無音検出：蓄積した音声を返す
                result = bytes(self._buffer)
                self._buffer.clear()
                self._is_accumulating = False
                return result
                
        return None
        
    def flush(self) -> Optional[bytes]:
        """蓄積中のデータを強制的に返す"""
        if len(self._buffer) > 0:
            result = bytes(self._buffer)
            self._buffer.clear()
            self._is_accumulating = False
            return result
        return None
        
    def clear(self):
        """バッファをクリア"""
        self._buffer.clear()
        self._is_accumulating = False
        
    @property
    def is_accumulating(self) -> bool:
        """蓄積中かどうか"""
        return self._is_accumulating
        
    @property
    def duration_s(self) -> float:
        """蓄積中の音声の長さ（秒）"""
        return len(self._buffer) / (self.sample_rate * 2)
