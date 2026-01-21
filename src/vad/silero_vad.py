"""
Silero VAD モジュール

Silero Voice Activity Detection を使用して音声区間を検出する。
無音区間を除外することで推論負荷を削減し、幻覚を防止する。
"""

import torch
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum


class VoiceState(Enum):
    """音声状態"""
    SILENCE = "silence"
    SPEECH = "speech"


@dataclass
class VADResult:
    """VAD 処理結果"""
    is_speech: bool
    confidence: float
    state: VoiceState


class SileroVAD:
    """
    Silero VAD ラッパー
    
    Usage:
        vad = SileroVAD()
        result = vad.process(audio_chunk)
        if result.is_speech:
            # 音声区間として処理
    """
    
    # Silero VAD の要件
    SAMPLE_RATE = 16000
    CHUNK_SAMPLES = 512  # Silero VAD は 512 サンプル (32ms @ 16kHz) を要求
    
    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100
    ):
        """
        Args:
            threshold: 音声判定の閾値 (0.0-1.0)
            min_speech_duration_ms: 音声開始と判定するまでの最小持続時間 (ms)
            min_silence_duration_ms: 無音と判定するまでの最小持続時間 (ms)
        """
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        
        self._model: Optional[torch.nn.Module] = None
        self._state: VoiceState = VoiceState.SILENCE
        self._speech_frames: int = 0
        self._silence_frames: int = 0
        
        # バッファ（512 サンプル未満のチャンクを蓄積）
        self._buffer: np.ndarray = np.array([], dtype=np.float32)
        
    def _load_model(self) -> None:
        """Silero VAD モデルをロード"""
        if self._model is None:
            self._model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                trust_repo=True
            )
            self._model.eval()
            
    def reset(self) -> None:
        """内部状態をリセット"""
        self._state = VoiceState.SILENCE
        self._speech_frames = 0
        self._silence_frames = 0
        self._buffer = np.array([], dtype=np.float32)
        if self._model is not None:
            self._model.reset_states()
            
    def _bytes_to_float(self, audio_bytes: bytes) -> np.ndarray:
        """
        int16 バイト列を float32 配列に変換
        
        Args:
            audio_bytes: 16bit PCM バイト列
            
        Returns:
            -1.0 から 1.0 の範囲の float32 配列
        """
        samples = np.frombuffer(audio_bytes, dtype=np.int16)
        return samples.astype(np.float32) / 32768.0
        
    def process(self, audio_chunk: bytes) -> List[VADResult]:
        """
        音声チャンクを処理して VAD 結果を返す
        
        Args:
            audio_chunk: 16kHz/mono/16bit PCM のバイト列
            
        Returns:
            VAD 結果のリスト（512 サンプルごとに1つ）
        """
        self._load_model()
        
        # バイト列を float32 に変換
        samples = self._bytes_to_float(audio_chunk)
        
        # バッファに追加
        self._buffer = np.concatenate([self._buffer, samples])
        
        results = []
        
        # 512 サンプルずつ処理
        while len(self._buffer) >= self.CHUNK_SAMPLES:
            chunk = self._buffer[:self.CHUNK_SAMPLES]
            self._buffer = self._buffer[self.CHUNK_SAMPLES:]
            
            result = self._process_chunk(chunk)
            results.append(result)
            
        return results
        
    def _process_chunk(self, samples: np.ndarray) -> VADResult:
        """
        512 サンプルのチャンクを処理
        
        Args:
            samples: float32 配列 (長さ 512)
            
        Returns:
            VAD 結果
        """
        # PyTorch テンソルに変換
        tensor = torch.from_numpy(samples)
        
        # 推論
        with torch.no_grad():
            confidence = self._model(tensor, self.SAMPLE_RATE).item()
            
        # 閾値判定
        is_speech_frame = confidence >= self.threshold
        
        # 状態遷移ロジック
        if is_speech_frame:
            self._speech_frames += 1
            self._silence_frames = 0
        else:
            self._silence_frames += 1
            self._speech_frames = 0
            
        # フレーム数から持続時間を計算
        frame_duration_ms = (self.CHUNK_SAMPLES / self.SAMPLE_RATE) * 1000
        speech_duration_ms = self._speech_frames * frame_duration_ms
        silence_duration_ms = self._silence_frames * frame_duration_ms
        
        # 状態更新
        if self._state == VoiceState.SILENCE:
            if speech_duration_ms >= self.min_speech_duration_ms:
                self._state = VoiceState.SPEECH
        else:  # VoiceState.SPEECH
            if silence_duration_ms >= self.min_silence_duration_ms:
                self._state = VoiceState.SILENCE
                
        return VADResult(
            is_speech=(self._state == VoiceState.SPEECH),
            confidence=confidence,
            state=self._state
        )
        
    def is_speech(self, audio_chunk: bytes) -> bool:
        """
        簡易インターフェース: 音声区間かどうかを判定
        
        Args:
            audio_chunk: 16kHz/mono/16bit PCM のバイト列
            
        Returns:
            True if 音声区間
        """
        results = self.process(audio_chunk)
        if not results:
            return self._state == VoiceState.SPEECH
        return results[-1].is_speech
        
    @property
    def current_state(self) -> VoiceState:
        """現在の状態"""
        return self._state


class SimpleEnergyVAD:
    """
    シンプルなエネルギーベース VAD
    
    Silero VAD が利用できない場合のフォールバック。
    RMS エネルギーに基づいて音声区間を判定。
    """
    
    def __init__(
        self,
        threshold_db: float = -40.0,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        sample_rate: int = 16000
    ):
        """
        Args:
            threshold_db: 音声判定の閾値 (dB)
            min_speech_duration_ms: 音声開始と判定するまでの最小持続時間
            min_silence_duration_ms: 無音と判定するまでの最小持続時間
            sample_rate: サンプルレート
        """
        self.threshold_db = threshold_db
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.sample_rate = sample_rate
        
        self._state = VoiceState.SILENCE
        self._speech_samples = 0
        self._silence_samples = 0
        
    def reset(self) -> None:
        """内部状態をリセット"""
        self._state = VoiceState.SILENCE
        self._speech_samples = 0
        self._silence_samples = 0
        
    def process(self, audio_chunk: bytes) -> VADResult:
        """
        音声チャンクを処理
        
        Args:
            audio_chunk: 16kHz/mono/16bit PCM のバイト列
            
        Returns:
            VAD 結果
        """
        samples = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
        
        # RMS 計算
        rms = np.sqrt(np.mean(samples ** 2)) / 32768.0
        db = 20 * np.log10(max(rms, 1e-10))
        
        # 閾値判定
        is_speech_frame = db >= self.threshold_db
        
        chunk_samples = len(samples)
        
        if is_speech_frame:
            self._speech_samples += chunk_samples
            self._silence_samples = 0
        else:
            self._silence_samples += chunk_samples
            self._speech_samples = 0
            
        # 持続時間計算
        speech_duration_ms = (self._speech_samples / self.sample_rate) * 1000
        silence_duration_ms = (self._silence_samples / self.sample_rate) * 1000
        
        # 状態更新
        if self._state == VoiceState.SILENCE:
            if speech_duration_ms >= self.min_speech_duration_ms:
                self._state = VoiceState.SPEECH
        else:
            if silence_duration_ms >= self.min_silence_duration_ms:
                self._state = VoiceState.SILENCE
                
        # 信頼度は正規化された RMS を使用
        confidence = min(1.0, max(0.0, (db + 60) / 60))
        
        return VADResult(
            is_speech=(self._state == VoiceState.SPEECH),
            confidence=confidence,
            state=self._state
        )
        
    def is_speech(self, audio_chunk: bytes) -> bool:
        """簡易インターフェース"""
        return self.process(audio_chunk).is_speech
        
    @property
    def current_state(self) -> VoiceState:
        """現在の状態"""
        return self._state
