"""
オーディオリサンプラーモジュール

任意のサンプルレート/チャンネル数から 16kHz/mono/16bit PCM への変換を行う。
"""

import numpy as np
from typing import Optional
import resampy


class AudioResampler:
    """
    オーディオリサンプラー
    
    任意の入力フォーマットから 16kHz/mono/16bit PCM への変換を行う。
    """
    
    TARGET_SAMPLE_RATE = 16000
    TARGET_CHANNELS = 1
    TARGET_DTYPE = np.int16
    
    def __init__(
        self,
        input_sample_rate: int = 44100,
        input_channels: int = 2,
        input_dtype: np.dtype = np.int16
    ):
        """
        Args:
            input_sample_rate: 入力サンプルレート (Hz)
            input_channels: 入力チャンネル数
            input_dtype: 入力データ型
        """
        self.input_sample_rate = input_sample_rate
        self.input_channels = input_channels
        self.input_dtype = input_dtype
        
    def resample(self, audio_data: bytes) -> bytes:
        """
        音声データをリサンプリングする
        
        Args:
            audio_data: 入力音声データ（バイト列）
            
        Returns:
            16kHz/mono/16bit PCM のバイト列
        """
        # バイト列を numpy 配列に変換
        samples = np.frombuffer(audio_data, dtype=self.input_dtype)
        
        # ステレオの場合はモノラルに変換
        if self.input_channels > 1:
            samples = self._to_mono(samples)
            
        # サンプルレートが異なる場合はリサンプリング
        if self.input_sample_rate != self.TARGET_SAMPLE_RATE:
            samples = self._resample_audio(samples)
            
        # 出力データ型に変換
        if samples.dtype != self.TARGET_DTYPE:
            samples = self._convert_dtype(samples)
            
        return samples.tobytes()
    
    def _to_mono(self, samples: np.ndarray) -> np.ndarray:
        """
        マルチチャンネルをモノラルに変換
        
        Args:
            samples: インターリーブされた音声サンプル
            
        Returns:
            モノラル音声サンプル
        """
        # インターリーブされたサンプルをリシェイプ
        samples = samples.reshape(-1, self.input_channels)
        # チャンネルの平均を取る
        mono = samples.mean(axis=1)
        return mono.astype(self.input_dtype)
    
    def _resample_audio(self, samples: np.ndarray) -> np.ndarray:
        """
        サンプルレートを変換
        
        Args:
            samples: 入力サンプル
            
        Returns:
            リサンプリングされたサンプル
        """
        # float32 に変換（resampy の要件）
        samples_float = samples.astype(np.float32)
        
        # 正規化
        if self.input_dtype == np.int16:
            samples_float = samples_float / 32768.0
        elif self.input_dtype == np.int32:
            samples_float = samples_float / 2147483648.0
            
        # リサンプリング
        resampled = resampy.resample(
            samples_float,
            self.input_sample_rate,
            self.TARGET_SAMPLE_RATE,
            filter='kaiser_fast'
        )
        
        # int16 に戻す
        resampled = (resampled * 32767).clip(-32768, 32767).astype(np.int16)
        
        return resampled
    
    def _convert_dtype(self, samples: np.ndarray) -> np.ndarray:
        """
        データ型を変換
        
        Args:
            samples: 入力サンプル
            
        Returns:
            int16 に変換されたサンプル
        """
        if samples.dtype == np.float32 or samples.dtype == np.float64:
            return (samples * 32767).clip(-32768, 32767).astype(np.int16)
        elif samples.dtype == np.int32:
            return (samples / 65536).astype(np.int16)
        else:
            return samples.astype(np.int16)


class PassthroughResampler:
    """
    パススルーリサンプラー
    
    入力がすでに 16kHz/mono/16bit PCM の場合に使用する。
    変換処理をスキップして効率化。
    """
    
    TARGET_SAMPLE_RATE = 16000
    TARGET_CHANNELS = 1
    TARGET_DTYPE = np.int16
    
    def resample(self, audio_data: bytes) -> bytes:
        """変換なしでそのまま返す"""
        return audio_data


def create_resampler(
    input_sample_rate: int = 16000,
    input_channels: int = 1,
    input_dtype: np.dtype = np.int16
) -> AudioResampler | PassthroughResampler:
    """
    適切なリサンプラーを作成するファクトリ関数
    
    入力がすでに目標フォーマットの場合は PassthroughResampler を返す。
    
    Args:
        input_sample_rate: 入力サンプルレート
        input_channels: 入力チャンネル数
        input_dtype: 入力データ型
        
    Returns:
        リサンプラーインスタンス
    """
    if (input_sample_rate == 16000 and 
        input_channels == 1 and 
        input_dtype == np.int16):
        return PassthroughResampler()
    else:
        return AudioResampler(
            input_sample_rate=input_sample_rate,
            input_channels=input_channels,
            input_dtype=input_dtype
        )
