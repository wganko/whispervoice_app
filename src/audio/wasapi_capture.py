"""
WASAPI マイク入力モジュール

Windows Audio Session API (WASAPI) を使用してマイクから音声データを取得する。
"""

import threading
import queue
from typing import Optional, Callable, List, Dict, Any
from dataclasses import dataclass

import pyaudiowpatch as pyaudio
import numpy as np


@dataclass
class AudioDevice:
    """オーディオデバイス情報"""
    index: int
    name: str
    channels: int
    sample_rate: int
    is_default: bool


class WasapiCapture:
    """
    WASAPI を使用したマイク入力キャプチャ
    
    Usage:
        capture = WasapiCapture()
        capture.list_devices()
        capture.start(callback=lambda data: print(len(data)))
        # ... 録音中 ...
        capture.stop()
    """
    
    # 定数
    TARGET_SAMPLE_RATE = 16000
    TARGET_CHANNELS = 1
    TARGET_FORMAT = pyaudio.paInt16
    CHUNK_SIZE = 1024  # サンプル数
    
    def __init__(self, device_index: Optional[int] = None):
        """
        Args:
            device_index: 使用するデバイスのインデックス。None の場合はデフォルトデバイスを使用。
        """
        self._pa: Optional[pyaudio.PyAudio] = None
        self._stream: Optional[pyaudio.Stream] = None
        self._device_index = device_index
        self._is_recording = False
        self._audio_queue: queue.Queue[bytes] = queue.Queue()
        self._callback: Optional[Callable[[bytes], None]] = None
        self._lock = threading.Lock()
        
    def __enter__(self):
        self._init_pyaudio()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        self._cleanup_pyaudio()
        
    def _init_pyaudio(self):
        """PyAudio インスタンスを初期化"""
        if self._pa is None:
            self._pa = pyaudio.PyAudio()
            
    def _cleanup_pyaudio(self):
        """PyAudio インスタンスをクリーンアップ"""
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None
            
    def list_devices(self) -> List[AudioDevice]:
        """
        利用可能な入力デバイスを列挙する
        
        Returns:
            入力デバイスのリスト
        """
        self._init_pyaudio()
        devices = []
        
        try:
            default_device = self._pa.get_default_input_device_info()
            default_index = default_device.get("index", -1)
        except IOError:
            default_index = -1
        
        for i in range(self._pa.get_device_count()):
            try:
                info = self._pa.get_device_info_by_index(i)
                # 入力デバイスのみをリストアップ
                if info.get("maxInputChannels", 0) > 0:
                    device = AudioDevice(
                        index=i,
                        name=info.get("name", "Unknown"),
                        channels=info.get("maxInputChannels", 0),
                        sample_rate=int(info.get("defaultSampleRate", 44100)),
                        is_default=(i == default_index)
                    )
                    devices.append(device)
            except IOError:
                continue
                
        return devices
    
    def get_device_info(self, device_index: Optional[int] = None) -> Dict[str, Any]:
        """
        デバイス情報を取得する
        
        Args:
            device_index: デバイスインデックス。None の場合はデフォルトデバイス。
            
        Returns:
            デバイス情報の辞書
        """
        self._init_pyaudio()
        
        if device_index is None:
            return self._pa.get_default_input_device_info()
        else:
            return self._pa.get_device_info_by_index(device_index)
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio コールバック関数"""
        if self._is_recording:
            if self._callback:
                self._callback(in_data)
            else:
                self._audio_queue.put(in_data)
        return (None, pyaudio.paContinue)
    
    def start(self, callback: Optional[Callable[[bytes], None]] = None) -> None:
        """
        録音を開始する
        
        Args:
            callback: 音声データを受け取るコールバック関数。
                      None の場合は内部キューにデータを蓄積。
        """
        with self._lock:
            if self._is_recording:
                return
                
            self._init_pyaudio()
            self._callback = callback
            
            # デバイス情報を取得
            device_index = self._device_index
            if device_index is None:
                device_info = self._pa.get_default_input_device_info()
                device_index = device_info["index"]
            else:
                device_info = self._pa.get_device_info_by_index(device_index)
            
            # ストリームを開く
            self._stream = self._pa.open(
                format=self.TARGET_FORMAT,
                channels=self.TARGET_CHANNELS,
                rate=self.TARGET_SAMPLE_RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.CHUNK_SIZE,
                stream_callback=self._audio_callback
            )
            
            self._is_recording = True
            self._stream.start_stream()
            
    def stop(self) -> None:
        """録音を停止する"""
        with self._lock:
            self._is_recording = False
            
            if self._stream is not None:
                self._stream.stop_stream()
                self._stream.close()
                self._stream = None
                
            self._callback = None
            
    def read(self, timeout: Optional[float] = None) -> Optional[bytes]:
        """
        キューから音声データを読み取る
        
        Args:
            timeout: タイムアウト秒数。None の場合はブロック。
            
        Returns:
            音声データ。タイムアウト時は None。
        """
        try:
            return self._audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def clear_queue(self) -> None:
        """キューをクリアする"""
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break
                
    @property
    def is_recording(self) -> bool:
        """録音中かどうか"""
        return self._is_recording
        
    @property
    def sample_rate(self) -> int:
        """出力サンプルレート"""
        return self.TARGET_SAMPLE_RATE
        
    @property
    def channels(self) -> int:
        """出力チャンネル数"""
        return self.TARGET_CHANNELS


def print_devices():
    """利用可能なデバイスを表示するユーティリティ関数"""
    with WasapiCapture() as capture:
        devices = capture.list_devices()
        print("利用可能な入力デバイス:")
        print("-" * 60)
        for device in devices:
            default_mark = " [DEFAULT]" if device.is_default else ""
            print(f"  [{device.index}] {device.name}{default_mark}")
            print(f"      チャンネル数: {device.channels}, サンプルレート: {device.sample_rate} Hz")
        print("-" * 60)


if __name__ == "__main__":
    print_devices()
