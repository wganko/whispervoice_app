"""
グローバルホットキーモジュール

システム全体で有効なホットキーを登録し、
録音の開始/停止をトグル制御する。
"""

import ctypes
from ctypes import wintypes
import threading
from typing import Optional, Callable, Dict
from dataclasses import dataclass
from enum import IntEnum


# 仮想キーコード
class VK(IntEnum):
    """仮想キーコード定数"""
    LBUTTON = 0x01
    RBUTTON = 0x02
    CANCEL = 0x03
    BACK = 0x08
    TAB = 0x09
    RETURN = 0x0D
    SHIFT = 0x10
    CONTROL = 0x11
    MENU = 0x12  # Alt
    PAUSE = 0x13
    CAPITAL = 0x14
    ESCAPE = 0x1B
    SPACE = 0x20
    F1 = 0x70
    F2 = 0x71
    F3 = 0x72
    F4 = 0x73
    F5 = 0x74
    F6 = 0x75
    F7 = 0x76
    F8 = 0x77
    F9 = 0x78
    F10 = 0x79
    F11 = 0x7A
    F12 = 0x7B


# モディファイアフラグ
class MOD(IntEnum):
    """ホットキーモディファイアフラグ"""
    ALT = 0x0001
    CONTROL = 0x0002
    SHIFT = 0x0004
    WIN = 0x0008
    NOREPEAT = 0x4000


@dataclass
class HotkeyConfig:
    """ホットキー設定"""
    key: int  # 仮想キーコード
    modifiers: int  # モディファイアフラグの組み合わせ
    description: str = ""
    
    def __str__(self) -> str:
        parts = []
        if self.modifiers & MOD.WIN:
            parts.append("Win")
        if self.modifiers & MOD.CONTROL:
            parts.append("Ctrl")
        if self.modifiers & MOD.ALT:
            parts.append("Alt")
        if self.modifiers & MOD.SHIFT:
            parts.append("Shift")
            
        # キー名を取得
        key_name = self._get_key_name()
        parts.append(key_name)
        
        return "+".join(parts)
        
    def _get_key_name(self) -> str:
        """キー名を取得"""
        for vk in VK:
            if vk.value == self.key:
                return vk.name
        if 0x30 <= self.key <= 0x39:
            return chr(self.key)
        if 0x41 <= self.key <= 0x5A:
            return chr(self.key)
        return f"0x{self.key:02X}"


class GlobalHotkeyManager:
    """
    グローバルホットキーマネージャー
    
    Win32 RegisterHotKey API を使用してシステム全体で有効な
    ホットキーを登録・管理する。
    
    Usage:
        manager = GlobalHotkeyManager()
        manager.register(
            hotkey_id=1,
            config=HotkeyConfig(key=VK.F8, modifiers=0),
            callback=on_recording_toggle
        )
        manager.start()
        # ... アプリケーション実行中 ...
        manager.stop()
    """
    
    # Windows メッセージ
    WM_HOTKEY = 0x0312
    
    def __init__(self):
        self._hotkeys: Dict[int, tuple[HotkeyConfig, Callable]] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Windows API
        self._user32 = ctypes.windll.user32
        
    def register(
        self,
        hotkey_id: int,
        config: HotkeyConfig,
        callback: Callable[[], None]
    ) -> bool:
        """
        ホットキーを登録
        
        Args:
            hotkey_id: ホットキーの ID（一意である必要がある）
            config: ホットキー設定
            callback: ホットキー押下時のコールバック
            
        Returns:
            登録成功したかどうか
        """
        # メッセージループスレッドで登録する必要がある
        self._hotkeys[hotkey_id] = (config, callback)
        
        if self._running:
            # 実行中の場合は直接登録
            return self._register_hotkey(hotkey_id, config)
            
        return True
        
    def _register_hotkey(self, hotkey_id: int, config: HotkeyConfig) -> bool:
        """Windows API でホットキーを登録"""
        result = self._user32.RegisterHotKey(
            None,
            hotkey_id,
            config.modifiers,
            config.key
        )
        return result != 0
        
    def unregister(self, hotkey_id: int) -> bool:
        """
        ホットキーを解除
        
        Args:
            hotkey_id: ホットキーの ID
            
        Returns:
            解除成功したかどうか
        """
        if hotkey_id in self._hotkeys:
            del self._hotkeys[hotkey_id]
            
        if self._running:
            return self._unregister_hotkey(hotkey_id)
            
        return True
        
    def _unregister_hotkey(self, hotkey_id: int) -> bool:
        """Windows API でホットキーを解除"""
        result = self._user32.UnregisterHotKey(None, hotkey_id)
        return result != 0
        
    def start(self) -> None:
        """ホットキーリスナーを開始"""
        if self._running:
            return
            
        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(target=self._message_loop, daemon=True)
        self._thread.start()
        
    def stop(self) -> None:
        """ホットキーリスナーを停止"""
        if not self._running:
            return
            
        self._stop_event.set()
        
        # PostThreadMessage で WM_QUIT を送信してループを抜ける
        if self._thread:
            self._user32.PostThreadMessageW(
                self._thread.ident,
                0x0012,  # WM_QUIT
                0,
                0
            )
            self._thread.join(timeout=2.0)
            
        self._running = False
        
    def _message_loop(self) -> None:
        """Windows メッセージループ"""
        # すべてのホットキーを登録
        for hotkey_id, (config, _) in self._hotkeys.items():
            self._register_hotkey(hotkey_id, config)
            
        # メッセージ構造体
        msg = wintypes.MSG()
        
        while not self._stop_event.is_set():
            # PeekMessage で非ブロッキングでメッセージを取得
            result = self._user32.PeekMessageW(
                ctypes.byref(msg),
                None,
                0,
                0,
                0x0001  # PM_REMOVE
            )
            
            if result:
                if msg.message == self.WM_HOTKEY:
                    hotkey_id = msg.wParam
                    if hotkey_id in self._hotkeys:
                        _, callback = self._hotkeys[hotkey_id]
                        try:
                            callback()
                        except Exception as e:
                            print(f"Hotkey callback error: {e}")
                elif msg.message == 0x0012:  # WM_QUIT
                    break
            else:
                # メッセージがない場合は少し待機
                self._stop_event.wait(0.01)
                
        # すべてのホットキーを解除
        for hotkey_id in self._hotkeys:
            self._unregister_hotkey(hotkey_id)
            
    @property
    def is_running(self) -> bool:
        """実行中かどうか"""
        return self._running


class RecordingToggle:
    """
    録音トグル制御
    
    ホットキーによる録音開始/停止のトグル制御を行う。
    """
    
    def __init__(
        self,
        on_start: Optional[Callable[[], None]] = None,
        on_stop: Optional[Callable[[], None]] = None
    ):
        """
        Args:
            on_start: 録音開始時のコールバック
            on_stop: 録音停止時のコールバック
        """
        self.on_start = on_start
        self.on_stop = on_stop
        self._is_recording = False
        
    def toggle(self) -> bool:
        """
        録音状態をトグル
        
        Returns:
            トグル後の録音状態（True=録音中）
        """
        if self._is_recording:
            self._is_recording = False
            if self.on_stop:
                self.on_stop()
        else:
            self._is_recording = True
            if self.on_start:
                self.on_start()
                
        return self._is_recording
        
    @property
    def is_recording(self) -> bool:
        """録音中かどうか"""
        return self._is_recording
        
    def start(self) -> None:
        """録音を開始"""
        if not self._is_recording:
            self._is_recording = True
            if self.on_start:
                self.on_start()
                
    def stop(self) -> None:
        """録音を停止"""
        if self._is_recording:
            self._is_recording = False
            if self.on_stop:
                self.on_stop()


# デフォルトのホットキー設定
DEFAULT_HOTKEY = HotkeyConfig(
    key=VK.F8,
    modifiers=0,
    description="録音開始/停止"
)
