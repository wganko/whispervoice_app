"""
グローバルホットキーモジュール

pynput を使用してシステム全体で有効なホットキーを登録し、
録音の開始/停止をトグル制御する。
"""

import threading
from typing import Optional, Callable, Set
from dataclasses import dataclass
from enum import Enum
import logging

from pynput import keyboard

logger = logging.getLogger(__name__)


class KeyCode(Enum):
    """ホットキー用キーコード"""
    F1 = keyboard.Key.f1
    F2 = keyboard.Key.f2
    F3 = keyboard.Key.f3
    F4 = keyboard.Key.f4
    F5 = keyboard.Key.f5
    F6 = keyboard.Key.f6
    F7 = keyboard.Key.f7
    F8 = keyboard.Key.f8
    F9 = keyboard.Key.f9
    F10 = keyboard.Key.f10
    F11 = keyboard.Key.f11
    F12 = keyboard.Key.f12
    SPACE = keyboard.Key.space
    ESCAPE = keyboard.Key.esc


# 後方互換性のための仮想キーコード
class VK:
    """仮想キーコード定数（後方互換性）"""
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


class MOD:
    """モディファイアフラグ（後方互換性）"""
    ALT = 0x0001
    CONTROL = 0x0002
    SHIFT = 0x0004
    WIN = 0x0008
    NOREPEAT = 0x4000


@dataclass
class HotkeyConfig:
    """ホットキー設定"""
    key: int  # 仮想キーコード（VK.F8 など）
    modifiers: int = 0  # モディファイアフラグの組み合わせ
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
        key_names = {
            VK.F1: "F1", VK.F2: "F2", VK.F3: "F3", VK.F4: "F4",
            VK.F5: "F5", VK.F6: "F6", VK.F7: "F7", VK.F8: "F8",
            VK.F9: "F9", VK.F10: "F10", VK.F11: "F11", VK.F12: "F12",
        }
        return key_names.get(self.key, f"0x{self.key:02X}")
    
    def to_pynput_key(self) -> keyboard.Key:
        """pynput のキーに変換"""
        mapping = {
            VK.F1: keyboard.Key.f1,
            VK.F2: keyboard.Key.f2,
            VK.F3: keyboard.Key.f3,
            VK.F4: keyboard.Key.f4,
            VK.F5: keyboard.Key.f5,
            VK.F6: keyboard.Key.f6,
            VK.F7: keyboard.Key.f7,
            VK.F8: keyboard.Key.f8,
            VK.F9: keyboard.Key.f9,
            VK.F10: keyboard.Key.f10,
            VK.F11: keyboard.Key.f11,
            VK.F12: keyboard.Key.f12,
        }
        return mapping.get(self.key, keyboard.Key.f8)


class GlobalHotkeyManager:
    """
    グローバルホットキーマネージャー（pynput ベース）
    
    pynput を使用してシステム全体で有効なホットキーを登録・管理する。
    """
    
    def __init__(self):
        self._hotkeys: dict[int, tuple[HotkeyConfig, Callable]] = {}
        self._running = False
        self._listener: Optional[keyboard.Listener] = None
        self._pressed_modifiers: Set[keyboard.Key] = set()
        
    def register(
        self,
        hotkey_id: int,
        config: HotkeyConfig,
        callback: Callable[[], None]
    ) -> bool:
        """ホットキーを登録"""
        self._hotkeys[hotkey_id] = (config, callback)
        logger.info(f"ホットキー登録: ID={hotkey_id}, キー={config}")
        return True
        
    def unregister(self, hotkey_id: int) -> bool:
        """ホットキーを解除"""
        if hotkey_id in self._hotkeys:
            del self._hotkeys[hotkey_id]
            return True
        return False
        
    def _on_press(self, key) -> None:
        """キー押下時のコールバック"""
        # モディファイアキーを追跡
        if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
            self._pressed_modifiers.add(keyboard.Key.ctrl_l)
        elif key in (keyboard.Key.alt_l, keyboard.Key.alt_r):
            self._pressed_modifiers.add(keyboard.Key.alt_l)
        elif key in (keyboard.Key.shift_l, keyboard.Key.shift_r):
            self._pressed_modifiers.add(keyboard.Key.shift_l)
        elif key in (keyboard.Key.cmd_l, keyboard.Key.cmd_r):
            self._pressed_modifiers.add(keyboard.Key.cmd_l)
            
        # 登録されたホットキーをチェック
        for hotkey_id, (config, callback) in self._hotkeys.items():
            target_key = config.to_pynput_key()
            
            # キーが一致するかチェック
            if key == target_key:
                # モディファイアをチェック
                ctrl_required = config.modifiers & MOD.CONTROL
                alt_required = config.modifiers & MOD.ALT
                shift_required = config.modifiers & MOD.SHIFT
                win_required = config.modifiers & MOD.WIN
                
                ctrl_pressed = keyboard.Key.ctrl_l in self._pressed_modifiers
                alt_pressed = keyboard.Key.alt_l in self._pressed_modifiers
                shift_pressed = keyboard.Key.shift_l in self._pressed_modifiers
                win_pressed = keyboard.Key.cmd_l in self._pressed_modifiers
                
                # モディファイアが0の場合は単独キー
                if config.modifiers == 0:
                    # モディファイアなしの場合
                    logger.info(f"ホットキー検出: {config}")
                    try:
                        callback()
                    except Exception as e:
                        logger.error(f"ホットキーコールバックエラー: {e}")
                elif (bool(ctrl_required) == ctrl_pressed and
                      bool(alt_required) == alt_pressed and
                      bool(shift_required) == shift_pressed and
                      bool(win_required) == win_pressed):
                    logger.info(f"ホットキー検出: {config}")
                    try:
                        callback()
                    except Exception as e:
                        logger.error(f"ホットキーコールバックエラー: {e}")
                        
    def _on_release(self, key) -> None:
        """キー解放時のコールバック"""
        # モディファイアキーを追跡から削除
        if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
            self._pressed_modifiers.discard(keyboard.Key.ctrl_l)
        elif key in (keyboard.Key.alt_l, keyboard.Key.alt_r):
            self._pressed_modifiers.discard(keyboard.Key.alt_l)
        elif key in (keyboard.Key.shift_l, keyboard.Key.shift_r):
            self._pressed_modifiers.discard(keyboard.Key.shift_l)
        elif key in (keyboard.Key.cmd_l, keyboard.Key.cmd_r):
            self._pressed_modifiers.discard(keyboard.Key.cmd_l)
        
    def start(self) -> None:
        """ホットキーリスナーを開始"""
        if self._running:
            return
            
        self._running = True
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        self._listener.start()
        logger.info("ホットキーリスナー開始")
        
    def stop(self) -> None:
        """ホットキーリスナーを停止"""
        if not self._running:
            return
            
        self._running = False
        if self._listener:
            self._listener.stop()
            self._listener = None
        logger.info("ホットキーリスナー停止")
            
    @property
    def is_running(self) -> bool:
        """実行中かどうか"""
        return self._running


import time

class RecordingToggle:
    """録音トグル制御"""
    
    def __init__(
        self,
        on_start: Optional[Callable[[], None]] = None,
        on_stop: Optional[Callable[[], None]] = None,
        debounce_ms: int = 500  # デバウンス時間（ミリ秒）
    ):
        self.on_start = on_start
        self.on_stop = on_stop
        self._is_recording = False
        self._last_toggle_time = 0.0
        self._debounce_ms = debounce_ms
        
    def _check_debounce(self) -> bool:
        """デバウンスチェック"""
        current_time = time.time() * 1000
        if current_time - self._last_toggle_time < self._debounce_ms:
            return False
        self._last_toggle_time = current_time
        return True
        
    def toggle(self) -> bool:
        """録音状態をトグル"""
        if not self._check_debounce():
            return self._is_recording
            
        if self._is_recording:
            self._is_recording = False
            if self.on_stop:
                try:
                    self.on_stop()
                except Exception as e:
                    logging.getLogger(__name__).error(f"録音停止エラー: {e}")
        else:
            self._is_recording = True
            if self.on_start:
                try:
                    self.on_start()
                except Exception as e:
                    logging.getLogger(__name__).error(f"録音開始エラー: {e}")
                    # エラー時は録音状態を戻す
                    self._is_recording = False
                    
        return self._is_recording
        
    @property
    def is_recording(self) -> bool:
        return self._is_recording
        
    def start(self) -> None:
        if not self._is_recording:
            self._is_recording = True
            if self.on_start:
                self.on_start()
                
    def stop(self) -> None:
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
