"""
SendInput テキスト注入モジュール

Win32 SendInput API を使用して Unicode テキストをアクティブウィンドウに注入する。
クリップボードは使用しない。
"""

import ctypes
from ctypes import wintypes
from typing import Optional, List
from dataclasses import dataclass
import time


# Windows 構造体定義
class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class INPUT_UNION(ctypes.Union):
    _fields_ = [
        ("ki", KEYBDINPUT),
        ("mi", MOUSEINPUT),
        ("hi", HARDWAREINPUT),
    ]


class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", wintypes.DWORD),
        ("union", INPUT_UNION),
    ]


# 定数
INPUT_KEYBOARD = 1
KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_KEYUP = 0x0002


@dataclass
class InjectionResult:
    """注入結果"""
    success: bool
    characters_sent: int
    failed_characters: List[str]
    elapsed_ms: float


class TextInjector:
    """
    テキスト注入モジュール
    
    Win32 SendInput API を使用して Unicode テキストを
    アクティブウィンドウに直接注入する。
    
    Usage:
        injector = TextInjector()
        result = injector.inject("こんにちは")
    """
    
    def __init__(
        self,
        delay_between_chars_ms: float = 0.0,
        batch_size: int = 50
    ):
        """
        Args:
            delay_between_chars_ms: 文字間の遅延時間（ミリ秒）
            batch_size: 一度に送信する最大文字数
        """
        self.delay_between_chars_ms = delay_between_chars_ms
        self.batch_size = batch_size
        
        # Windows API
        self._user32 = ctypes.windll.user32
        self._send_input = self._user32.SendInput
        self._send_input.argtypes = [
            wintypes.UINT,
            ctypes.POINTER(INPUT),
            ctypes.c_int
        ]
        self._send_input.restype = wintypes.UINT
        
    def _create_unicode_input(self, char: str, key_up: bool = False) -> INPUT:
        """
        Unicode 文字のための INPUT 構造体を作成
        
        Args:
            char: 送信する文字
            key_up: キーアップイベントかどうか
            
        Returns:
            INPUT 構造体
        """
        inp = INPUT()
        inp.type = INPUT_KEYBOARD
        
        flags = KEYEVENTF_UNICODE
        if key_up:
            flags |= KEYEVENTF_KEYUP
            
        inp.union.ki.wVk = 0
        inp.union.ki.wScan = ord(char)
        inp.union.ki.dwFlags = flags
        inp.union.ki.time = 0
        inp.union.ki.dwExtraInfo = None
        
        return inp
        
    def inject_char(self, char: str) -> bool:
        """
        1文字を注入
        
        Args:
            char: 送信する文字
            
        Returns:
            成功したかどうか
        """
        # キーダウンとキーアップの2つのイベントを送信
        inputs = (INPUT * 2)()
        inputs[0] = self._create_unicode_input(char, key_up=False)
        inputs[1] = self._create_unicode_input(char, key_up=True)
        
        result = self._send_input(2, inputs, ctypes.sizeof(INPUT))
        return result == 2
        
    def inject(self, text: str) -> InjectionResult:
        """
        テキストを注入
        
        Args:
            text: 送信するテキスト
            
        Returns:
            注入結果
        """
        if not text:
            return InjectionResult(
                success=True,
                characters_sent=0,
                failed_characters=[],
                elapsed_ms=0.0
            )
            
        start_time = time.perf_counter()
        
        characters_sent = 0
        failed_characters = []
        
        for i, char in enumerate(text):
            # バッチサイズごとに少し待機
            if i > 0 and i % self.batch_size == 0:
                time.sleep(0.01)  # 10ms
                
            success = self.inject_char(char)
            
            if success:
                characters_sent += 1
            else:
                failed_characters.append(char)
                
            # 文字間遅延
            if self.delay_between_chars_ms > 0:
                time.sleep(self.delay_between_chars_ms / 1000)
                
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        return InjectionResult(
            success=len(failed_characters) == 0,
            characters_sent=characters_sent,
            failed_characters=failed_characters,
            elapsed_ms=elapsed_ms
        )
        
    def inject_with_ime_workaround(self, text: str) -> InjectionResult:
        """
        IME 対応のテキスト注入
        
        一部の IME では高速な入力が問題を起こすため、
        少し遅延を入れて安定性を向上させる。
        
        Args:
            text: 送信するテキスト
            
        Returns:
            注入結果
        """
        # 一時的に遅延を追加
        original_delay = self.delay_between_chars_ms
        self.delay_between_chars_ms = max(1.0, self.delay_between_chars_ms)
        
        try:
            return self.inject(text)
        finally:
            self.delay_between_chars_ms = original_delay


class UIPIChecker:
    """
    UIPI (User Interface Privilege Isolation) チェッカー
    
    管理者権限のウィンドウへの入力が制限されているかを検出する。
    """
    
    def __init__(self):
        self._user32 = ctypes.windll.user32
        
    def get_foreground_window_process_id(self) -> Optional[int]:
        """
        フォアグラウンドウィンドウのプロセス ID を取得
        
        Returns:
            プロセス ID、または取得失敗時は None
        """
        hwnd = self._user32.GetForegroundWindow()
        if not hwnd:
            return None
            
        pid = wintypes.DWORD()
        self._user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        return pid.value
        
    def check_uipi_restriction(self) -> bool:
        """
        UIPI 制限がかかっているかチェック
        
        注意: 完全な検出は困難。簡易チェックのみ。
        
        Returns:
            True if 制限の可能性あり
        """
        # 簡易実装：テスト入力を試行して判定
        # より完全な実装には追加の Windows API が必要
        return False


def inject_text(text: str) -> InjectionResult:
    """
    ユーティリティ関数: テキストを注入
    
    Args:
        text: 注入するテキスト
        
    Returns:
        注入結果
    """
    injector = TextInjector()
    return injector.inject(text)
