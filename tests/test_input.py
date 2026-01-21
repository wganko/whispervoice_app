"""
テキスト注入モジュールのテスト
"""

import pytest
from unittest.mock import MagicMock, patch
import ctypes

from src.input.send_input import (
    TextInjector, InjectionResult, UIPIChecker, inject_text,
    INPUT, KEYBDINPUT, KEYEVENTF_UNICODE
)


class TestInjectionResult:
    """InjectionResult データクラスのテスト"""
    
    def test_create_result(self):
        """結果オブジェクト作成テスト"""
        result = InjectionResult(
            success=True,
            characters_sent=10,
            failed_characters=[],
            elapsed_ms=5.0
        )
        assert result.success is True
        assert result.characters_sent == 10
        assert result.failed_characters == []
        assert result.elapsed_ms == 5.0
        
    def test_failed_result(self):
        """失敗結果テスト"""
        result = InjectionResult(
            success=False,
            characters_sent=8,
            failed_characters=["X", "Y"],
            elapsed_ms=10.0
        )
        assert result.success is False
        assert len(result.failed_characters) == 2


class TestTextInjector:
    """TextInjector クラスのテスト"""
    
    def test_init(self):
        """初期化テスト"""
        injector = TextInjector()
        assert injector.delay_between_chars_ms == 0.0
        assert injector.batch_size == 50
        
    def test_init_custom_params(self):
        """カスタムパラメータでの初期化テスト"""
        injector = TextInjector(
            delay_between_chars_ms=5.0,
            batch_size=100
        )
        assert injector.delay_between_chars_ms == 5.0
        assert injector.batch_size == 100
        
    def test_create_unicode_input(self):
        """Unicode INPUT 構造体作成テスト"""
        injector = TextInjector()
        
        inp = injector._create_unicode_input("あ", key_up=False)
        
        assert inp.type == 1  # INPUT_KEYBOARD
        assert inp.union.ki.wVk == 0
        assert inp.union.ki.wScan == ord("あ")
        assert inp.union.ki.dwFlags == KEYEVENTF_UNICODE
        
    def test_create_unicode_input_keyup(self):
        """Unicode INPUT 構造体作成テスト（キーアップ）"""
        injector = TextInjector()
        
        inp = injector._create_unicode_input("A", key_up=True)
        
        assert inp.union.ki.dwFlags == (KEYEVENTF_UNICODE | 0x0002)  # KEYEVENTF_KEYUP
        
    def test_inject_empty_string(self):
        """空文字列注入テスト"""
        injector = TextInjector()
        
        result = injector.inject("")
        
        assert result.success is True
        assert result.characters_sent == 0
        assert result.elapsed_ms == 0.0
        
    @patch.object(TextInjector, '_send_input')
    def test_inject_success(self, mock_send_input):
        """注入成功テスト（モック）"""
        mock_send_input.return_value = 2  # 2 events sent per char
        
        injector = TextInjector()
        result = injector.inject("AB")
        
        assert result.characters_sent == 2
        
    @patch.object(TextInjector, '_send_input')
    def test_inject_failure(self, mock_send_input):
        """注入失敗テスト（モック）"""
        mock_send_input.return_value = 0  # failure
        
        injector = TextInjector()
        result = injector.inject("X")
        
        assert result.success is False
        assert "X" in result.failed_characters


class TestUIPIChecker:
    """UIPIChecker クラスのテスト"""
    
    def test_init(self):
        """初期化テスト"""
        checker = UIPIChecker()
        assert checker._user32 is not None
        
    def test_check_uipi_restriction(self):
        """UIPI 制限チェックテスト"""
        checker = UIPIChecker()
        
        # 現在の簡易実装では常に False を返す
        result = checker.check_uipi_restriction()
        assert result is False


class TestInjectTextUtility:
    """inject_text ユーティリティ関数のテスト"""
    
    def test_inject_text_empty(self):
        """空文字列テスト"""
        result = inject_text("")
        
        assert result.success is True
        assert result.characters_sent == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
