"""
ホットキーモジュールのテスト
"""

import pytest
from unittest.mock import MagicMock, call

from src.hotkey.global_hotkey import (
    GlobalHotkeyManager,
    RecordingToggle,
    HotkeyConfig,
    VK,
    MOD,
    DEFAULT_HOTKEY
)


class TestVK:
    """VK 定数のテスト"""
    
    def test_function_keys(self):
        """ファンクションキー定数テスト"""
        assert VK.F1 == 0x70
        assert VK.F8 == 0x77
        assert VK.F12 == 0x7B
        
    def test_modifier_keys(self):
        """修飾キー定数テスト"""
        assert VK.CONTROL == 0x11
        assert VK.SHIFT == 0x10
        assert VK.MENU == 0x12  # Alt


class TestMOD:
    """MOD 定数のテスト"""
    
    def test_modifiers(self):
        """モディファイア定数テスト"""
        assert MOD.ALT == 0x0001
        assert MOD.CONTROL == 0x0002
        assert MOD.SHIFT == 0x0004
        assert MOD.WIN == 0x0008


class TestHotkeyConfig:
    """HotkeyConfig データクラスのテスト"""
    
    def test_create_config(self):
        """設定作成テスト"""
        config = HotkeyConfig(
            key=VK.F8,
            modifiers=0,
            description="テスト"
        )
        assert config.key == VK.F8
        assert config.modifiers == 0
        assert config.description == "テスト"
        
    def test_str_simple(self):
        """文字列表現テスト（単純）"""
        config = HotkeyConfig(key=VK.F8, modifiers=0)
        assert str(config) == "F8"
        
    def test_str_with_modifiers(self):
        """文字列表現テスト（修飾キー付き）"""
        config = HotkeyConfig(
            key=VK.F8,
            modifiers=MOD.CONTROL | MOD.SHIFT
        )
        result = str(config)
        assert "Ctrl" in result
        assert "Shift" in result
        assert "F8" in result
        
    def test_str_with_win(self):
        """文字列表現テスト（Windows キー）"""
        config = HotkeyConfig(key=VK.F8, modifiers=MOD.WIN)
        result = str(config)
        assert "Win" in result


class TestDefaultHotkey:
    """デフォルトホットキーのテスト"""
    
    def test_default_config(self):
        """デフォルト設定テスト"""
        assert DEFAULT_HOTKEY.key == VK.F8
        assert DEFAULT_HOTKEY.modifiers == 0


class TestRecordingToggle:
    """RecordingToggle クラスのテスト"""
    
    def test_init(self):
        """初期化テスト"""
        toggle = RecordingToggle()
        assert toggle.is_recording is False
        
    def test_toggle_start(self):
        """トグル開始テスト"""
        toggle = RecordingToggle()
        
        result = toggle.toggle()
        
        assert result is True
        assert toggle.is_recording is True
        
    def test_toggle_stop(self):
        """トグル停止テスト"""
        toggle = RecordingToggle()
        toggle.toggle()  # start
        
        result = toggle.toggle()  # stop
        
        assert result is False
        assert toggle.is_recording is False
        
    def test_callbacks(self):
        """コールバックテスト"""
        on_start = MagicMock()
        on_stop = MagicMock()
        toggle = RecordingToggle(on_start=on_start, on_stop=on_stop)
        
        toggle.toggle()  # start
        on_start.assert_called_once()
        
        toggle.toggle()  # stop
        on_stop.assert_called_once()
        
    def test_start_method(self):
        """start メソッドテスト"""
        on_start = MagicMock()
        toggle = RecordingToggle(on_start=on_start)
        
        toggle.start()
        
        assert toggle.is_recording is True
        on_start.assert_called_once()
        
    def test_stop_method(self):
        """stop メソッドテスト"""
        on_stop = MagicMock()
        toggle = RecordingToggle(on_stop=on_stop)
        toggle._is_recording = True
        
        toggle.stop()
        
        assert toggle.is_recording is False
        on_stop.assert_called_once()
        
    def test_start_when_already_recording(self):
        """録音中に start を呼んでもコールバックは1回だけ"""
        on_start = MagicMock()
        toggle = RecordingToggle(on_start=on_start)
        
        toggle.start()
        toggle.start()  # 2回目
        
        on_start.assert_called_once()
        
    def test_stop_when_not_recording(self):
        """停止中に stop を呼んでもコールバックは呼ばれない"""
        on_stop = MagicMock()
        toggle = RecordingToggle(on_stop=on_stop)
        
        toggle.stop()
        
        on_stop.assert_not_called()


class TestGlobalHotkeyManager:
    """GlobalHotkeyManager クラスのテスト（Windows APIなし）"""
    
    def test_init(self):
        """初期化テスト"""
        manager = GlobalHotkeyManager()
        assert manager.is_running is False
        
    def test_register_before_start(self):
        """開始前の登録テスト"""
        manager = GlobalHotkeyManager()
        callback = MagicMock()
        
        result = manager.register(
            hotkey_id=1,
            config=HotkeyConfig(key=VK.F8, modifiers=0),
            callback=callback
        )
        
        assert result is True
        assert 1 in manager._hotkeys
        
    def test_unregister_before_start(self):
        """開始前の解除テスト"""
        manager = GlobalHotkeyManager()
        callback = MagicMock()
        
        manager.register(1, HotkeyConfig(key=VK.F8, modifiers=0), callback)
        result = manager.unregister(1)
        
        assert result is True
        assert 1 not in manager._hotkeys


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
