"""テキスト注入モジュール"""

from .send_input import TextInjector, InjectionResult, UIPIChecker, inject_text

__all__ = ["TextInjector", "InjectionResult", "UIPIChecker", "inject_text"]
