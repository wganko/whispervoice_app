"""
レイテンシ計測モジュール

発話終了から SendInput 完了までの時間を ms 単位で記録する。
"""

import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import statistics
import json
import logging


class MeasurementPoint(Enum):
    """計測ポイント"""
    SPEECH_END = "speech_end"           # 発話終了（VAD 検出）
    STT_START = "stt_start"             # 音声認識開始
    STT_END = "stt_end"                 # 音声認識完了
    INJECTION_START = "injection_start" # テキスト注入開始
    INJECTION_END = "injection_end"     # テキスト注入完了


@dataclass
class LatencyMeasurement:
    """レイテンシ計測結果"""
    speech_end_to_stt_start_ms: float = 0.0
    stt_duration_ms: float = 0.0
    stt_end_to_injection_ms: float = 0.0
    injection_duration_ms: float = 0.0
    total_latency_ms: float = 0.0
    text_length: int = 0
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "speech_end_to_stt_start_ms": round(self.speech_end_to_stt_start_ms, 2),
            "stt_duration_ms": round(self.stt_duration_ms, 2),
            "stt_end_to_injection_ms": round(self.stt_end_to_injection_ms, 2),
            "injection_duration_ms": round(self.injection_duration_ms, 2),
            "total_latency_ms": round(self.total_latency_ms, 2),
            "text_length": self.text_length,
            "timestamp": self.timestamp
        }


@dataclass
class LatencyStatistics:
    """レイテンシ統計"""
    count: int = 0
    mean_ms: float = 0.0
    median_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    stddev_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "count": self.count,
            "mean_ms": round(self.mean_ms, 2),
            "median_ms": round(self.median_ms, 2),
            "min_ms": round(self.min_ms, 2),
            "max_ms": round(self.max_ms, 2),
            "stddev_ms": round(self.stddev_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "p99_ms": round(self.p99_ms, 2)
        }


class LatencyTimer:
    """
    レイテンシ計測タイマー
    
    各処理フェーズの開始/終了時刻を記録し、
    レイテンシを計算する。
    
    Usage:
        timer = LatencyTimer()
        timer.mark(MeasurementPoint.SPEECH_END)
        # ... STT 処理 ...
        timer.mark(MeasurementPoint.STT_START)
        timer.mark(MeasurementPoint.STT_END)
        # ... テキスト注入 ...
        timer.mark(MeasurementPoint.INJECTION_START)
        timer.mark(MeasurementPoint.INJECTION_END)
        measurement = timer.get_measurement(text_length=10)
    """
    
    def __init__(self):
        self._timestamps: Dict[MeasurementPoint, float] = {}
        
    def mark(self, point: MeasurementPoint) -> None:
        """
        計測ポイントを記録
        
        Args:
            point: 計測ポイント
        """
        self._timestamps[point] = time.perf_counter() * 1000  # ms
        
    def reset(self) -> None:
        """タイマーをリセット"""
        self._timestamps.clear()
        
    def get_measurement(self, text_length: int = 0) -> LatencyMeasurement:
        """
        計測結果を取得
        
        Args:
            text_length: 認識したテキストの長さ
            
        Returns:
            レイテンシ計測結果
        """
        measurement = LatencyMeasurement(text_length=text_length)
        
        # 各区間を計算
        if MeasurementPoint.SPEECH_END in self._timestamps:
            speech_end = self._timestamps[MeasurementPoint.SPEECH_END]
            
            if MeasurementPoint.STT_START in self._timestamps:
                stt_start = self._timestamps[MeasurementPoint.STT_START]
                measurement.speech_end_to_stt_start_ms = stt_start - speech_end
                
            if MeasurementPoint.STT_END in self._timestamps:
                stt_end = self._timestamps[MeasurementPoint.STT_END]
                if MeasurementPoint.STT_START in self._timestamps:
                    stt_start = self._timestamps[MeasurementPoint.STT_START]
                    measurement.stt_duration_ms = stt_end - stt_start
                    
                if MeasurementPoint.INJECTION_START in self._timestamps:
                    injection_start = self._timestamps[MeasurementPoint.INJECTION_START]
                    measurement.stt_end_to_injection_ms = injection_start - stt_end
                    
            if MeasurementPoint.INJECTION_END in self._timestamps:
                injection_end = self._timestamps[MeasurementPoint.INJECTION_END]
                if MeasurementPoint.INJECTION_START in self._timestamps:
                    injection_start = self._timestamps[MeasurementPoint.INJECTION_START]
                    measurement.injection_duration_ms = injection_end - injection_start
                    
                # 合計レイテンシ
                measurement.total_latency_ms = injection_end - speech_end
                
        return measurement


class LatencyLogger:
    """
    レイテンシログ記録
    
    計測結果を保存し、統計情報を提供する。
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Args:
            max_history: 保持する履歴の最大数
        """
        self.max_history = max_history
        self._measurements: deque[LatencyMeasurement] = deque(maxlen=max_history)
        self._logger = logging.getLogger(__name__)
        
    def log(self, measurement: LatencyMeasurement) -> None:
        """
        計測結果を記録
        
        Args:
            measurement: レイテンシ計測結果
        """
        self._measurements.append(measurement)
        
        # ログ出力
        self._logger.info(
            f"Latency: {measurement.total_latency_ms:.2f}ms "
            f"(STT: {measurement.stt_duration_ms:.2f}ms, "
            f"Inject: {measurement.injection_duration_ms:.2f}ms)"
        )
        
    def get_statistics(self) -> LatencyStatistics:
        """
        統計情報を取得
        
        Returns:
            レイテンシ統計
        """
        if not self._measurements:
            return LatencyStatistics()
            
        latencies = [m.total_latency_ms for m in self._measurements]
        sorted_latencies = sorted(latencies)
        n = len(latencies)
        
        return LatencyStatistics(
            count=n,
            mean_ms=statistics.mean(latencies),
            median_ms=statistics.median(latencies),
            min_ms=min(latencies),
            max_ms=max(latencies),
            stddev_ms=statistics.stdev(latencies) if n > 1 else 0.0,
            p95_ms=sorted_latencies[int(n * 0.95)] if n >= 20 else max(latencies),
            p99_ms=sorted_latencies[int(n * 0.99)] if n >= 100 else max(latencies)
        )
        
    def get_recent(self, count: int = 10) -> List[LatencyMeasurement]:
        """
        直近の計測結果を取得
        
        Args:
            count: 取得する件数
            
        Returns:
            計測結果のリスト
        """
        return list(self._measurements)[-count:]
        
    def clear(self) -> None:
        """履歴をクリア"""
        self._measurements.clear()
        
    def export_json(self) -> str:
        """
        JSON 形式でエクスポート
        
        Returns:
            JSON 文字列
        """
        data = {
            "statistics": self.get_statistics().to_dict(),
            "measurements": [m.to_dict() for m in self._measurements]
        }
        return json.dumps(data, indent=2, ensure_ascii=False)
        
    def check_target(self, target_ms: float = 500.0) -> bool:
        """
        目標レイテンシを達成しているか確認
        
        Args:
            target_ms: 目標レイテンシ (ms)
            
        Returns:
            True if 直近の計測が目標を達成
        """
        if not self._measurements:
            return True
            
        recent = self._measurements[-1]
        return recent.total_latency_ms <= target_ms


# グローバルインスタンス
_default_logger: Optional[LatencyLogger] = None


def get_latency_logger() -> LatencyLogger:
    """デフォルトのレイテンシロガーを取得"""
    global _default_logger
    if _default_logger is None:
        _default_logger = LatencyLogger()
    return _default_logger
