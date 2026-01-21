"""
レイテンシ計測モジュールのテスト
"""

import pytest
import time
import json

from src.metrics.latency import (
    LatencyTimer,
    LatencyLogger,
    LatencyMeasurement,
    LatencyStatistics,
    MeasurementPoint,
    get_latency_logger
)


class TestMeasurementPoint:
    """MeasurementPoint Enum のテスト"""
    
    def test_points(self):
        """計測ポイント定数テスト"""
        assert MeasurementPoint.SPEECH_END.value == "speech_end"
        assert MeasurementPoint.STT_START.value == "stt_start"
        assert MeasurementPoint.STT_END.value == "stt_end"
        assert MeasurementPoint.INJECTION_START.value == "injection_start"
        assert MeasurementPoint.INJECTION_END.value == "injection_end"


class TestLatencyMeasurement:
    """LatencyMeasurement データクラスのテスト"""
    
    def test_create_measurement(self):
        """計測結果作成テスト"""
        measurement = LatencyMeasurement(
            total_latency_ms=100.0,
            stt_duration_ms=80.0,
            text_length=10
        )
        assert measurement.total_latency_ms == 100.0
        assert measurement.stt_duration_ms == 80.0
        assert measurement.text_length == 10
        
    def test_to_dict(self):
        """辞書変換テスト"""
        measurement = LatencyMeasurement(
            total_latency_ms=100.5,
            stt_duration_ms=80.3
        )
        result = measurement.to_dict()
        
        assert result["total_latency_ms"] == 100.5
        assert result["stt_duration_ms"] == 80.3
        assert "timestamp" in result


class TestLatencyStatistics:
    """LatencyStatistics データクラスのテスト"""
    
    def test_create_statistics(self):
        """統計作成テスト"""
        stats = LatencyStatistics(
            count=100,
            mean_ms=150.0,
            median_ms=140.0
        )
        assert stats.count == 100
        assert stats.mean_ms == 150.0
        
    def test_to_dict(self):
        """辞書変換テスト"""
        stats = LatencyStatistics(
            count=50,
            mean_ms=200.5,
            min_ms=100.0,
            max_ms=500.0
        )
        result = stats.to_dict()
        
        assert result["count"] == 50
        assert result["mean_ms"] == 200.5


class TestLatencyTimer:
    """LatencyTimer クラスのテスト"""
    
    def test_init(self):
        """初期化テスト"""
        timer = LatencyTimer()
        assert len(timer._timestamps) == 0
        
    def test_mark(self):
        """マーク記録テスト"""
        timer = LatencyTimer()
        
        timer.mark(MeasurementPoint.SPEECH_END)
        
        assert MeasurementPoint.SPEECH_END in timer._timestamps
        
    def test_reset(self):
        """リセットテスト"""
        timer = LatencyTimer()
        timer.mark(MeasurementPoint.SPEECH_END)
        
        timer.reset()
        
        assert len(timer._timestamps) == 0
        
    def test_get_measurement(self):
        """計測結果取得テスト"""
        timer = LatencyTimer()
        
        timer.mark(MeasurementPoint.SPEECH_END)
        time.sleep(0.01)  # 10ms
        timer.mark(MeasurementPoint.STT_START)
        time.sleep(0.01)
        timer.mark(MeasurementPoint.STT_END)
        time.sleep(0.01)
        timer.mark(MeasurementPoint.INJECTION_START)
        time.sleep(0.01)
        timer.mark(MeasurementPoint.INJECTION_END)
        
        measurement = timer.get_measurement(text_length=5)
        
        assert measurement.text_length == 5
        assert measurement.total_latency_ms > 0
        assert measurement.stt_duration_ms > 0
        
    def test_get_measurement_empty(self):
        """空の計測結果テスト"""
        timer = LatencyTimer()
        
        measurement = timer.get_measurement()
        
        assert measurement.total_latency_ms == 0.0


class TestLatencyLogger:
    """LatencyLogger クラスのテスト"""
    
    def test_init(self):
        """初期化テスト"""
        logger = LatencyLogger()
        assert logger.max_history == 1000
        assert len(logger._measurements) == 0
        
    def test_log(self):
        """ログ記録テスト"""
        logger = LatencyLogger()
        measurement = LatencyMeasurement(total_latency_ms=100.0)
        
        logger.log(measurement)
        
        assert len(logger._measurements) == 1
        
    def test_get_statistics_empty(self):
        """空の統計テスト"""
        logger = LatencyLogger()
        
        stats = logger.get_statistics()
        
        assert stats.count == 0
        
    def test_get_statistics(self):
        """統計取得テスト"""
        logger = LatencyLogger()
        
        for latency in [100, 200, 300, 400, 500]:
            logger.log(LatencyMeasurement(total_latency_ms=latency))
            
        stats = logger.get_statistics()
        
        assert stats.count == 5
        assert stats.mean_ms == 300.0
        assert stats.min_ms == 100.0
        assert stats.max_ms == 500.0
        
    def test_get_recent(self):
        """直近取得テスト"""
        logger = LatencyLogger()
        
        for i in range(20):
            logger.log(LatencyMeasurement(total_latency_ms=i * 10))
            
        recent = logger.get_recent(5)
        
        assert len(recent) == 5
        assert recent[-1].total_latency_ms == 190.0
        
    def test_clear(self):
        """クリアテスト"""
        logger = LatencyLogger()
        logger.log(LatencyMeasurement(total_latency_ms=100.0))
        
        logger.clear()
        
        assert len(logger._measurements) == 0
        
    def test_export_json(self):
        """JSON エクスポートテスト"""
        logger = LatencyLogger()
        logger.log(LatencyMeasurement(total_latency_ms=100.0))
        
        result = logger.export_json()
        data = json.loads(result)
        
        assert "statistics" in data
        assert "measurements" in data
        assert len(data["measurements"]) == 1
        
    def test_check_target_success(self):
        """目標達成テスト（成功）"""
        logger = LatencyLogger()
        logger.log(LatencyMeasurement(total_latency_ms=400.0))
        
        assert logger.check_target(500.0) is True
        
    def test_check_target_failure(self):
        """目標達成テスト（失敗）"""
        logger = LatencyLogger()
        logger.log(LatencyMeasurement(total_latency_ms=600.0))
        
        assert logger.check_target(500.0) is False
        
    def test_max_history(self):
        """最大履歴テスト"""
        logger = LatencyLogger(max_history=10)
        
        for i in range(20):
            logger.log(LatencyMeasurement(total_latency_ms=i))
            
        assert len(logger._measurements) == 10


class TestGetLatencyLogger:
    """get_latency_logger 関数のテスト"""
    
    def test_returns_singleton(self):
        """シングルトンテスト"""
        logger1 = get_latency_logger()
        logger2 = get_latency_logger()
        
        assert logger1 is logger2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
