import time
from contextlib import contextmanager
from typing import Dict, List

class MetricsCollector:
    def __init__(self):
        self.latencies: Dict[str, List[float]] = {}
        self.hallucinations: Dict[str, int] = {"total": 0, "prevented": 0}
        self.abstentions = 0

    def record_latency(self, endpoint: str, latency_ms: float):
        if endpoint not in self.latencies:
            self.latencies[endpoint] = []
        self.latencies[endpoint].append(latency_ms)

    def record_hallucination_prevented(self):
        self.hallucinations["prevented"] += 1
        self.hallucinations["total"] += 1

    def record_hallucination_total(self):
        self.hallucinations["total"] += 1

    def record_abstention(self):
        self.abstentions += 1

    def get_stats(self):
        stats = {
            "latencies": {},
            "hallucination_rate": 0.0,
            "abstentions": self.abstentions,
        }
        
        for endpoint, times in self.latencies.items():
            if times:
                stats["latencies"][endpoint] = {
                    "avg_ms": round(sum(times) / len(times), 2),
                    "min_ms": round(min(times), 2),
                    "max_ms": round(max(times), 2),
                    "count": len(times),
                }
        
        if self.hallucinations["total"] > 0:
            prevented_rate = self.hallucinations["prevented"] / self.hallucinations["total"]
            stats["hallucination_rate"] = round(prevented_rate, 4)
        
        return stats

    def reset(self):
        self.latencies.clear()
        self.hallucinations = {"total": 0, "prevented": 0}
        self.abstentions = 0


metrics = MetricsCollector()


@contextmanager
def measure_latency(endpoint: str):
    start = time.time()
    try:
        yield
    finally:
        latency_ms = (time.time() - start) * 1000
        metrics.record_latency(endpoint, latency_ms)
