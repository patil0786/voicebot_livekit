from prometheus_client import start_http_server, Histogram, Counter

# 🔹 Latency metrics
stt_latency = Histogram("stt_latency_seconds", "STT latency")
llm_latency = Histogram("llm_latency_seconds", "LLM latency")
tts_latency = Histogram("tts_latency_seconds", "TTS latency")
e2e_latency = Histogram("e2e_latency_seconds", "End-to-End latency")

# 🔹 Counters
requests_count = Counter("voice_requests_total", "Total voice requests")


def start_metrics_server():
    # Expose metrics at http://localhost:8000/metrics
    start_http_server(8000)