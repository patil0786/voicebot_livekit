from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
)

from livekit.plugins import openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from livekit.agents.llm import function_tool
from livekit.agents import RunContext

# ✅ Metrics imports
from livekit.agents import (
    AgentStateChangedEvent,
    MetricsCollectedEvent,
    metrics,
)

import aiohttp
import logging
import time

load_dotenv()

logger = logging.getLogger("voice-agent")
logging.basicConfig(level=logging.INFO)


# =========================
# 🤖 AGENT WITH TOOL
# =========================
class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful voice AI assistant.give me short answers. "
                "If user asks about weather, ALWAYS use the weather tool."
            ),
        )

    async def get_coordinates(self, city: str):
        url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as res:
                data = await res.json()
                if "results" not in data:
                    return None
                result = data["results"][0]
                return result["latitude"], result["longitude"]

    @function_tool
    async def get_weather(self, context: RunContext, location: str) -> str:
        logger.info(f"🌦️ Fetching weather for {location}")

        coords = await self.get_coordinates(location)
        if not coords:
            return f"Could not find location {location}"

        lat, lon = coords

        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as res:
                    data = await res.json()

            current = data.get("current_weather", {})
            temp = current.get("temperature")
            wind = current.get("windspeed")

            return f"The weather in {location} is {temp}°C with wind speed {wind} km/h."

        except Exception as e:
            logger.error(e)
            return "Weather service is unavailable right now."


# =========================
# 🚀 ENTRYPOINT
# =========================
async def entrypoint(ctx: JobContext):

    session = AgentSession(
        stt=openai.STT(model="gpt-4o-mini-transcribe"),
        llm=openai.LLM(model="gpt-4.1-mini"),
        tts=openai.TTS(model="gpt-4o-mini-tts", voice="alloy"),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
        preemptive_generation=True,
    )

    # =========================
    # 📊 METRICS SETUP
    # =========================
    usage_collector = metrics.UsageCollector()
    last_eou_metrics: metrics.EOUMetrics | None = None

    @session.on("metrics_collected")
    def on_metrics(ev: MetricsCollectedEvent):
        nonlocal last_eou_metrics

        # Store EOU for latency calc
        if ev.metrics.type == "eou_metrics":
            last_eou_metrics = ev.metrics

        # Log all metrics (STT, LLM, TTS)
        metrics.log_metrics(ev.metrics)

        # Collect usage
        usage_collector.collect(ev.metrics)

    # =========================
    # ⚡ REAL LATENCY (IMPORTANT)
    # =========================
    @session.on("agent_state_changed")
    def on_state_change(ev: AgentStateChangedEvent):
        try:
            if (
                ev.new_state == "speaking"
                and last_eou_metrics
                and session.current_speech
                and last_eou_metrics.speech_id == session.current_speech.id
            ):
                # Safe attribute handling
                if hasattr(last_eou_metrics, "last_speaking_time"):
                    start_time = last_eou_metrics.last_speaking_time
                elif hasattr(last_eou_metrics, "end_of_utterance"):
                    start_time = last_eou_metrics.end_of_utterance
                else:
                    return

                latency = (ev.created_at - start_time).total_seconds() * 1000

                logger.info(f"🚀 Time to first audio: {latency:.2f} ms")

        except Exception as e:
            logger.warning(f"Latency calc skipped: {e}")

    # =========================
    # 📦 SHUTDOWN SUMMARY
    # =========================
    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"📊 Usage summary: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # =========================
    # ▶ START SESSION
    # =========================
    await session.start(
        agent=Assistant(),
        room=ctx.room,
    )

    await ctx.connect()


# =========================
# 🏁 RUN APP
# =========================
if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
        )
    )