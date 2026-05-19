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

from livekit.agents import (
    AgentStateChangedEvent,
    MetricsCollectedEvent,
    metrics,
)

import aiohttp
import logging
import os

load_dotenv()

logger = logging.getLogger("voice-agent")
logging.basicConfig(level=logging.INFO)

# =========================
#  CUSTOM FUNCTIONS
# =========================

def enforce_hindi(text: str) -> str:
    """Force LLM to respond in Hindi"""
    return f"कृपया केवल हिंदी में उत्तर दें:\n{text}"

def clean_text(text: str) -> str:
    """Normalize and limit response for faster TTS"""
    return text.strip().replace("\n", " ")[:300]

def get_fixed_tts():
    """Fix voice (no random switching)"""
    return openai.TTS(
        base_url=os.getenv("TTS_BASE_URL"),
        api_key="none",
        model="k2-fsa/OmniVoice",
        voice="speaker_0",  #  fixed voice
    )

# =========================
#  AGENT
# =========================

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a Hindi voice AI interviewer. "
                "Always respond ONLY in Hindi. "
                "Never respond in English. "
                "Keep answers short and clear. "
                "If user asks about weather, use the weather tool."
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
        logger.info(f" Fetching weather for {location}")

        coords = await self.get_coordinates(location)
        if not coords:
            return f"{location} का लोकेशन नहीं मिला।"

        lat, lon = coords

        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as res:
                    data = await res.json()

            current = data.get("current_weather", {})
            temp = current.get("temperature")
            wind = current.get("windspeed")

            return f"{location} में तापमान {temp} C है और हवा की गति {wind} km/h है।"

        except Exception as e:
            logger.error(e)
            return "अभी मौसम की जानकारी उपलब्ध नहीं है।"

# =========================
#  ENTRYPOINT
# =========================

async def entrypoint(ctx: JobContext):

    session = AgentSession(

        #  STT (Hindi locked)
        stt=openai.STT(
            model="Systran/faster-distil-whisper-large-v3",
            base_url=os.getenv("STT_BASE_URL"),
            api_key="cant-be-empty",
            language="hi"
        ),

        #  FAST LLM (major latency fix)
        llm=openai.LLM.with_ollama(
            # model="qwen2.5:7b-instruct",  #  fast model
            model= "qwen3:30b-a3b-instruct-2507-q4_K_M",
            base_url=os.getenv("OLLAMA_BASE_URL"),
        ),

        #  FIXED VOICE TTS
        tts=get_fixed_tts(),

        #  Audio processing
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),  #  kept as requested

        preemptive_generation=True,
        max_output_tokens=120,  #  reduce latency
    )

    # =========================
    #  FORCE HINDI INPUT
    # =========================
    @session.on("user_input_transcribed")
    def handle_input(ev):
        ev.text = enforce_hindi(ev.text)

    # =========================
    #  CLEAN OUTPUT
    # =========================
    @session.on("agent_response_generated")
    def clean_response(ev):
        ev.text = clean_text(ev.text)

    # =========================
    #  METRICS
    # =========================
    usage_collector = metrics.UsageCollector()
    last_eou_metrics: metrics.EOUMetrics | None = None

    @session.on("metrics_collected")
    def on_metrics(ev: MetricsCollectedEvent):
        nonlocal last_eou_metrics

        if ev.metrics.type == "eou_metrics":
            last_eou_metrics = ev.metrics

        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    # =========================
    #  LATENCY TRACK
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
                if hasattr(last_eou_metrics, "last_speaking_time"):
                    start_time = last_eou_metrics.last_speaking_time
                elif hasattr(last_eou_metrics, "end_of_utterance"):
                    start_time = last_eou_metrics.end_of_utterance
                else:
                    return

                latency = (ev.created_at - start_time).total_seconds() * 1000
                logger.info(f"⚡ Time to first audio: {latency:.2f} ms")

        except Exception as e:
            logger.warning(f"Latency calc skipped: {e}")

    # =========================
    #  SHUTDOWN SUMMARY
    # =========================
    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f" Usage summary: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # =========================
    # START SESSION
    # =========================
    await ctx.connect()
    await session.start(
        agent=Assistant(),
        room=ctx.room,
    )

# =========================
#  RUN APP
# =========================

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
        )
    )