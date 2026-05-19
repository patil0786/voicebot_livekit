from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
    inference,
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
            instructions=("""
                You are Rahul, a friendly, human-like voice assistant for Apex Capital helping customers get SME loans.

                Speak naturally with short sentences, light pauses (...), and a warm tone.

                You can speak maximum 4 sentences per response.

                Flow:
                Step 1: Greet and understand need  
                Step 2: Ask business type, loan amount, tenure  
                Step 3: Suggest suitable loan  
                Step 4: Encourage application
                

                Loan Details:
                - Amount: Upto ₹10 Crore  
                - Tenure: upto 15 years  
                - Interest: starting from 10%    
                - Approval: within 72 hours  
                - Collateral: not needed for small loans  
                - Basic KYC required

                Apex Capital Contact Details:
                Address = Apex Capital, Nariman Point, Mumbai
                contact-apex@gmail.com
                +91-777-2334-777

                Rules:
                - Max 4 sentences 
                - Ask only one question at a time
                - Use simple conversational English  
                - Use "..." or commas for pauses  

                Be polite, confirm details, and guide step-by-step.
                """
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

            return f"The weather in {location} is {temp}C with wind speed {wind} km/h."

        except Exception as e:
            logger.error(e)
            return "Weather service is unavailable right now."


# =========================
# 🚀 ENTRYPOINT
# =========================
async def entrypoint(ctx: JobContext):

    session = AgentSession(

       
        
        # stt=inference.STT(
        # model="assemblyai/universal-streaming-multilingual",
        # ),


    #     tts = inference.TTS(
    #        model="xai/tts-1",
    # ),

        stt=inference.STT(
                model="elevenlabs/scribe_v2_realtime",
                language="hi-IN"
            ),
        llm="openai/gpt-4.1-mini",
        tts=inference.TTS(
            model="elevenlabs/eleven_turbo_v2_5",
            voice="Xb7hH8MSUJpSbSDYk0k2",
            language="hi"
        ),

        
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
        logger.info(f"Usage summary: {summary}")

    ctx.add_shutdown_callback(log_usage)


    await ctx.connect()
    await session.start(
        agent=Assistant(),
        room=ctx.room,
    )

    

# =========================
# 🏁 RUN APP
# =========================
if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
        )
    )