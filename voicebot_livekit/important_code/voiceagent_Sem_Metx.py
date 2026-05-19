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

from livekit.agents import AgentStateChangedEvent, MetricsCollectedEvent, metrics

import logging

logger = logging.getLogger("voice-agent")
logging.basicConfig(level=logging.INFO)

load_dotenv()


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are an upbeat, slightly sarcastic voice AI for tech support. "
                "Help the Technical Support Representatives without rambling, and keep replies under 4 sentences."
            ),
        )


async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt=openai.STT(model="gpt-4o-mini-transcribe"),
        llm=openai.LLM(model="gpt-4.1-mini"),
        tts=openai.TTS(model="gpt-4o-mini-tts", voice="alloy"),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
        preemptive_generation=True,
    )

    # . Metrics setup
    usage_collector = metrics.UsageCollector()
    last_eou_metrics: metrics.EOUMetrics | None = None

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        nonlocal last_eou_metrics

        if ev.metrics.type == "eou_metrics":
            last_eou_metrics = ev.metrics

        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    # . Track latency (time to first audio)
    @session.on("agent_state_changed")
    def _on_agent_state_changed(ev: AgentStateChangedEvent):
        try:
            if (
                ev.new_state == "speaking"
                and last_eou_metrics
                and session.current_speech
                and last_eou_metrics.speech_id == session.current_speech.id
            ):
                # ✅ Use fallback-safe attribute
                if hasattr(last_eou_metrics, "last_speaking_time"):
                    start_time = last_eou_metrics.last_speaking_time
                elif hasattr(last_eou_metrics, "end_of_utterance"):
                    start_time = last_eou_metrics.end_of_utterance
                else:
                    return  # skip if not available

                delta = ev.created_at - start_time

                logger.info(
                    "Time to first audio frame: %sms",
                    delta.total_seconds() * 1000,
                )

        except Exception as e:
            logger.warning(f"Latency calc skipped: {e}")

    # . Shutdown summary
    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info("Usage summary: %s", summary)

    ctx.add_shutdown_callback(log_usage)

    # . Start session
    await session.start(
        agent=Assistant(),
        room=ctx.room,
    )

    await ctx.connect()

if __name__ == "__main__": cli.run_app( WorkerOptions( entrypoint_fnc=entrypoint, ) )