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

import aiohttp
import logging

load_dotenv()

logger = logging.getLogger("voice-agent")
logging.basicConfig(level=logging.INFO)


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful voice AI assistant. "
                "If user asks about weather, ALWAYS use the weather tool."
            ),
        )

    # ✅ Tool 1: Get coordinates from city
    async def get_coordinates(self, city: str):
        url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as res:
                data = await res.json()
                if "results" not in data:
                    return None
                result = data["results"][0]
                return result["latitude"], result["longitude"]

    # ✅ Tool 2: Weather function tool
    @function_tool
    async def get_weather(
        self, context: RunContext, location: str
    ) -> str:
        """Get current weather for a city"""
        logger.info(f"Fetching weather for {location}")

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

    


async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt=openai.STT(model="gpt-4o-mini-transcribe"),
        llm=openai.LLM(model="gpt-4.1-mini"),
        tts=openai.TTS(model="gpt-4o-mini-tts", voice="alloy"),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
        preemptive_generation=True,
    )

    await session.start(
        agent=Assistant(),
        room=ctx.room,
    )

    
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
        )
    )