import asyncio
import json
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import aiohttp

app = FastAPI()

# 🔧 Tool definition (VERY IMPORTANT)
TOOLS = [
    {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "input_schema": {   # ✅ NOT "parameters"
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name"
                }
            },
            "required": ["location"]
        }
    }
]


# 🌐 Weather API call
async def fetch_weather(location: str):
    url = f"http://shayne.app/weather?location={location}"

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                return {"error": f"API error {response.status}"}

            try:
                data = await response.json()
            except:
                return {"error": "Invalid JSON response"}

            return {
                "condition": data.get("condition"),
                "temperature": data.get("temperature"),
                "unit": data.get("unit"),
            }


# 📡 MCP SSE endpoint
@app.get("/sse")
async def sse_endpoint(request: Request):

    async def event_generator():
        # ✅ Proper JSON-RPC notification
        message = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "params": {
                "tools": TOOLS
            }
        }

        yield f"data: {json.dumps(message)}\n\n"

        while True:
            if await request.is_disconnected():
                break
            await asyncio.sleep(1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# 🧠 MCP tool execution endpoint
@app.post("/call")
async def call_tool(request: Request):
    body = await request.json()

    tool_name = body.get("method")
    params = body.get("params", {})
    request_id = body.get("id")

    if tool_name == "get_weather":
        location = params.get("location")
        result = await fetch_weather(location)

        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result
        }

    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {
            "code": -32601,
            "message": "Tool not found"
        }
    }