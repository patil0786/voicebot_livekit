from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from livekit import api
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.getenv("LIVEKIT_API_KEY")
API_SECRET = os.getenv("LIVEKIT_API_SECRET")
LIVEKIT_URL = os.getenv("LIVEKIT_URL")

@app.get("/token")
def get_token():
    token_request = api.AccessToken(API_KEY, API_SECRET) \
        .with_identity("user1") \
        .with_name("User") \
        .with_grants(api.VideoGrants(
            room_join=True,
            room="test-room"
        ))

    token_jwt = token_request.to_jwt()
    
    # This will print the token to your terminal/console
    # print(f"\n--- GENERATED LIVEKIT TOKEN ---\n{token_jwt}\n-------------------------------\n")

    return {"token": token_jwt}