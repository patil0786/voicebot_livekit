# from livekit import api

# api_key = "APIcD9xtBwazU69"
# api_secret = "TXtXOFSwWZeJbOV0euyje5YZQ12nmIHTwmJLwlpFP2VB"

# token = api.AccessToken(api_key, api_secret) \
#     .with_identity("test-user") \
#     .with_name("Test User") \
#     .with_grants(api.VideoGrants(
#         room_join=True,
#         room="test-room"
#     ))

# print(token.to_jwt())

from livekit import api

# 🔑 same as docker
API_KEY = "APIcD9xtBwazU69"
API_SECRET = "TXtXOFSwWZeJbOV0euyje5YZQ12nmIHTwmJLwlpFP2VB"

# room + user
ROOM_NAME = "test-room"
IDENTITY = "user1"

token = api.AccessToken(API_KEY, API_SECRET) \
    .with_identity(IDENTITY) \
    .with_name("Test User") \
    .with_grants(
        api.VideoGrants(
            room_join=True,
            room=ROOM_NAME,
        )
    )

jwt_token = token.to_jwt()

print("\n✅ TOKEN:\n")
print(jwt_token)