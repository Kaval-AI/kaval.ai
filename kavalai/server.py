import os
import secrets

import uvicorn
from authlib.integrations.starlette_client import OAuth
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi import HTTPException
from httpx import AsyncClient
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import JSONResponse, RedirectResponse

load_dotenv()

app = FastAPI()

# OAuth setup
oauth = OAuth()

oauth.register(
    name='google',
    client_id=os.getenv("GOOGLE_OAUTH_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_OAUTH_CLIENT_SECRET"),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={
        'scope': 'openid email profile'
    }
)

ALLOWED_USERS = ["tpetmanson@gmail.com"]

AGENT_ENDPOINT_URL = "http://127.0.0.1:25123"

# Add SessionMiddleware with a secret key
app.add_middleware(SessionMiddleware, secret_key=secrets.token_urlsafe(16))


def is_logged_in(request: Request):
    return request.session.get("user_info") is not None


@app.get("/login")
async def login(request: Request):
    redirect_uri = request.url_for("google_auth_callback")
    return await oauth.google.authorize_redirect(request, redirect_uri)


@app.get("/logout")
async def login(request: Request):
    del request.session["user_info"]
    return JSONResponse({"logged_in": is_logged_in(request)})


@app.get("/user/get_details")
async def user_details(request: Request):
    if not is_logged_in(request):
        raise HTTPException(status_code=401, detail="Unauthorized.")
    return request.session.get("user_info")


@app.get("/auth/google/callback")
async def google_auth_callback(request: Request):
    token = await oauth.google.authorize_access_token(request)
    # Use userinfo endpoint instead of trying to parse id_token
    user_info = await oauth.google.userinfo(token=token)

    if user_info.email not in ALLOWED_USERS:
        return HTTPException(status_code=402, detail="Unauthorized.")

    request.session["user_info"] = user_info
    return RedirectResponse(url="http://localhost:4200")


@app.get("/chat/list")
async def chats_list(request: Request):
    """Queries the list of chats available to the agent."""
    # if not is_logged_in(request):
    #     raise HTTPException(status_code=401, detail="Unauthorized.")
    client = AsyncClient()
    results = await client.get(AGENT_ENDPOINT_URL + "/chat/list")
    assert results.status_code == 200
    return JSONResponse(results.json())


@app.get("/chat/messages/{chat_id}")
async def chats_messages(chat_id: str, request: Request):
    """Queries the list of chats available to the agent."""
    # if not is_logged_in(request):
    #     raise HTTPException(status_code=401, detail="Unauthorized.")
    client = AsyncClient()
    results = await client.get(AGENT_ENDPOINT_URL + "/chat/messages/" + chat_id)
    assert results.status_code == 200
    return JSONResponse(results.json())


@app.get("/input_schema")
async def input_schema(request: Request):
    # if not is_logged_in(request):
    #     raise HTTPException(status_code=401, detail="Unauthorized.")
    client = AsyncClient()
    results = await client.get(AGENT_ENDPOINT_URL + "/input_schema")
    assert results.status_code == 200
    return JSONResponse(results.json())


@app.get("/output_schema")
async def output_schema(request: Request):
    # if not is_logged_in(request):
    #     raise HTTPException(status_code=401, detail="Unauthorized.")
    client = AsyncClient()
    results = await client.get(AGENT_ENDPOINT_URL + "/output_schema")
    assert results.status_code == 200
    return JSONResponse(results.json())


if __name__ == "__main__":
    config = uvicorn.Config("kavalai.server:app", port=8000, log_level="debug", reload=True, access_log=True)
    server = uvicorn.Server(config)
    server.run()
