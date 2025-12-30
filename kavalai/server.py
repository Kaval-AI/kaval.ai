import logging
import os
import secrets

import uvicorn
from authlib.integrations.starlette_client import OAuth
from fastapi import FastAPI, Request, HTTPException, status
from sqlalchemy import select
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import JSONResponse, RedirectResponse

from kavalai.db import AsyncKavalaiSession, User

# Set up the app logger
logger = logging.getLogger(__name__)
logger.propagate = True

app = FastAPI()

# OAuth setup
oauth = OAuth()

oauth.register(
    name="google",
    client_id=os.getenv("GOOGLE_OAUTH_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_OAUTH_CLIENT_SECRET"),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

AGENT_ENDPOINT_URL = "http://127.0.0.1:25123"

# Add SessionMiddleware with a secret key
app.add_middleware(SessionMiddleware, secret_key=secrets.token_urlsafe(16))


async def authenticate_and_sync_user(user_info: dict):
    async with AsyncKavalaiSession() as session:
        # Check if user exists in the database.
        email = user_info.get("email")
        stmt = select(User).where(User.email == email)
        result = await session.execute(stmt)
        user = result.scalars().first()

        # If not, raise exception.
        if not user:
            logger.warning(f"Unauthorized login attempt: {email}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User not registered in the system.",
            )

        # Update picture and name
        user.name = user_info.get("name", user.name)
        user.picture = user_info.get("picture", user.picture)

        await session.commit()
        await session.refresh(user)
        return user


def is_logged_in(request: Request):
    return request.session.get("user_info") is not None


@app.get("/login")
async def login(request: Request):
    redirect_uri = request.url_for("google_auth_callback")
    return await oauth.google.authorize_redirect(request, redirect_uri)


@app.get("/logout")
async def logout(request: Request):
    del request.session["user_info"]
    return JSONResponse({"logged_in": is_logged_in(request)})


@app.get("/user/get_details")
async def user_details(request: Request):
    if not is_logged_in(request):
        raise HTTPException(status_code=401, detail="Unauthorized.")
    return request.session.get("user_info")


@app.get("/auth/google/callback")
async def google_auth_callback(request: Request):
    try:
        token = await oauth.google.authorize_access_token(request)
        user_info = await oauth.google.userinfo(token=token)
        db_user = await authenticate_and_sync_user(user_info)
        # Store essential info in the session
        request.session["user_info"] = {
            "id": str(db_user.id),
            "email": db_user.email,
            "name": db_user.name,
            "picture": db_user.picture,
            "is_admin": db_user.is_admin,
        }
        return RedirectResponse(url="http://localhost:4200")
    except HTTPException as e:
        # Re-raise the 403 we created in the helper
        raise e
    except Exception as e:
        logger.error(f"Auth error: {e}")
        raise HTTPException(status_code=400, detail="Authentication failed.")


# @app.get("/chat/list")
# async def chats_list(request: Request):
#     """Queries the list of chats available to the agent."""
#     # if not is_logged_in(request):
#     #     raise HTTPException(status_code=401, detail="Unauthorized.")
#     client = AsyncClient()
#     results = await client.get(AGENT_ENDPOINT_URL + "/chat/list")
#     assert results.status_code == 200
#     return JSONResponse(results.json())
#
#
# @app.get("/chat/messages/{chat_id}")
# async def chats_messages(chat_id: str, request: Request):
#     """Queries the list of chats available to the agent."""
#     # if not is_logged_in(request):
#     #     raise HTTPException(status_code=401, detail="Unauthorized.")
#     client = AsyncClient()
#     results = await client.get(AGENT_ENDPOINT_URL + "/chat/messages/" + chat_id)
#     assert results.status_code == 200
#     return JSONResponse(results.json())
#
#
# @app.get("/chat/agent_runs/{chat_id}")
# async def input_schema(chat_id: str, request: Request):
#     # if not is_logged_in(request):
#     #     raise HTTPException(status_code=401, detail="Unauthorized.")
#     client = AsyncClient()
#     results = await client.get(AGENT_ENDPOINT_URL + "/chat/agent_runs/" + chat_id)
#     assert results.status_code == 200
#     return JSONResponse(results.json())
#
#
# @app.post("/agent/run")
# async def run_agent(request: Request):
#     # if not is_logged_in(request):
#     #     raise HTTPException(status_code=401, detail="Unauthorized.")
#     request_data = await request.json()
#     client = AsyncClient()
#     results = await client.post(AGENT_ENDPOINT_URL + "/run", json=request_data, timeout=60.0)
#     if results.status_code != 200:
#         raise HTTPException(status_code=results.status_code, detail=results.content)
#     return JSONResponse(results.json())
#
#
# @app.get("/input_schema")
# async def input_schema(request: Request):
#     # if not is_logged_in(request):
#     #     raise HTTPException(status_code=401, detail="Unauthorized.")
#     client = AsyncClient()
#     results = await client.get(AGENT_ENDPOINT_URL + "/input_schema")
#     if results.status_code != 200:
#         raise HTTPException(status_code=results.status_code, detail=results.content)
#     return JSONResponse(results.json())
#
#
# @app.get("/output_schema")
# async def output_schema(request: Request):
#     # if not is_logged_in(request):
#     #     raise HTTPException(status_code=401, detail="Unauthorized.")
#     client = AsyncClient()
#     results = await client.get(AGENT_ENDPOINT_URL + "/output_schema")
#     if results.status_code != 200:
#         raise HTTPException(status_code=results.status_code, detail=results.content)
#     assert results.status_code == 200
#     return JSONResponse(results.json())


if __name__ == "__main__":
    config = uvicorn.Config(
        "kavalai.server:app", port=8000, log_level="debug", reload=True, access_log=True
    )
    server = uvicorn.Server(config)
    server.run()
