import logging
import os
import secrets
from uuid import UUID

import uvicorn
from authlib.integrations.starlette_client import OAuth
from fastapi import FastAPI, Request, HTTPException, status, Body
from kavalai.crud import insert, select, delete, update, get_one
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import JSONResponse, RedirectResponse
from kavalai.backoffice import db
from kavalai.backoffice.db import is_owner, is_member
from kavalai.agents.db import db_manager, Agent
from kavalai.agents.workflow import WorkflowModel
from kavalai.backoffice.svg import generate_workflow_svg
from fastapi.responses import Response

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

# Add SessionMiddleware with a secret key
app.add_middleware(SessionMiddleware, secret_key=secrets.token_urlsafe(16))


async def authenticate_and_sync_user(user_info: dict):
    async with db.AsyncBackofficeSession() as session:
        # Check if user exists in the database.
        email = user_info.get("email")
        stmt = select(db.User).where(db.User.email == email)
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


def assert_logged_in(request: Request):
    if not is_logged_in(request):
        raise HTTPException(status_code=401, detail="Unauthorized.")


def assert_is_admin(request: Request):
    user_session = request.session.get("user_info")
    if not user_session.get("is_admin"):
        raise HTTPException(
            status_code=403, detail="Only administrators can create new projects."
        )


async def assert_is_owner(session: db.AsyncSession, request: Request, project_id: UUID):
    if not await is_owner(
        session, UUID(request.session.get("user_info")["id"]), project_id
    ):
        raise HTTPException(
            status_code=403, detail="Only administrators can create new projects."
        )


async def assert_is_member(
    session: db.AsyncSession, request: Request, project_id: UUID
):
    if not await is_member(
        session, UUID(request.session.get("user_info")["id"]), project_id
    ):
        raise HTTPException(status_code=403, detail="Must be a member of the project.")


async def get_project_and_assert_access(
    request: Request, project_id: UUID
) -> db.Project:
    """Fetch project and assert user is a member."""
    async with db.AsyncBackofficeSession() as session:
        await assert_is_member(session, request, project_id)
        project = await get_one(session, db.Project, project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        return project


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
            "active_project_id": str(db_user.active_project_id),
        }
        return RedirectResponse(url="http://localhost:4200")
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Auth error: {e}")
        raise HTTPException(status_code=400, detail="Authentication failed.")


@app.post("/user/set_active_project/{project_id}")
async def set_active_project(project_id: UUID, request: Request):
    assert_logged_in(request)
    user_id = UUID(request.session.get("user_info")["id"])
    async with db.AsyncBackofficeSession() as session:
        if not await db.is_member(session, user_id, project_id):
            raise HTTPException(
                status_code=403, detail="Must be a member of the project."
            )

        await update(session, db.User, user_id, {"active_project_id": project_id})

        # Update session
        request.session["user_info"]["active_project_id"] = str(project_id)
        return {"status": "ok", "active_project_id": project_id}


@app.post("/projects/create")
async def projects_create(request: Request, data: dict = Body(...)):
    assert_logged_in(request)
    # Check that user is admin.
    user_session = request.session.get("user_info")
    if not user_session.get("is_admin"):
        raise HTTPException(
            status_code=403, detail="Only administrators can create new projects."
        )
    async with db.AsyncBackofficeSession() as session:
        new_project = await insert(session, db.Project, data)
        # Automatically make the creator the owner in ProjectMembership.
        membership_data = {
            "user_id": UUID(user_session["id"]),
            "project_id": new_project.id,
            "role": db.ProjectRole.owner,
        }
        await insert(session, db.ProjectMembership, membership_data)

        return new_project


@app.get("/projects/get/{project_id}")
async def projects_get_by_id(project_id: UUID, request: Request):
    assert_logged_in(request)
    return await get_project_and_assert_access(request, project_id)


@app.get("/projects/all")
async def projects_get_all(request: Request):
    assert_logged_in(request)
    user_id = UUID(request.session.get("user_info")["id"])
    async with db.AsyncBackofficeSession() as session:
        # Instead of db.get_all, use the filtered join query
        projects = await db.get_user_projects(session, user_id)
        return projects


@app.put("/projects/update/{project_id}")
async def projects_update(project_id: UUID, request: Request, data: dict = Body(...)):
    assert_logged_in(request)
    print(data)
    async with db.AsyncBackofficeSession() as session:
        await assert_is_owner(session, request, project_id)
        updated = await update(session, db.Project, project_id, data)
        if not updated:
            raise HTTPException(status_code=404, detail="Project not found.")
        return updated


@app.delete("/projects/delete/{project_id}")
async def projects_delete(project_id: UUID, request: Request):
    assert_logged_in(request)
    async with db.AsyncBackofficeSession() as session:
        await assert_is_owner(session, request, project_id)
        success = await delete(session, db.Project, project_id)
        if not success:
            raise HTTPException(status_code=404, detail="Project not found")
        return {"status": "deleted"}


@app.get("/agents/get/{project_id}/{agent_id}")
async def agents_get_by_id(project_id: UUID, agent_id: UUID, request: Request):
    assert_logged_in(request)
    project = await get_project_and_assert_access(request, project_id)

    # Connect to the project database
    project_session_maker = db_manager.get_sessionmaker(
        user=project.db_user,
        password=project.db_password,
        host=project.db_host,
        port=project.db_port,
        db_name=project.db_name,
    )

    async with project_session_maker() as project_session:
        agent = await get_one(project_session, Agent, agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        return agent


@app.get("/agents/all/{project_id}")
async def agents_get_all(project_id: UUID, request: Request):
    """Fetch all agents belonging to a specific project."""
    assert_logged_in(request)
    project = await get_project_and_assert_access(request, project_id)

    # Connect to the project database
    project_session_maker = db_manager.get_sessionmaker(
        user=project.db_user,
        password=project.db_password,
        host=project.db_host,
        port=project.db_port,
        db_name=project.db_name,
    )

    async with project_session_maker() as project_session:
        stmt = select(Agent)
        result = await project_session.execute(stmt)
        agents = result.scalars().all()
        return agents


@app.get("/agents/svg/{project_id}/{agent_id}")
async def agents_get_svg(project_id: UUID, agent_id: UUID, request: Request):
    """Fetch and return the workflow SVG for a specific agent."""
    assert_logged_in(request)
    project = await get_project_and_assert_access(request, project_id)

    # Connect to the project database
    project_session_maker = db_manager.get_sessionmaker(
        user=project.db_user,
        password=project.db_password,
        host=project.db_host,
        port=project.db_port,
        db_name=project.db_name,
    )

    async with project_session_maker() as project_session:
        agent = await get_one(project_session, Agent, agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        if not agent.workflow:
            raise HTTPException(status_code=400, detail="Agent has no workflow defined")

        model = WorkflowModel(**agent.workflow)
        svg_content = generate_workflow_svg(model, return_content=True)
        return Response(content=svg_content, media_type="image/svg+xml")


@app.post("/projects/test-connection/{project_id}")
async def projects_test_connection(project_id: UUID, request: Request):
    """Test connection to the project database."""
    assert_logged_in(request)
    project = await get_project_and_assert_access(request, project_id)

    try:
        project_session_maker = db_manager.get_sessionmaker(
            user=project.db_user,
            password=project.db_password,
            host=project.db_host,
            port=project.db_port,
            db_name=project.db_name,
        )
        async with project_session_maker() as project_session:
            # Attempt to execute a simple query to verify connection
            from sqlalchemy import text

            await project_session.execute(text("SELECT 1"))
        return {"status": "success", "message": "Connection successful"}
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"status": "error", "message": str(e)},
        )


if __name__ == "__main__":
    config = uvicorn.Config(
        "kavalai.backoffice.server:app",
        port=8000,
        log_level="info",
        reload=True,
        access_log=True,
    )
    server = uvicorn.Server(config)
    server.run()
