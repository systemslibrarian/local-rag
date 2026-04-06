import asyncio
import uuid
from pathlib import Path

import httpx
import nest_asyncio  # type: ignore
import streamlit as st

from app.chat.ui.chat_ui import ChatUI
from internal.config.logging_config import setup_logging
from internal.config.setting import setting
from internal.di.container import Container

nest_asyncio.apply()
setup_logging()

@st.cache_resource
def get_container() -> Container:
    return Container()


async def get_system_health(di: Container) -> dict[str, tuple[bool, str]]:
    health: dict[str, tuple[bool, str]] = {}

    try:
        temp_path = Path(setting.temp_folder)
        temp_path.mkdir(parents=True, exist_ok=True)
        health["temp_folder"] = (True, f"Temp folder ready at {temp_path}")
    except Exception as exc:
        health["temp_folder"] = (False, f"Temp folder unavailable: {exc}")

    try:
        await di.chat_service().all()
        health["database"] = (True, "Database connection is healthy")
    except Exception as exc:
        health["database"] = (False, f"Database unavailable: {exc}")

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{setting.ollama_host}/api/tags")
            response.raise_for_status()
            payload = response.json()
        model_names = {model["name"].split(":", 1)[0] for model in payload.get("models", [])}
        missing_models = [
            model_name
            for model_name in (setting.llm_model, setting.text_embedding_model)
            if model_name not in model_names
        ]
        if missing_models:
            health["ollama"] = (False, f"Ollama is running but missing models: {', '.join(missing_models)}")
        else:
            health["ollama"] = (True, "Ollama and required models are ready")
    except Exception as exc:
        health["ollama"] = (False, f"Ollama unavailable: {exc}")

    return health


def render_system_health(health: dict[str, tuple[bool, str]]) -> None:
    with st.sidebar:
        with st.expander("System Status", expanded=False):
            for _, (is_healthy, message) in health.items():
                if is_healthy:
                    st.success(message)
                else:
                    st.error(message)

async def main() -> None:
    st.set_page_config(
        page_title="Home",
        layout="wide",
        page_icon="👋",
        menu_items={
            'Get Help': 'https://my_website.com/help',
            'Report a bug': "https://my_website.com/bug",
            'About': "# This is a header. This is an *extremely* cool app!"
        }
    )
    st.title("Welcome to Local RAG! 👋")
    query_params = st.query_params
    di = get_container()
    health = await get_system_health(di)
    render_system_health(health)

    if not health.get("database", (False, ""))[0]:
        st.error("Database is unavailable. Fix the connection and reload the app.")
        st.stop()
        return

    if not health.get("ollama", (False, ""))[0]:
        st.warning("Ollama is not fully ready. You can browse chats, but answering questions may fail until models are available.")

    chat_service = di.chat_service()
    file_service = di.file_service()
    message_service = di.message_service()
    ai_service = di.ai_service()
    if "chat_id" in query_params:
        try:
            chat_id = uuid.UUID(query_params["chat_id"])
        except ValueError:
            st.error("Invalid chat ID")
            st.stop()
            return
        await ChatUI.view(
            chat_id=chat_id,
            chat_service=chat_service,
            file_service=file_service,
            message_service=message_service,
            ai_service=ai_service,
        )
    else:
        await ChatUI.list(chat_service=chat_service)



if __name__ == "__main__":
    asyncio.run(main())
