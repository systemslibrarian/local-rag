import uuid

import streamlit as st

from app.ai.service.ai_service import AIService
from app.chat.dto.chat_schema import ChatCreate, ChatUpdate
from app.chat.service.chat_service import ChatService
from app.file.service.file_service import FileService
from app.file.ui.file_ui import FileUI
from app.message.dto.message_enum import MessageType
from app.message.service.message_service import MessageService
from app.message.ui.message_ui import MessageUI


class ChatUI:

    @staticmethod
    async def view(
        chat_id: uuid.UUID,
        chat_service: ChatService,
        file_service: FileService,
        message_service: MessageService,
        ai_service: AIService,
    ) -> None:
        try:
            chat = await chat_service.get_by_id(chat_id)
        except ValueError:
            st.error("Chat not found")
            st.stop()
            return
        files = await file_service.all(conditions={"chat_id": chat_id})
        messages = await message_service.all(conditions={"chat_id": chat_id})
        jobs = await file_service.list_jobs(chat_id)
        active_jobs = sum(1 for job in jobs if job.status in {"queued", "running"})
        st.title(f"💬 {chat.name}")
        st.markdown("---")

        with st.sidebar:
            st.subheader("Workspace")
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric("Docs", len(files))
                st.metric("Jobs", len(jobs))
            with metric_col2:
                st.metric("Messages", len(messages))
                st.metric("Active", active_jobs)
            st.markdown("---")

            await FileUI.view(
                chat_id=chat_id,
                file_service=file_service
            )

            # --- Export conversation ---
            if messages:
                st.markdown("---")
                st.subheader("📥 Export")
                md_lines = [f"# {chat.name}\n"]
                for msg in messages:
                    ts = msg.created_at.strftime("%Y-%m-%d %H:%M")
                    label = "**You**" if msg.type == MessageType.USER else "**Assistant**"
                    md_lines.append(f"### {label}  \n_{ts}_\n\n{msg.text}\n")
                md_export = "\n---\n\n".join(md_lines)
                safe_name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in chat.name)
                st.download_button(
                    label="Download conversation (.md)",
                    data=md_export.encode("utf-8"),
                    file_name=f"{safe_name}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

            st.subheader("🔧 Chat Available:")
            st.markdown("""
            - 📎 Upload PDF files
            - 🔍 Automated content analysis
            - 💬 Discussion of file content
            - 💾 Save chat history
            """)

        await MessageUI.chat(
            chat_id=chat_id,
            message_service=message_service,
            ai_service=ai_service,
            file_service=file_service,
        )


    @staticmethod
    async def list(
        chat_service: ChatService
    ) -> None:
        st.title("💬 Chat Manager")
        st.markdown("---")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.header("📋 Chats")

            chats = await chat_service.all()
            search_query = st.text_input("Search chats", placeholder="Filter by name")
            sort_order = st.segmented_control(
                "Sort",
                options=["Newest", "Oldest", "A-Z"],
                default="Newest",
                selection_mode="single",
            )

            filtered_chats = list(chats)
            if search_query:
                query = search_query.strip().lower()
                filtered_chats = [chat for chat in filtered_chats if query in chat.name.lower()]

            if sort_order == "Oldest":
                filtered_chats.sort(key=lambda chat: chat.created_at)
            elif sort_order == "A-Z":
                filtered_chats.sort(key=lambda chat: chat.name.lower())
            else:
                filtered_chats.sort(key=lambda chat: chat.created_at, reverse=True)

            if not chats:
                st.info("No chats yet")
            elif not filtered_chats:
                st.info("No chats match your filter")
            else:
                for chat in filtered_chats:
                    with st.container():
                        chat_col1, chat_col2, chat_col3, chat_col4 = st.columns([3, 1, 1, 1])

                        with chat_col1:
                            st.subheader(f"💬 {chat.name}")
                            st.caption(f"Created: {chat.created_at.strftime('%d.%m.%Y %H:%M')}")

                        with chat_col2:
                            if st.button("Open", key=f"open_{chat.id}"):
                                st.query_params["chat_id"] = str(chat.id)
                                st.rerun()

                        with chat_col3:
                            if st.button("Rename", key=f"rename_toggle_{chat.id}"):
                                st.session_state[f"rename_chat_{chat.id}"] = not st.session_state.get(f"rename_chat_{chat.id}", False)

                        with chat_col4:
                            if st.button("🗑️", key=f"delete_{chat.id}", help="Remove"):
                                await chat_service.delete(chat.id)
                                st.rerun()

                        if st.session_state.get(f"rename_chat_{chat.id}", False):
                            with st.form(f"rename_chat_form_{chat.id}"):
                                new_name = st.text_input("Rename chat", value=chat.name, key=f"rename_input_{chat.id}")
                                submitted = st.form_submit_button("Save", type="primary")
                                if submitted:
                                    candidate = new_name.strip()
                                    if not candidate:
                                        st.error("Enter chat name")
                                    elif candidate == chat.name:
                                        st.info("Chat name is unchanged")
                                    else:
                                        existing = await chat_service.find_by_name(candidate)
                                        if existing is not None and existing.id != chat.id:
                                            st.error(f"Chat '{candidate}' already exists. Please choose a different name.")
                                        else:
                                            await chat_service.rename(chat.id, ChatUpdate(name=candidate))
                                            st.session_state[f"rename_chat_{chat.id}"] = False
                                            st.toast(f"Renamed chat to '{candidate}'")
                                            st.rerun()

                        st.markdown("---")

        with col2:
            st.header("➕ Create a new chat")

            with st.form("new_chat_form"):
                chat_name = st.text_input("Name", placeholder="Enter chat name")
                if st.form_submit_button("Submit", type="primary"):
                    if chat_name:
                        existing = await chat_service.find_by_name(chat_name)
                        if existing is not None:
                            st.error(f"Chat '{chat_name}' already exists. Please choose a different name.")
                        else:
                            await chat_service.create(ChatCreate(name=chat_name))
                            st.toast(f"Chat '{chat_name}' created!")
                            st.rerun()
                    else:
                        st.error("Enter chat name")
