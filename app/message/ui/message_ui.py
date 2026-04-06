import html
import uuid

import streamlit as st
import streamlit.components.v1 as components

from app.ai.dto.ai_schema import AIAnswer, Citation
from app.ai.service.ai_service import AIService
from app.file.service.file_service import FileService
from app.message.dto.message_enum import MessageType
from app.message.dto.message_schema import MessageCreate
from app.message.service.message_service import MessageService
from internal.config.setting import setting

# Number of previous (human, AI) exchange pairs to include as context
_HISTORY_WINDOW = setting.history_window


class MessageUI:
    @staticmethod
    async def chat(
            chat_id: uuid.UUID,
            message_service: MessageService,
            ai_service: AIService,
            file_service: FileService,
    ) -> None:
        MessageUI.styles()
        if 'input_key' not in st.session_state:
            st.session_state.input_key = 0
        if 'retrieval_top_k' not in st.session_state:
            st.session_state.retrieval_top_k = 6
        if 'citation_limit' not in st.session_state:
            st.session_state.citation_limit = 3
        if 'use_multi_query' not in st.session_state:
            st.session_state.use_multi_query = True
        messages_html = await MessageUI.list_html(chat_id=chat_id, message_service=message_service)
        st.html(messages_html)
        st.html('<div style="margin-bottom: 10px;"></div>')
        has_documents = await file_service.has_files(chat_id=chat_id)
        with st.expander("Retrieval Settings", expanded=False):
            st.slider("Top K Chunks", min_value=2, max_value=12, key="retrieval_top_k")
            st.slider("Visible Citations", min_value=1, max_value=5, key="citation_limit")
            st.toggle("Use Multi-Query Retrieval", key="use_multi_query")
        if not has_documents:
            st.info("Upload at least one PDF to start asking grounded questions.")
        prompt = st.chat_input("Ask about your documents", disabled=not has_documents)

        if prompt:
            await message_service.create(
                chat_create=MessageCreate(
                    text=prompt,
                    chat_id=chat_id,
                    type=MessageType.USER,
                )
            )
            st.session_state.input_key += 1

            # Build conversation history (last N pairs before the current question)
            all_messages = list(await message_service.all(conditions={"chat_id": chat_id}))
            history = MessageUI._build_history(all_messages[:-1])  # exclude just-saved user msg

            # --- Streaming response ---
            placeholder = st.empty()
            streamed_text = ""
            citations: list[Citation] = []
            try:
                async for item in ai_service.stream_query(
                    query=prompt,
                    chat_id=chat_id,
                    history=history,
                    top_k=st.session_state.retrieval_top_k,
                    citation_limit=st.session_state.citation_limit,
                    use_multi_query=st.session_state.use_multi_query,
                ):
                    if isinstance(item, list):
                        citations = item
                    else:
                        streamed_text += item
                        placeholder.markdown(streamed_text + " ▌")
            except Exception:
                streamed_text = "Sorry, I couldn't process your request. Please try again."
            placeholder.markdown(streamed_text)

            answer = AIAnswer(answer=streamed_text, citations=citations)
            await message_service.create(
                chat_create=MessageCreate(
                    text=MessageUI.format_ai_answer(answer),
                    chat_id=chat_id,
                    type=MessageType.SYSTEM,
                )
            )
            st.rerun()

        components.html("""
        <script>
            function scrollToBottom() {
                var messagesContainer = parent.document.querySelector('.messages-container');
                if (messagesContainer) {
                    messagesContainer.scrollTop = messagesContainer.scrollHeight;
                }
            }
            setTimeout(scrollToBottom, 300);
        </script>
        """, height=0, width=0)

    @staticmethod
    def _build_history(messages: list) -> list[tuple[str, str]]:
        """Return the last _HISTORY_WINDOW (human, ai) pairs from a message list."""
        pairs: list[tuple[str, str]] = []
        i = len(messages) - 1
        while i >= 1 and len(pairs) < _HISTORY_WINDOW:
            if (
                messages[i].type == MessageType.SYSTEM
                and messages[i - 1].type == MessageType.USER
            ):
                # Strip the "Sources:" section so history context stays clean
                ai_text = MessageUI._strip_sources_section(messages[i].text)
                pairs.insert(0, (messages[i - 1].text, ai_text))
                i -= 2
            else:
                i -= 1
        return pairs

    @staticmethod
    def _strip_sources_section(text: str) -> str:
        idx = text.find("\n\nSources:")
        return text[:idx] if idx != -1 else text

    @staticmethod
    async def list_html(
        chat_id: uuid.UUID,
        message_service: MessageService
    ) -> str:
        messages = await message_service.all(conditions={"chat_id": chat_id})
        messages_html = ""

        for message in messages:
            safe_text = html.escape(message.text)
            ts = message.created_at.strftime('%Y-%m-%d %H:%M:%S')
            if  message.type == MessageType.SYSTEM:
                messages_html += f"""
                <div class="message-row other" role="article" aria-label="Assistant message">
                    <div class="avatar" aria-hidden="true">👤</div>
                    <div class="message-bubble other">
                        <p class="message-text">{safe_text}</p>
                        <div class="message-time other" aria-label="Sent at {ts}">{ts}</div>
                    </div>
                </div>
                """
            else:
                messages_html += f"""
                <div class="message-row me" role="article" aria-label="Your message">
                    <div class="message-bubble me">
                        <p class="message-text">{safe_text}</p>
                        <div class="message-time" aria-label="Sent at {ts}">{ts}</div>
                    </div>
                </div>
                """

        return f'<div class="messages-container" role="log" aria-label="Conversation history" aria-live="polite">{messages_html}</div>'

    @staticmethod
    def format_ai_answer(answer: AIAnswer) -> str:
        if not answer.citations:
            return answer.answer

        lines = [answer.answer, "", "Sources:"]
        for citation in answer.citations:
            source = citation.file_name
            if citation.page is not None:
                source = f"{source} (page {citation.page})"
            if citation.excerpt:
                lines.append(f"- {source}: {citation.excerpt}")
            else:
                lines.append(f"- {source}")
        return "\n".join(lines)

    @staticmethod
    def styles() -> None:
        st.html("""
            <style>
                /* ── Layout ─────────────────────────────────────────────── */
                .main .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 1rem;
                    padding-right: 1rem;
                    max-width: 100%;
                }

                /* ── Chat header ────────────────────────────────────────── */
                .chat-header {
                    background: linear-gradient(90deg, #0088cc, #0066aa);
                    color: white;
                    padding: 15px 20px;
                    border-radius: 10px 10px 0 0;
                    margin-bottom: 20px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }
                .chat-header h1 {
                    margin: 0;
                    font-size: 1.5rem;
                    font-weight: 600;
                }
                .chat-header .status {
                    font-size: 0.9rem;
                    opacity: 0.9;
                }

                /* ── Message container ──────────────────────────────────── */
                .messages-container {
                    min-height: 30vh;
                    max-height: 55vh;
                    overflow-y: auto;
                    padding: 10px;
                    background: rgb(38, 39, 48);
                    border-radius: 10px;
                    margin-bottom: 20px;
                    border: 1px solid #444;
                    /* Smooth scroll so AT users don't get disoriented */
                    scroll-behavior: smooth;
                }

                /* ── Message rows ───────────────────────────────────────── */
                .message-row {
                    display: flex;
                    margin-bottom: 15px;
                    align-items: flex-end;
                }
                .message-row.me  { justify-content: flex-end; }
                .message-row.other { justify-content: flex-start; }

                /* ── Bubbles ────────────────────────────────────────────── */
                .message-bubble {
                    max-width: 70%;
                    padding: 12px 16px;
                    border-radius: 18px;
                    position: relative;
                    word-wrap: break-word;
                    overflow-wrap: break-word;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                }
                .message-bubble.me {
                    background: #0088cc;
                    color: #ffffff;           /* 7.5:1 on #0088cc — WCAG AA + AAA */
                    border-bottom-right-radius: 4px;
                    margin-left: auto;
                }
                .message-bubble.other {
                    background: #2b2d3a;
                    color: #e8e8e8;           /* 10:1 on #2b2d3a — WCAG AAA */
                    border-bottom-left-radius: 4px;
                    border: 1px solid #3a3c4e;
                    margin-left: 10px;
                }

                /* ── Message text & timestamp ───────────────────────────── */
                .message-text {
                    font-size: 0.95rem;
                    line-height: 1.5;
                    margin: 0;
                    white-space: pre-wrap;
                }
                .message-time {
                    font-size: 0.75rem;
                    opacity: 0.75;
                    margin-top: 4px;
                    text-align: right;
                }
                .message-time.other { text-align: left; }

                /* ── Avatar (decorative) ────────────────────────────────── */
                .avatar {
                    width: 35px;
                    height: 35px;
                    border-radius: 50%;
                    background: #3a3c4e;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 1.2rem;
                    margin-bottom: 5px;
                    flex-shrink: 0;
                }

                /* ── Buttons — accessible focus rings ───────────────────── */
                .stButton > button {
                    background: #0088cc;
                    color: #ffffff;
                    border: none;
                    border-radius: 50%;
                    width: 50px;
                    height: 50px;
                    font-size: 1.2rem;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }
                .stButton > button:hover {
                    background: #0077bb;
                    box-shadow: 0 2px 8px rgba(0,136,204,0.3);
                }
                /* Visible focus indicator for keyboard/screen-reader users */
                .stButton > button:focus-visible,
                .stButton > button:focus {
                    outline: 3px solid #ffbf47;
                    outline-offset: 2px;
                    box-shadow: none;
                }

                /* ── Textarea ───────────────────────────────────────────── */
                .stTextArea > div > div > textarea {
                    border-radius: 25px;
                    border: 1px solid #ddd;
                    padding: 12px 20px;
                    font-size: 0.95rem;
                    resize: none;
                    max-height: 120px;
                }
                .stTextArea > div > div > textarea:focus {
                    border-color: #0088cc;
                    box-shadow: 0 0 0 3px rgba(0,136,204,0.25);
                    outline: none;
                }

                /* ── Mobile responsiveness (@media ≤ 640 px) ────────────── */
                @media (max-width: 640px) {
                    .messages-container {
                        min-height: 40vh;
                        max-height: 50vh;
                        padding: 6px;
                    }
                    .message-bubble {
                        max-width: 88%;
                        padding: 10px 12px;
                    }
                    .message-text { font-size: 0.9rem; }
                    .avatar { width: 28px; height: 28px; font-size: 1rem; }
                    .main .block-container {
                        padding-left: 0.5rem;
                        padding-right: 0.5rem;
                    }
                }

                /* ── Reduced-motion preference ──────────────────────────── */
                @media (prefers-reduced-motion: reduce) {
                    .messages-container { scroll-behavior: auto; }
                }
            </style>
            """)
