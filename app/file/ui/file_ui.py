import uuid
from io import BytesIO

import streamlit as st

from app.file.model.file import File
from app.file.service.file_service import FileService


@st.fragment(run_every="4s")
async def _jobs_panel(chat_id: uuid.UUID, file_service: FileService) -> None:
    """Auto-refreshing jobs panel — re-runs every 4 s via st.fragment."""
    jobs = await file_service.list_jobs(chat_id)
    has_active = any(j.status in {"queued", "running"} for j in jobs)

    with st.expander("⏱️ Index Jobs", expanded=has_active):
        if jobs:
            refresh_col, clear_col, _ = st.columns([1, 1, 2])
            with refresh_col:
                if st.button("Refresh", key=f"refresh_jobs_{chat_id}", use_container_width=True):
                    st.rerun()
            with clear_col:
                if st.button("Clear", key=f"clear_jobs_{chat_id}", use_container_width=True):
                    deleted = await file_service.clear_finished_jobs(chat_id)
                    if deleted:
                        st.toast(f"Cleared {deleted} finished jobs")
                    st.rerun()
            for job in jobs:
                label = f"{job.file_name} · {job.status}"
                if job.status == "completed":
                    st.success(f"{label} — {job.message}")
                    if job.chunks:
                        st.caption(
                            f"{job.pages} pages · {job.chunks} chunks · "
                            f"updated {job.updated_at.strftime('%H:%M:%S')}"
                        )
                elif job.status == "failed":
                    st.error(f"{label} — {job.message}")
                elif job.status == "running":
                    st.warning(f"{label} — {job.message}")
                else:
                    st.info(f"{label} — {job.message}")
            if has_active:
                st.caption("Updating automatically while jobs are running.")
        else:
            st.info("No indexing jobs yet.")


class FileUI:

    @staticmethod
    async def view(
        chat_id: uuid.UUID,
        file_service: FileService
    ) -> None:
        files = await file_service.all(conditions={"chat_id": chat_id})
        st.header("📁 Upload")
        st.caption("Upload a PDF to index it for retrieval and cited answers.")
        if files:
            st.caption(f"Indexed documents: {len(files)}")
        with st.form(f"upload_form_{chat_id}"):
            uploaded_file = st.file_uploader(
                "📎 File",
                type=["pdf"],
                help="Allowed file types: pdf"
            )
            submitted = st.form_submit_button("Start Indexing", type="primary", use_container_width=True)

        if submitted:
            if uploaded_file is None:
                st.error("Select a PDF before starting indexing.")
            else:
                job = await file_service.submit_upload_job(
                    file_name=uploaded_file.name,
                    pdf_bytes=uploaded_file.getvalue(),
                    chat_id=chat_id,
                )
                if job.status == "completed" and job.file_id is not None and job.pages == 0 and job.chunks == 0:
                    st.info(job.message)
                else:
                    st.toast(f"Started indexing {uploaded_file.name}")
                    st.rerun()

        # Jobs panel — auto-refreshes every 4 s via st.fragment (no JS reload)
        _jobs_panel(chat_id=chat_id, file_service=file_service)

        with st.expander("🗂️ Manage Documents", expanded=bool(files)):
            if files:
                for file in files:
                    meta_col, action_col = st.columns([3, 2])
                    with meta_col:
                        st.markdown(f"**{file.name}**")
                        st.caption(f"Added {file.created_at.strftime('%d.%m.%Y %H:%M')}")
                    with action_col:
                        reindex_key = f"reindex_{file.id}"
                        delete_key = f"delete_{file.id}"
                        if st.button("Re-index", key=reindex_key, use_container_width=True):
                            with st.spinner(f"Re-indexing {file.name}..."):
                                try:
                                    result = await file_service.reindex(file.id)
                                except Exception as exc:
                                    st.error(f"Failed to re-index {file.name}: {exc}")
                                else:
                                    st.toast(f"{result.message} {result.chunks} chunks ready.")
                                    st.rerun()
                        if st.button("Delete", key=delete_key, use_container_width=True):
                            try:
                                await file_service.delete(file.id)
                            except Exception as exc:
                                st.error(f"Failed to delete {file.name}: {exc}")
                            else:
                                st.toast(f"Deleted {file.name}")
                                st.rerun()
                    st.markdown("---")
            else:
                st.info("No indexed documents yet.")

        with st.expander("👁️ Preview"):
            if files:
                for file in files:
                    FileUI.file_preview(file, file_service)
            else:
                st.info("No documents uploaded yet.")
        st.markdown("---")

    @staticmethod
    def file_preview(
            file: File,
            file_service: FileService
    ) -> None:
        try:
            img = file_service.pdf_to_image(file.storage_path, only_first_page=True)
            if img and img[0].width > 0 and img[0].height > 0:
                img_byte_arr = BytesIO()
                img[0].save(img_byte_arr, format='JPEG')
                img_byte_arr.seek(0)
                st.image(img_byte_arr, caption=f"First page of {file.name}", use_container_width=True)
            else:
                st.info(f"📄 {file.name} (preview not available)")
        except Exception:
            st.info(f"📄 {file.name} (preview not available)")

