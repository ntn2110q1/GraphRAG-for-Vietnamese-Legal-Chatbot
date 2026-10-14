import os
from html import escape
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from frontend.api_client import ApiClient, ApiError

load_dotenv()
st.set_page_config(page_title="Legal GraphRAG", page_icon="⚖️", layout="wide", initial_sidebar_state="expanded")

NAVY = "#14223D"
BLUE = "#6559E8"
TEAL = "#21B9A3"
BG = "#F4F6FB"
THEME = st.session_state.get("theme_mode", "Sáng")
IS_DARK = THEME == "Tối"
colors = {
    "app": "#0C1425" if IS_DARK else BG,
    "surface": "#17233A" if IS_DARK else "#FFFFFF",
    "surface_soft": "#1E2C45" if IS_DARK else "#F8FAFE",
    "text": "#EAF0FB" if IS_DARK else NAVY,
    "muted": "#A9B8D0" if IS_DARK else "#61718A",
    "border": "#30405B" if IS_DARK else "#E4EAF3",
    "shadow": "rgba(0, 0, 0, .18)" if IS_DARK else "rgba(32, 48, 81, .08)",
}
st.markdown(
    "<style>:root {"
    + ";".join(f"--{name}:{value}" for name, value in colors.items())
    + f";--navy:{NAVY};--accent:{BLUE};--mint:{TEAL};"
    + "}</style>",
    unsafe_allow_html=True,
)
st.markdown(f"<style>{(Path(__file__).with_name('styles.css')).read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
if IS_DARK:
    st.markdown(f"<style>{(Path(__file__).with_name('styles_dark.css')).read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

if "admin_token" not in st.session_state:
    st.session_state.admin_token = os.getenv("ADMIN_API_TOKEN", "")
if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = None
if "chat_state" not in st.session_state:
    st.session_state.chat_state = None

with st.sidebar:
    st.markdown(
        "<div class='brand'><div class='brand-icon'>§</div><div><div class='brand-name'>Legal <strong>GraphRAG</strong></div>"
        "<div class='brand-caption'>LEGAL INTELLIGENCE</div></div></div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='sidebar-label'>Không gian làm việc</div>", unsafe_allow_html=True)
    nav_labels = {
        "Tổng quan": "◈  Tổng quan",
        "Trợ lý pháp luật": "✦  Trợ lý pháp luật",
        "Knowledge Graph": "◎  Knowledge Graph",
        "Văn bản pháp luật": "▤  Văn bản pháp luật",
        "So sánh retrieval": "⇄  So sánh retrieval",
        "Đánh giá": "▧  Đánh giá",
        "Cấu hình": "⚙  Cấu hình",
    }
    page = st.radio("Điều hướng", list(nav_labels), format_func=nav_labels.get, label_visibility="collapsed")
    st.divider()
    with st.expander("Giao diện & kết nối", expanded=False):
        st.selectbox("Chế độ hiển thị", ["Sáng", "Tối"], key="theme_mode")
        token = st.text_input("Admin API token", value=st.session_state.admin_token, type="password", help="Token chỉ gửi đến backend local để gọi chức năng quản trị.")
        if token != st.session_state.admin_token:
            st.session_state.admin_token = token
        api_url = st.text_input("Backend URL", value=os.getenv("LEGAL_API_URL", "http://127.0.0.1:8000"))
    api = ApiClient(api_url, st.session_state.admin_token)
    try:
        health = api.get("/health", timeout=4)
        model_ready = health.get("llm", {}).get("available", False)
        status_text = "Đã kết nối · LLM sẵn sàng" if model_ready else "Đã kết nối · đang tra cứu nguồn"
        st.markdown(f"<div class='status-pill'><span class='status-dot'></span>{status_text}</div>", unsafe_allow_html=True)
    except ApiError as exc:
        health = None
        st.markdown("<div class='status-pill offline'><span class='status-dot'></span>Backend chưa kết nối</div>", unsafe_allow_html=True)
        st.caption(str(exc))
    if page == "Trợ lý pháp luật" and health:
        st.markdown("<div class='sidebar-label'>Hội thoại gần đây</div>", unsafe_allow_html=True)
        if st.button("＋ Cuộc trò chuyện mới", key="new_chat", use_container_width=True, type="primary"):
            st.session_state.active_session_id = None
            st.session_state.chat_state = None
            st.rerun()
        try:
            sessions = api.get("/chat/sessions")
        except ApiError as exc:
            sessions = []
            st.caption(str(exc))
        for session in sessions[:20]:
            label = (session["title"] or "Cuộc trò chuyện")[:25]
            if st.button(
                f"◦  {label}", key=f"session_{session['id']}", use_container_width=True,
                type="primary" if session["id"] == st.session_state.active_session_id else "secondary",
            ):
                st.session_state.active_session_id = session["id"]
                try:
                    st.session_state.chat_state = api.get(f"/chat/sessions/{session['id']}")
                except ApiError as exc:
                    st.error(str(exc))
                st.rerun()
        if st.session_state.active_session_id:
            with st.expander("Quản lý cuộc trò chuyện"):
                rename = st.text_input("Đổi tên", key="rename_chat")
                if st.button("Lưu tên") and rename.strip():
                    try:
                        api.patch(f"/chat/sessions/{st.session_state.active_session_id}", json={"title": rename.strip()})
                        st.rerun()
                    except ApiError as exc:
                        st.error(str(exc))
                if st.button("Xóa cuộc trò chuyện", type="secondary"):
                    try:
                        api.delete(f"/chat/sessions/{st.session_state.active_session_id}")
                        st.session_state.active_session_id = None
                        st.session_state.chat_state = None
                        st.rerun()
                    except ApiError as exc:
                        st.error(str(exc))
    st.markdown("<div class='sidebar-foot'>Tra cứu văn bản pháp luật Việt Nam<br>Nguồn dữ liệu do quản trị viên nạp</div>", unsafe_allow_html=True)


def hero(title: str, subtitle: str, tag: str = "LEGAL AI WORKSPACE", compact: bool = False):
    hero_class = "hero hero-compact" if compact else "hero"
    meta = "" if compact else (
        "<div class='hero-meta'><span>✦ Truy xuất có căn cứ</span>"
        "<span>◎ Đồ thị tri thức</span><span>▤ Nguồn minh bạch</span></div>"
    )
    st.markdown(
        f"<div class='{hero_class}'><div class='eyebrow'>{escape(tag)}</div><h1>{escape(title)}</h1>"
        f"<p>{escape(subtitle)}</p>{meta}</div>",
        unsafe_allow_html=True,
    )


def metric(label: str, value, hint: str = "", icon: str = "◈"):
    st.markdown(
        f"<div class='metric-card'><div class='metric-icon'>{escape(icon)}</div>"
        f"<div class='metric-label'>{escape(label)}</div><div class='metric-value'>{escape(str(value))}</div>"
        f"<div class='muted'>{escape(hint)}</div></div>",
        unsafe_allow_html=True,
    )


def show_api_error(exc: Exception):
    st.error(str(exc))


def display_sources(sources: list[dict]):
    if not sources:
        st.markdown(
            "<div class='source-empty'><span>◎</span><strong>Chưa có nguồn trích dẫn</strong>"
            "<small>Nguồn sẽ xuất hiện ở đây khi có kết quả phù hợp.</small></div>",
            unsafe_allow_html=True,
        )
        return
    for source in sources:
        title = f"[{source.get('citation_id', 'Nguồn')}] {source.get('document_title', 'Văn bản')}"
        article = source.get("location_label") or (f"Điều {source['article_number']}" if source.get("article_number") else "Đoạn văn bản")
        status = source.get("validity_status", "unverified")
        with st.expander(f"{title} · {article} · hiệu lực: {status}"):
            st.caption(f"Số hiệu: {source.get('document_number') or 'Chưa khai báo'} · nguồn: {source.get('source_url') or source.get('source_name') or 'Tài liệu tải lên'}")
            if source.get("source_url"):
                st.markdown(f"[Mở nguồn gốc]({source['source_url']})")
            st.write(source.get("text", ""))


def dashboard():
    hero("Trung tâm tra cứu pháp luật", "Tìm câu trả lời có căn cứ trong văn bản đã nạp, kèm dấu vết từ đồ thị tri thức.")
    if not health:
        st.warning("Khởi động backend để xem số liệu thực từ SQLite.")
        return
    try:
        stats = api.get("/stats/overview")
        documents = api.get("/documents?limit=500")
    except ApiError as exc:
        show_api_error(exc)
        return
    cols = st.columns(4, gap="medium")
    metrics = [
        ("Văn bản", "documents_processed", "Nguồn đã xử lý", "▤"),
        ("Đoạn văn bản", "chunks", "Đã lập chỉ mục", "◫"),
        ("Thực thể", "entities", "Trong đồ thị tri thức", "◎"),
        ("Quan hệ", "relationships", "Liên kết có nguồn", "⇄"),
    ]
    for col, (label, key, hint, icon) in zip(cols, metrics):
        with col:
            metric(label, stats.get(key, 0), hint, icon)
    st.write("")
    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Trạng thái kho tài liệu")
        status_counts = pd.Series([doc["processing_status"] for doc in documents], dtype="string").value_counts()
        if status_counts.empty:
            st.info("Kho tài liệu đang trống. Nạp một văn bản pháp luật có nguồn rõ ràng để bắt đầu.")
        else:
            chart = go.Figure(go.Pie(labels=status_counts.index.tolist(), values=status_counts.values.tolist(), hole=.62, marker_colors=[BLUE, TEAL, "#F0A44B", "#d76565"]))
            chart.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), showlegend=True, paper_bgcolor="rgba(0,0,0,0)", font=dict(color=colors["text"]))
            st.plotly_chart(chart, use_container_width=True)
    with right:
        st.subheader("Dịch vụ đang hoạt động")
        llm = stats.get("llm", {})
        embed = stats.get("embeddings", {})
        services = [
            ("LLM", "Sẵn sàng · " + str(llm.get("model")) if llm.get("available") else "Chưa kết nối · trả đoạn nguồn"),
            ("Truy xuất", str(embed.get("backend", "BM25 lexical retrieval"))),
            ("Đánh giá", f"{stats.get('evaluation_cases', 0)} câu hỏi gán nhãn"),
        ]
        for name, detail in services:
            st.markdown(
                f"<div class='service-row'><span class='service-bullet'></span><span><strong>{escape(name)}</strong>"
                f"<small>{escape(detail)}</small></span></div>", unsafe_allow_html=True,
            )
        st.caption("Chỉ số accuracy, recall và citation sẽ xuất hiện sau khi có bộ đánh giá gắn với các chunk thật.")
        if stats.get("documents_failed"):
            st.error(f"{stats['documents_failed']} tài liệu xử lý lỗi. Xem chi tiết trong mục Văn bản pháp luật.")
    st.subheader("Tài liệu gần đây")
    if documents:
        st.dataframe(pd.DataFrame(documents)[["title", "document_number", "processing_status", "validity_status", "chunk_count"]].head(8), use_container_width=True, hide_index=True)
    else:
        st.caption("Chưa có tài liệu nào được nạp.")


def chat_page():
    hero("Legal AI Assistant", "Đặt câu hỏi bằng tiếng Việt. Mỗi câu trả lời đều đi kèm nguồn và trạng thái hiệu lực được khai báo.", "HỎI ĐÁP CÓ BẰNG CHỨNG", compact=True)
    if not health:
        st.warning("Backend chưa kết nối.")
        return
    main, sources_col = st.columns([3.4, 1.35], gap="large")
    with main:
        with st.expander("Tùy chọn truy xuất", expanded=False):
            mode = st.selectbox("Phương thức truy xuất", ["graphrag", "vector"], format_func=lambda v: "GraphRAG · vector + đồ thị" if v == "graphrag" else "Vector RAG · baseline", key="chat_mode")
            top_k = st.slider("Số đoạn bằng chứng tối đa", 1, 12, 5, key="chat_top_k")
        if st.session_state.active_session_id:
            try:
                if not st.session_state.chat_state or st.session_state.chat_state.get("id") != st.session_state.active_session_id:
                    st.session_state.chat_state = api.get(f"/chat/sessions/{st.session_state.active_session_id}")
            except ApiError as exc:
                show_api_error(exc)
        state = st.session_state.chat_state or {}
        messages = state.get("messages", [])
        if not messages:
            st.markdown(
                "<div class='empty-state'><div class='empty-icon'>✦</div>"
                "<h3>Bắt đầu với một câu hỏi pháp luật</h3>"
                "<p>Trợ lý tìm trong văn bản đã nạp, nối thông tin từ đồ thị tri thức và hiển thị nguồn để bạn đối chiếu.</p>"
                "</div>", unsafe_allow_html=True,
            )
            st.markdown("<div class='section-label'>Gợi ý câu hỏi</div>", unsafe_allow_html=True)
            suggestion_cols = st.columns(2, gap="small")
            for col, sample in zip(suggestion_cols, ["Điều 5 quy định những quyền nào của người lao động?", "Người lao động được nghỉ hằng năm bao nhiêu ngày?"]):
                with col:
                    if st.button(sample, key=f"suggest_{sample}", use_container_width=True):
                        st.session_state.pending_question = sample
        for message in messages:
            with st.chat_message("user" if message["role"] == "user" else "assistant"):
                st.markdown(message["content"])
                if message["role"] == "assistant" and message.get("citation_details"):
                    st.caption("Nguồn được đính kèm trong cuộc trò chuyện")
                if message["role"] == "assistant":
                    fb1, fb2, _ = st.columns([.6, .6, 5])
                    if fb1.button("👍", key=f"up_{message['id']}"):
                        try:
                            api.patch(f"/chat/messages/{message['id']}/feedback", json={"value": 1})
                        except ApiError:
                            pass
                    if fb2.button("👎", key=f"down_{message['id']}"):
                        try:
                            api.patch(f"/chat/messages/{message['id']}/feedback", json={"value": -1})
                        except ApiError:
                            pass
        prompt = st.chat_input("Hỏi về văn bản pháp luật…") or st.session_state.pop("pending_question", None)
        if prompt:
            with st.spinner("Đang tìm văn bản và mở rộng quan hệ đồ thị…"):
                try:
                    response = api.post("/chat/ask", json={"question": prompt, "session_id": st.session_state.active_session_id, "retrieval_mode": mode, "top_k": top_k}, timeout=150)
                    st.session_state.active_session_id = response["session_id"]
                    st.session_state.chat_state = api.get(f"/chat/sessions/{response['session_id']}")
                    st.session_state.chat_state["active_response"] = response
                    st.rerun()
                except ApiError as exc:
                    st.error(str(exc))
        active = state.get("active_response")
        if active:
            st.caption(f"{active.get('generation_mode')} · {active.get('metrics', {}).get('retrieval_backend')} · {active.get('metrics', {}).get('graph_relationships_used', 0)} quan hệ đồ thị đã duyệt")
            for warning in active.get("warnings", []):
                st.warning(warning)
    with sources_col:
        st.subheader("Nguồn đối chiếu")
        st.markdown(
            "<div class='source-intro'><strong>Căn cứ cho câu trả lời</strong><br>"
            "Mở từng đoạn để đọc nguyên văn, số hiệu và tình trạng hiệu lực được khai báo.</div>",
            unsafe_allow_html=True,
        )
        active = (st.session_state.chat_state or {}).get("active_response")
        if active:
            if active.get("citations"):
                st.caption("Các nguồn được gắn với câu trả lời")
                display_sources(active.get("citations", []))
            else:
                st.caption("Đoạn đã truy xuất; chưa có citation ID được xác minh từ câu trả lời")
                display_sources(active.get("retrieved_evidence", []))
        else:
            chat_state = st.session_state.chat_state or {}
            last_assistant = next((item for item in reversed(chat_state.get("messages", [])) if item["role"] == "assistant"), None)
            display_sources(last_assistant.get("citation_details", []) if last_assistant else [])


def graph_page():
    hero("Knowledge Graph Explorer", "Khám phá thực thể và quan hệ được trích từ tài liệu đã lập chỉ mục.", "ĐỒ THỊ TRI THỨC")
    if not health:
        st.warning("Backend chưa kết nối.")
        return
    query = st.text_input("Tìm thực thể", placeholder="Ví dụ: Điều 5, tiền lương, thời giờ nghỉ ngơi")
    try:
        params = f"?q={requests_quote(query)}&limit=500" if query else "?limit=500"
        nodes_payload = api.get(f"/graph/nodes{params}")
        rel_payload = api.get(f"/graph/relationships{params}")
    except ApiError as exc:
        show_api_error(exc)
        return
    nodes, relations = nodes_payload["nodes"], rel_payload["relationships"]
    counts = nodes_payload.get("counts", {})
    m1, m2 = st.columns(2)
    m1.metric("Node trong kho", counts.get("nodes", len(nodes)))
    m2.metric("Quan hệ trong kho", counts.get("relationships", len(relations)))
    type_filter = st.multiselect("Loại thực thể", sorted({node["type"] for node in nodes}), default=sorted({node["type"] for node in nodes}))
    filtered_nodes = [node for node in nodes if node["type"] in type_filter]
    node_ids = {node["id"] for node in filtered_nodes}
    filtered_edges = [edge for edge in relations if edge["source"] in node_ids and edge["target"] in node_ids]
    if not filtered_nodes:
        st.info("Đồ thị đang trống. Tải lên và xử lý một tài liệu để tạo node cùng quan hệ có nguồn.")
        return
    graph = nx.Graph()
    for node in filtered_nodes:
        graph.add_node(node["id"])
    for edge in filtered_edges:
        graph.add_edge(edge["source"], edge["target"])
    positions = nx.spring_layout(graph, seed=17, k=0.9, iterations=35)
    edge_x, edge_y = [], []
    for edge in filtered_edges:
        start, end = positions[edge["source"]], positions[edge["target"]]
        edge_x.extend([start[0], end[0], None])
        edge_y.extend([start[1], end[1], None])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1, color="#aab8c8"), hoverinfo="none", name="Quan hệ"))
    palette = {"LegalDocument": NAVY, "LegalArticle": BLUE, "LegalConcept": TEAL, "Chunk": "#d89b4c"}
    for kind in type_filter:
        subset = [node for node in filtered_nodes if node["type"] == kind]
        fig.add_trace(go.Scatter(
            x=[positions[node["id"]][0] for node in subset],
            y=[positions[node["id"]][1] for node in subset],
            mode="markers+text", text=[node["name"][:32] for node in subset], textposition="top center",
            marker=dict(size=15, color=palette.get(kind, BLUE), line=dict(width=1, color="white")),
            customdata=[[node["id"], node.get("chunk_id") or ""] for node in subset],
            hovertemplate="%{text}<br>Loại: " + kind + "<extra></extra>", name=kind,
        ))
    fig.update_layout(height=570, showlegend=True, hovermode="closest", margin=dict(l=5, r=5, t=10, b=5), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=colors["surface"], font=dict(color=colors["text"]), xaxis=dict(visible=False), yaxis=dict(visible=False))
    st.plotly_chart(fig, use_container_width=True)
    names = {f"{node['name']} · {node['type']} · {node['id'][:8]}": node for node in filtered_nodes}
    selected_name = st.selectbox("Chi tiết node", ["Chọn một node…", *names.keys()])
    if selected_name != "Chọn một node…":
        selected = names[selected_name]
        related = [edge for edge in filtered_edges if selected["id"] in {edge["source"], edge["target"]}]
        st.json({"node": selected, "relationships": related})
        try:
            detail = api.get(f"/graph/nodes/{selected['id']}")
            if detail.get("document"):
                st.markdown(f"**Nguồn:** {detail['document']['title']} · {detail['document'].get('document_number') or 'chưa có số hiệu'} · {detail['document']['validity_status']}")
                if detail["document"].get("source_url"):
                    st.markdown(f"[Mở văn bản nguồn]({detail['document']['source_url']})")
            if detail.get("source_chunk"):
                st.write(detail["source_chunk"]["content"])
            if detail.get("relationships"):
                st.dataframe(pd.DataFrame(detail["relationships"]), use_container_width=True, hide_index=True)
        except ApiError as exc:
            st.warning(str(exc))
    with st.expander("Quan hệ hiển thị"):
        if filtered_edges:
            st.dataframe(pd.DataFrame([{"loại": e["type"], "nguồn": e["source"], "đích": e["target"], "đoạn bằng chứng": e.get("evidence", "")[:220]} for e in filtered_edges]), use_container_width=True, hide_index=True)
        else:
            st.caption("Không có cạnh trong bộ lọc hiện tại.")


def requests_quote(value: str) -> str:
    from urllib.parse import quote

    return quote(value, safe="")


def documents_page():
    hero("Văn bản pháp luật", "Nạp tài liệu có nguồn, theo dõi xử lý và quản lý index. Tình trạng hiệu lực không được tự suy diễn.", "KHO TRI THỨC")
    with st.expander("＋ Nạp văn bản", expanded=False):
        tab_file, tab_url = st.tabs(["Tải tệp", "Nhập URL HTTPS"])
        with tab_file:
            with st.form("upload_form"):
                upload = st.file_uploader("Chọn PDF, DOCX hoặc TXT", type=["pdf", "docx", "txt"])
                title = st.text_input("Tên văn bản *")
                number = st.text_input("Số, ký hiệu")
                issuing_body = st.text_input("Cơ quan ban hành")
                col_a, col_b = st.columns(2)
                issued = col_a.date_input("Ngày ban hành", value=None)
                effective = col_b.date_input("Ngày hiệu lực", value=None)
                effective_to = st.date_input("Ngày hết hiệu lực (nếu biết)", value=None)
                status_value = st.selectbox("Tình trạng hiệu lực", ["unverified", "verified_active", "expired", "repealed"], format_func=lambda x: {"unverified":"Chưa xác minh", "verified_active":"Đã xác minh còn hiệu lực", "expired":"Hết hiệu lực", "repealed":"Đã bị bãi bỏ"}[x])
                source_url = st.text_input("URL nguồn chính thức")
                submitted = st.form_submit_button("Tải lên và lập chỉ mục", type="primary")
            if submitted:
                if not upload or not title.strip():
                    st.error("Cần chọn tệp và nhập tên văn bản.")
                else:
                    data = {"title": title, "document_number": number or None, "issuing_body": issuing_body or None, "validity_status": status_value}
                    if issued:
                        data["issued_at"] = issued.isoformat()
                    if effective:
                        data["effective_from"] = effective.isoformat()
                    if effective_to:
                        data["effective_to"] = effective_to.isoformat()
                    if source_url.strip():
                        data["source_url"] = source_url.strip()
                    try:
                        result = api.post("/admin/documents/upload", data=data, files={"file": (upload.name, upload.getvalue(), upload.type or "application/octet-stream")}, timeout=90)
                        st.success(f"Đã tạo job {result['job_id']} — theo dõi ở trạng thái tài liệu.")
                        st.rerun()
                    except ApiError as exc:
                        show_api_error(exc)
        with tab_url:
            with st.form("url_form"):
                url = st.text_input("URL tới tệp PDF, DOCX hoặc TXT (HTTPS công khai)")
                url_title = st.text_input("Tên văn bản", key="url_title")
                url_number = st.text_input("Số, ký hiệu", key="url_number")
                url_status = st.selectbox("Tình trạng hiệu lực", ["unverified", "verified_active", "expired", "repealed"], key="url_status", format_func=lambda x: {"unverified":"Chưa xác minh", "verified_active":"Đã xác minh còn hiệu lực", "expired":"Hết hiệu lực", "repealed":"Đã bị bãi bỏ"}[x])
                url_submitted = st.form_submit_button("Tạo job nhập URL", type="primary")
            if url_submitted:
                try:
                    result = api.post("/admin/documents/import-url", json={"url": url, "title": url_title, "document_number": url_number or None, "validity_status": url_status}, timeout=30)
                    st.success(f"Đã tạo job {result['job_id']}.")
                    st.rerun()
                except ApiError as exc:
                    show_api_error(exc)
    try:
        total = api.get("/documents?limit=500")
    except ApiError as exc:
        show_api_error(exc)
        return
    search_col, status_col, validity_col, page_col = st.columns([1.8, 1, 1.2, .8])
    search = search_col.text_input("Tìm theo tên")
    status_filter = status_col.selectbox("Trạng thái xử lý", ["Tất cả", "queued", "processing", "completed", "failed"])
    validity_filter = validity_col.selectbox("Hiệu lực", ["Tất cả", "unverified", "verified_active", "expired", "repealed"])
    page_num = page_col.number_input("Trang", min_value=1, max_value=max(1, (len(total) + 9) // 10), step=1)
    params = f"?limit=10&offset={(int(page_num)-1)*10}"
    if search:
        params += f"&q={requests_quote(search)}"
    if status_filter != "Tất cả":
        params += f"&processing_status={status_filter}"
    if validity_filter != "Tất cả":
        params += f"&validity_status={validity_filter}"
    try:
        docs = api.get("/documents" + params)
    except ApiError as exc:
        show_api_error(exc)
        return
    st.caption(f"{len(total)} tài liệu trong kho · trang {int(page_num)}")
    for doc in docs:
        with st.expander(f"{doc['title']} · {doc.get('document_number') or 'chưa có số hiệu'} · {doc['processing_status']}"):
            col1, col2, col3 = st.columns([2, 2, 1])
            col1.write(f"Hiệu lực: **{doc['validity_status']}**")
            col1.write(f"Đoạn: {doc['chunk_count']} · nguồn: {doc.get('source_name') or doc.get('source_url') or 'tệp tải lên'}")
            if doc.get("processing_error"):
                col1.error(doc["processing_error"])
            if doc.get("source_url"):
                col2.markdown(f"[Mở nguồn]({doc['source_url']})")
            if col2.button("Xem nội dung và chunks", key=f"detail_{doc['id']}"):
                try:
                    st.session_state.doc_detail = api.get(f"/documents/{doc['id']}")
                except ApiError as exc:
                    show_api_error(exc)
            if st.session_state.get("doc_detail", {}).get("id") == doc["id"]:
                detail = st.session_state.doc_detail
                st.write(detail.get("content", "Chưa trích xuất được nội dung."))
                st.write(f"Chunks đã lập chỉ mục: {len(detail.get('chunks', []))}")
                for chunk in detail.get("chunks", [])[:12]:
                    st.caption(f"{chunk['location_label']} · {chunk['id'][:8]}")
                    st.code(chunk["content"][:1500])
            if doc["processing_status"] == "failed" and col3.button("Thử lập chỉ mục lại", key=f"reindex_{doc['id']}"):
                try:
                    result = api.post(f"/admin/documents/{doc['id']}/reindex")
                    st.info(f"Job: {result['job_id']}")
                    st.rerun()
                except ApiError as exc:
                    show_api_error(exc)
            confirm = col3.checkbox("Xác nhận xóa", key=f"confirm_delete_{doc['id']}")
            if confirm and col3.button("Xóa văn bản", key=f"delete_{doc['id']}"):
                try:
                    api.delete(f"/admin/documents/{doc['id']}")
                    st.success("Đã xóa văn bản và các index liên quan.")
                    st.rerun()
                except ApiError as exc:
                    show_api_error(exc)


def compare_page():
    hero("Vector RAG và GraphRAG", "Chạy cùng một câu hỏi trên cùng kho văn bản, rồi đối chiếu bằng chứng và quan hệ được mở rộng.", "SO SÁNH RETRIEVAL")
    question = st.text_area("Câu hỏi kiểm tra", height=95, placeholder="Nhập câu hỏi cần so sánh")
    top_k = st.slider("Số đoạn tối đa", 1, 12, 5, key="compare_k")
    if st.button("Chạy so sánh", type="primary", disabled=not question.strip()):
        try:
            result = api.post("/retrieval/compare", json={"question": question, "retrieval_mode": "graphrag", "top_k": top_k}, timeout=180)
            st.session_state.compare_result = result
        except ApiError as exc:
            show_api_error(exc)
    result = st.session_state.get("compare_result")
    if not result:
        return
    vector, graph = st.columns(2)
    for col, label, value in [(vector, "Vector RAG", result["vector_rag"]), (graph, "GraphRAG", result["graph_rag"] )]:
        with col:
            st.subheader(label)
            st.markdown(value["answer"]["answer"])
            st.caption(f"{value['metrics']['retrieval_backend']} · {value['metrics']['chunks_retrieved']} chunks · {value['metrics']['graph_relationships_used']} quan hệ · {value['metrics'].get('retrieval_latency_ms', 0)} ms retrieval")
            for warning in value.get("warnings", []) + value["answer"].get("warnings", []):
                st.warning(warning)
            display_sources(value.get("evidence", []))
    baseline = result.get("baseline", {})
    if baseline.get("available"):
        st.subheader("Baseline LLM · không retrieval")
        st.markdown(baseline.get("answer") or "")
        for warning in baseline.get("warnings", []):
            st.warning(warning)
    else:
        st.info("Baseline LLM không retrieval chưa chạy được vì chưa có model khả dụng; không tạo câu trả lời thiếu căn cứ.")


def evaluation_page():
    hero("Đánh giá hệ thống", "Chỉ hiển thị chỉ số tính từ bộ câu hỏi có nhãn chunk nguồn. Không có benchmark giả lập.", "ĐO LƯỜNG CÓ THỂ KIỂM TRA")
    try:
        stats = api.get("/stats/overview")
    except ApiError as exc:
        show_api_error(exc)
        return
    st.metric("Câu hỏi có nhãn", stats.get("evaluation_cases", 0))
    if stats.get("evaluation_cases", 0) == 0:
        st.info("Chưa có dữ liệu đánh giá. Thêm case vào evaluation/datasets/eval.jsonl với nhãn relevant_article_numbers hoặc relevant_chunk_ids, kèm nguồn và phạm vi phiên bản đã kiểm tra.")
    mode = st.selectbox("Phương thức", ["both", "vector", "graphrag"], format_func=lambda x: {"both":"Vector RAG + GraphRAG", "vector":"Vector RAG", "graphrag":"GraphRAG"}[x])
    top_k = st.slider("K", 1, 20, 5, key="eval_k")
    if st.button("Chạy đánh giá", type="primary"):
        try:
            st.session_state.eval_result = api.post("/evaluation/run", json={"retrieval_mode": mode, "top_k": top_k}, timeout=300)
        except ApiError as exc:
            show_api_error(exc)
    result = st.session_state.get("eval_result")
    if result:
        st.caption(f"Run ID: {result['run_id']} · {result['dataset_cases']} case")
        for method, metrics in result["aggregates"].items():
            st.subheader(method)
            cols = st.columns(4)
            for col, (label, key) in zip(cols, [("Recall@K", "mean_recall_at_k"), ("MRR", "mean_mrr"), ("Citation precision vs labels", "mean_citation_accuracy"), ("Latency ms", "mean_latency_ms")]):
                col.metric(label, metrics.get(key) if metrics.get(key) is not None else "N/A")
        frame = pd.DataFrame(result["results"])
        st.dataframe(frame, use_container_width=True, hide_index=True)
        st.download_button("Xuất CSV", frame.to_csv(index=False).encode("utf-8-sig"), "evaluation_results.csv", "text/csv")
    if st.session_state.admin_token:
        try:
            history = api.get("/evaluation/results")
            if history.get("results"):
                st.subheader("Lịch sử chạy")
                result_frame = pd.DataFrame(history["results"])
                st.dataframe(result_frame.drop(columns=["answer_text", "details"], errors="ignore"), use_container_width=True, hide_index=True)
                review_ids = [row["id"] for row in history["results"]]
                selected_review = st.selectbox("Chấm thủ công answer correctness / faithfulness", review_ids)
                existing_review = next(row for row in history["results"] if row["id"] == selected_review)
                with st.expander("Câu trả lời và bằng chứng để chấm"):
                    st.markdown(f"**Câu hỏi:** {existing_review['question']}")
                    expected_answer = existing_review.get("details", {}).get("expected_answer")
                    if expected_answer:
                        st.markdown(f"**Đáp án tham chiếu đã gắn nguồn:** {expected_answer}")
                    st.markdown(existing_review.get("answer_text") or "Không có câu trả lời được tạo.")
                    for source in existing_review.get("details", {}).get("sources", []):
                        st.markdown(f"**[{source.get('citation_id')}] {source.get('document_title')} · {source.get('location_label')} · {source.get('validity_status')}**")
                        st.write(source.get("text", ""))
                with st.form("manual_answer_review"):
                    correctness = st.slider("Answer correctness", 0.0, 1.0, float(existing_review.get("answer_correctness_score") or 0.0), 0.05)
                    faithfulness = st.slider("Faithfulness to cited evidence", 0.0, 1.0, float(existing_review.get("faithfulness_score") or 0.0), 0.05)
                    review_notes = st.text_area("Ghi chú rubric", value=existing_review.get("review_notes") or "")
                    save_review = st.form_submit_button("Lưu đánh giá thủ công")
                if save_review:
                    try:
                        api.patch(f"/evaluation/results/{selected_review}/review", json={"answer_correctness_score": correctness, "faithfulness_score": faithfulness, "review_notes": review_notes or None})
                        st.success("Đã lưu điểm đánh giá thủ công.")
                        st.rerun()
                    except ApiError as exc:
                        show_api_error(exc)
        except ApiError:
            pass


def settings_page():
    hero("Cấu hình hệ thống", "Kiểm tra model, thiết bị, index và nhật ký. Thay đổi tham số retrieval chỉ có hiệu lực trong runtime hiện tại.", "QUẢN TRỊ")
    try:
        info = api.get("/system/status")
    except ApiError as exc:
        show_api_error(exc)
        return
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Dịch vụ")
        st.json({"API": info["api"], "Database": info["database_url_kind"], "LLM": info["llm"], "Embeddings": info["embeddings"], "Graph": info["graph"], "Training": info["training"]})
    with col_b:
        st.subheader("Thiết bị")
        st.json(info.get("device", {}))
        st.caption("GTX 750 Ti 4 GB không phù hợp để fine-tune Qwen3-8B; QLoRA được tách riêng và không tự chạy.")
    if st.session_state.admin_token:
        try:
            current = api.get("/admin/settings/retrieval")
            with st.form("retrieval_settings"):
                st.subheader("Tham số truy xuất an toàn")
                top_k = st.number_input("Số đoạn mặc định", 1, 20, int(current["default_top_k"]))
                chunk_size = st.number_input("Kích thước chunk (ký tự)", 500, 4000, int(current["chunk_size"]), step=100)
                overlap = st.number_input("Chunk overlap", 0, 1000, int(current["chunk_overlap"]), step=20)
                save = st.form_submit_button("Áp dụng trong phiên chạy")
            if save:
                updated = api.put("/admin/settings/retrieval", json={"default_top_k": top_k, "chunk_size": chunk_size, "chunk_overlap": overlap})
                st.success(f"Đã áp dụng: {updated}")
        except ApiError as exc:
            st.warning(str(exc))
        with st.expander("Nhật ký backend"):
            if st.button("Tải nhật ký mới nhất"):
                try:
                    logs = api.get("/admin/logs?limit=200")
                    st.code("".join(logs.get("lines", [])) or "Chưa có log.")
                except ApiError as exc:
                    show_api_error(exc)
    st.markdown("**Cấu hình Ollama:** đặt `OLLAMA_MODEL` trong `.env` sau khi đã tải đúng model. Health API hiển thị model nào đang có thật.")
    st.markdown("**QLoRA:** tạo bộ dữ liệu JSONL đã kiểm chứng rồi chạy script độc lập trên phần cứng tương thích. Chưa có training nào được báo là hoàn tất.")


ROUTES = {
    "Tổng quan": dashboard,
    "Trợ lý pháp luật": chat_page,
    "Knowledge Graph": graph_page,
    "Văn bản pháp luật": documents_page,
    "So sánh retrieval": compare_page,
    "Đánh giá": evaluation_page,
    "Cấu hình": settings_page,
}
ROUTES[page]()
