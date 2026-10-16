# GraphRAG-for-Vietnamese-Legal-Chatbot
GraphRAG for Vietnamese Legal Chatbot
# Legal GraphRAG — chatbot hỏi đáp văn bản pháp luật

Ứng dụng local gồm FastAPI, SQLite, Streamlit, vector retrieval và Knowledge Graph. Bộ cài kèm một văn bản nguồn chính thức để có thể chạy thử ngay: Bộ luật Lao động, văn bản hợp nhất 125/VBHN-VPQH. Tài liệu do Văn phòng Quốc hội ban hành ngày 27/08/2025; văn bản hợp nhất ghi nhận sửa đổi theo Luật 71/2025/QH15 có hiệu lực từ 01/01/2026. Trạng thái nguồn được đặt `unverified`: dự án chưa xác minh các thay đổi pháp luật sau bản hợp nhất này. Xem [ghi chú nguồn mẫu](data/samples/README.md).

## Khởi chạy trên Windows

Yêu cầu Python 3.11 (khuyến nghị) hoặc 3.14, PowerShell và kết nối mạng để cài thư viện. Từ thư mục dự án, chạy:

```powershell
.\scripts\setup.ps1
```

Script tạo `.venv`, cài dependencies, tạo `.env` với admin token ngẫu nhiên nếu chưa có, rồi nạp văn bản mẫu vào SQLite và xây chỉ mục. Nếu script tạo token mới, hãy lưu lại và nhập vào ô **Admin API token** trong sidebar Streamlit để dùng các thao tác quản trị.

Mở hai cửa sổ PowerShell riêng:

```powershell
.\scripts\run_backend.ps1
```

```powershell
.\scripts\run_frontend.ps1
```

Giao diện ở `http://127.0.0.1:8501`, Swagger ở `http://127.0.0.1:8000/docs`, health check ở `http://127.0.0.1:8000/health`. Dừng mỗi tiến trình bằng `Ctrl+C`. SQLite và file nạp vào được lưu trong `data/`.

Chạy unit tests bằng:

```powershell
.\scripts\run_tests.ps1
```

## Tính năng

- **Chat:** lưu hội thoại vào SQLite; Vector RAG và GraphRAG dùng chung tập tài liệu; trả nguồn kèm đoạn trích, trạng thái hiệu lực và cảnh báo.
- **Nạp văn bản:** PDF có text layer, DOCX, TXT hoặc URL HTTPS công khai; phát hiện tệp trùng theo SHA-256; theo dõi background job, xem chunk/lỗi, re-index và xóa tài liệu. URL mạng riêng/local bị chặn. PDF scan chưa có OCR.
- **Knowledge Graph:** NetworkX dựng từ SQLite với node văn bản, điều, đoạn và khái niệm; quan hệ lưu đoạn bằng chứng cùng ID chunk nguồn. Tham chiếu điều được trích trực tiếp từ văn bản; khái niệm dùng tập từ khóa xác định.
- **Truy xuất:** mặc định TF-IDF sparse vectors kết hợp BM25; GraphRAG mở rộng theo tham chiếu điều và chủ đề, tránh đi qua node văn bản tổng hợp nối tới mọi chunk. Có thể bật dense vectors/FAISS bằng `requirements-ml.txt` và `LEGAL_EMBEDDING_MODEL`.
- **LLM:** nếu chưa cấu hình model, chatbot hiển thị bằng chứng đã truy xuất thay vì giả lập câu trả lời. Có thể kết nối Ollama hoặc Transformers/PEFT. Citation ID của model được đối chiếu với bằng chứng của request; kiểm tra ID không bảo đảm mọi nhận định đều đúng.
- **Đánh giá:** ba câu hỏi có nhãn theo điều trong `evaluation/datasets/eval.jsonl`; evaluator phân giải nhãn sang chunk của tài liệu đã index và tính Recall@K, MRR, citation precision khi model trả citation, cùng latency. Chấm answer correctness/faithfulness bằng rubric thủ công trong UI.
- **QLoRA:** pipeline riêng để kiểm tra provenance, chia train/validation/test theo nguồn, huấn luyện 4-bit và lưu adapter/metrics. Chat không tự fine-tune; repo không kèm adapter đã huấn luyện.

## Cấu hình model

### Ollama

1. Cài Ollama và tải model gọn hỗ trợ tiếng Việt: `ollama pull qwen2.5:3b-instruct`.
2. Đặt `OLLAMA_MODEL=qwen2.5:3b-instruct`, `OLLAMA_BASE_URL=http://127.0.0.1:11434` và `LEGAL_LLM_EVIDENCE_K=1` trong `.env`. Giá trị cuối giới hạn số đoạn đưa vào LLM; tăng nếu muốn tổng hợp nhiều đoạn và chấp nhận câu trả lời chậm hơn.
3. Khởi động lại backend; `/health` hiển thị trạng thái model. Ollama cần đang chạy khi đặt câu hỏi.

### Dense embeddings/FAISS (tùy chọn)

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt
```

Đặt `LEGAL_EMBEDDING_MODEL` thành model multilingual đã kiểm tra, rồi chạy `python indexing/build_embeddings.py` để tạo vector cho chunk hiện có. Model embedding nặng không được tải mặc định. Nếu wheel FAISS không tương thích, truy xuất dùng NumPy cosine similarity với vector đã lưu.

### Transformers adapter

Đặt `LEGAL_LLM_BACKEND=transformers`, `LEGAL_HF_MODEL` và, nếu có, `LEGAL_ADAPTER_PATH`. Adapter phải tương thích với model nền. Có thể đặt `LEGAL_HF_4BIT=true` trên môi trường CUDA/bitsandbytes được hỗ trợ. Model chỉ được nạp khi có câu hỏi và trạng thái sẵn sàng phụ thuộc vào lần inference thực tế.

## QLoRA và giới hạn phần cứng

Máy đã dùng để chạy ứng dụng có NVIDIA RTX 3060 với 6 GB VRAM. Ollama và Qwen2.5 3B Instruct đã được cài để sinh câu trả lời tại máy; mỗi câu có thể mất vài giây đến vài chục giây. Đây là model nhỏ nên vẫn cần đối chiếu câu trả lời với đoạn trích nguồn. Script QLoRA yêu cầu GPU CUDA tương thích (compute capability từ 7.0) và tối thiểu 16 GB VRAM cho cấu hình Qwen3-8B; GTX 750 Ti không đáp ứng. Dùng máy có GPU phù hợp để train. Training loss không phải thước đo độ chính xác pháp luật.

## Kiến trúc

```text
backend/          FastAPI, SQLite models, ingestion, retrieval, graph, LLM
frontend/         Streamlit workspace và API client
indexing/         CLI ingest, graph rebuild, embedding build
training/         Chuẩn hóa JSONL và QLoRA độc lập
evaluation/       Dataset có nhãn và benchmark
data/samples/     Văn bản mẫu chính thức và provenance
data/             SQLite, file gốc, log và index local (không commit)
tests/            Unit tests dùng nội dung synthetic có đánh dấu test-only
```

Admin routes yêu cầu `X-Admin-Token`. Dữ liệu hội thoại trong bản local single-user không gắn với tài khoản; bảng `users` sẵn sàng cho phần mở rộng. Không mở ứng dụng trực tiếp ra Internet.

Tài liệu pháp luật mới mặc định `unverified`; metadata ngày hiệu lực và trạng thái do admin nhập cần đối chiếu nguồn chính thức. Hệ thống lọc tài liệu hết hiệu lực/bị bãi bỏ khỏi câu hỏi hiện tại và chỉ tìm trong phạm vi lịch sử khi người dùng nêu thời điểm. Nếu không có bằng chứng, chatbot từ chối suy đoán. Đây là công cụ tra cứu/đồ án, không thay thế tư vấn pháp lý chuyên nghiệp.
