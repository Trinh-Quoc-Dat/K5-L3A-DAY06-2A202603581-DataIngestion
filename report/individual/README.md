# Day 06 – Data Ingestion Engineering

**Học viên:** Trịnh Quốc Đạt  
**MSSV:** 2A202603581  
**Lớp:** L3A

Dự án xây dựng pipeline Data Ingestion để trích xuất, chuẩn hóa, chia nhỏ tài liệu, tạo embedding và lưu trữ vector phục vụ truy xuất tri thức. Đồng thời, dự án chuẩn bị benchmark để so sánh ba chiến lược xử lý dữ liệu.

## 1. Project overview

| Thành phần | Công nghệ |
|---|---|
| Language | Python |
| Embedding provider | Google Gemini |
| Embedding model | `gemini-embedding-001` |
| Embedding dimensions | 768 |
| Vector database | PostgreSQL 16 |
| Vector extension | pgvector 0.8.7 |
| Similarity metric | Cosine distance |
| PDF extraction | pypdf |

**Corpus:** OpenStax – *Principles of Data Science*.

- Tổng số trang vật lý: 569.
- Số trang có nội dung văn bản: 567.
- Nội dung chính Chương 1–10: 485 trang, từ trang vật lý 19–503.
- Manifest: `starter/data/manifest.json`.

PDF không được đưa vào repository. Người chạy cần tải tài liệu từ nguồn OpenStax chính thức và đặt tại:

`starter/data/principles-data-science.pdf`

**Lưu ý về quyền sử dụng:** Nguồn tài liệu có quy định riêng về việc sử dụng nội dung để huấn luyện hoặc đưa vào các dịch vụ AI tạo sinh. Real-corpus ingestion qua dịch vụ embedding bên ngoài hiện vẫn bị khóa cho đến khi xác nhận được quyền sử dụng phù hợp.

## 2. Retrieval configurations

Ba cấu hình được chuẩn bị để so sánh:

| Config | Chunking | Parameters | Metadata filter |
|---|---|---|---|
| A | Fixed-size | Size 1000, overlap 150 | Off |
| B | Paragraph-aware | Max size 1000 | Off |
| C | Paragraph-aware | Max size 1000 | On |

Mục tiêu so sánh:

- A vs B: tác động của chiến lược chia chunk.
- B vs C: tác động riêng của metadata filtering.

Các cấu hình được thiết kế để sử dụng cùng corpus, embedding model và benchmark.

## 3. Benchmark design

Bộ benchmark gồm 20 câu hỏi:

- 15 câu development.
- 5 câu held-out test.
- Các dạng câu hỏi trực tiếp, diễn đạt lại và câu hỏi có điều kiện.
- Có câu hỏi yêu cầu metadata filtering.
- Mỗi câu có expected answer, nguồn, trang và evidence quote.

File: `benchmark/questions.json`.

Các tiêu chí đánh giá:

1. Retrieval accuracy: Hit@1 và Evidence Hit@3.
2. Chunk coherence.
3. Metadata usefulness và filter compliance.
4. Grounding quality.
5. Strategy impact và resource trade-offs.

## 4. Repository structure

```text
.
├── STUDENT.md
├── README.md
├── benchmark/
│   └── questions.json
├── artifacts/
│   └── experiment_log.csv
├── report/
│   └── individual/
│       ├── 2A202603581_TrinhQuocDat.md
│       └── README.md
└── starter/
    ├── data/
    │   └── manifest.json
    ├── results/
    ├── scripts/
    │   └── check_setup.py
    ├── src/
    ├── requirements.txt
    └── docker-compose.yml
```

## 5. Environment setup

Thực hiện từ thư mục gốc repository trên Windows CMD:

```cmd
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r starter\requirements.txt
```

Tạo file cấu hình local từ file mẫu:

```cmd
copy starter\.env.example starter\.env
```

Điền `GEMINI_API_KEY` và các cấu hình cần thiết trong `starter/.env`. Không commit API key hoặc file `.env`.

Khởi động PostgreSQL + pgvector:

```cmd
docker compose -p day6_student_starter -f starter\docker-compose.yml up -d --wait
```

Kiểm tra database:

```cmd
python starter\scripts\check_setup.py
```

## 6. Available commands

Chạy các lệnh từ thư mục gốc repository:

**Chuẩn bị corpus và thống kê chunk, không tạo embedding:**

```cmd
python -m starter.src.run --prepare-only
```

**Kiểm tra Gemini và pgvector bằng dữ liệu tổng hợp:**

```cmd
python -m starter.src.run --smoke-test
```

**Ingest corpus thật:**

```cmd
python -m starter.src.run --ingest
```

Lệnh ingest hiện bị khóa có chủ đích cho đến khi xác nhận quyền sử dụng corpus với dịch vụ embedding bên ngoài.

## 7. Current implementation status

### Completed

- PDF extraction và metadata assignment.
- Fixed-size và paragraph-aware chunking.
- Gemini embedding adapter với batch processing và request statistics.
- PostgreSQL + pgvector integration.
- Synthetic embedding và vector retrieval smoke test.
- Metadata filtering smoke test.
- Bộ benchmark 20 câu hỏi.
- Cấu trúc experiment log và preparation summary.

### Pending

- Xác nhận quyền sử dụng corpus cho external AI embedding.
- Real-corpus ingestion và retrieval experiments.
- Manual audit trên kết quả top-3.
- Tổng hợp dev/test metrics.
- Phân tích và kết luận chiến lược A/B/C.

**Lưu ý:** Kết quả synthetic smoke test chỉ xác nhận tích hợp kỹ thuật, không phải kết quả benchmark trên corpus thật.

## 8. Individual report

Báo cáo cá nhân:

`report/individual/2A202603581_TrinhQuocDat.md`

Các tài liệu hướng dẫn và tiêu chí chấm điểm gốc được giữ trong repository: `GUIDE.md`, `BENCHMARK.md`, `RUBRIC.md`, `RULES.md` và `SUBMISSION.md`.
