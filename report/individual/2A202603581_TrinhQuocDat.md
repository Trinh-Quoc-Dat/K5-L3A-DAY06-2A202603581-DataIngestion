# Báo cáo cá nhân — Day 06: Data Ingestion Engineering

**Họ và tên:** Trịnh Quốc Đạt  
**MSSV:** 2A202603581  
**Lớp:** L3A

## 1. Dữ liệu và pipeline

### 1.1. Corpus

- **Tên tài liệu:** Principles of Data Science
- **Nguồn:** OpenStax
- **Định dạng:** PDF
- **Tổng số trang vật lý:** 569
- **Số trang có nội dung văn bản:** 567
- **Phạm vi nội dung chính:** Chương 1–10, trang vật lý 19–503, tổng cộng 485 trang.
- **Phương pháp trích xuất:** pypdf.
- **Manifest:** `starter/data/manifest.json`.

Tài liệu được giữ nguyên số trang vật lý để bảo toàn khả năng truy xuất nguồn và đối chiếu evidence.

**Giới hạn quyền sử dụng:** Tài liệu có quy định riêng về việc sử dụng nội dung trong các hệ thống AI tạo sinh. Vì chưa xác nhận được quyền sử dụng phù hợp cho dịch vụ embedding bên ngoài, pipeline real-corpus ingestion hiện chưa được thực thi.

### 1.2. Tiền xử lý và chia chunk

Các bước xử lý đã triển khai:

1. Đọc nội dung PDF theo từng trang.
2. Chuẩn hóa Unicode bằng NFKC.
3. Chuẩn hóa line endings và khoảng trắng.
4. Giữ lại paragraph breaks.
5. Gán metadata cho từng trang.
6. Chia chunk theo hai phương pháp: fixed-size và paragraph-aware.

Metadata hiện có:

- `language`
- `category`
- `chapter`
- `section`
- `page_has_text`
- `extraction_method`

Kết quả chuẩn bị dữ liệu sơ bộ:

| Cấu hình | Số chunk | Tổng ký tự | Độ dài trung bình |
|---|---:|---:|---:|
| A | 1.661 | 1.412.406 | 850,33 |
| B | 1.615 | 1.247.621 | 772,52 |
| C | 1.615 | 1.247.621 | 772,52 |

Đây là thống kê chuẩn bị chunk cục bộ, không phải kết quả retrieval hoặc benchmark.

### 1.3. Embedding và vector store

| Thành phần | Cấu hình |
|---|---|
| Provider | Google Gemini |
| Model | `gemini-embedding-001` |
| Dimensions | 768 |
| Vector store | PostgreSQL 16 + pgvector 0.8.7 |
| Similarity metric | Cosine distance |
| Python SDK | google-genai |

Embedding adapter hỗ trợ batch processing, phân biệt document embedding và query embedding, đồng thời ghi nhận request count và thời gian xử lý.

Đã kiểm tra embedding và truy xuất vector bằng dữ liệu tổng hợp. Chưa thực hiện embedding trên corpus PDF thật.

Token usage chưa được API trả về trong lần kiểm thử, do đó chưa có cơ sở xác định tổng token và chi phí embedding thực tế.

## 2. Cấu hình và benchmark

### 2.1. Ba chiến lược

| Config | Chunking | Tham số | Metadata filter |
|---|---|---|---|
| A | Fixed-size | Size = 1000, overlap = 150 | Tắt |
| B | Paragraph-aware | Max size = 1000 | Tắt |
| C | Paragraph-aware | Max size = 1000 | Bật |

Mục tiêu so sánh:

- A–B: đánh giá ảnh hưởng của phương pháp chia chunk.
- B–C: đánh giá ảnh hưởng riêng của metadata filtering.

### 2.2. Benchmark

Bộ câu hỏi tại `benchmark/questions.json` gồm:

- 20 câu hỏi tổng cộng.
- 15 câu dev.
- 5 câu test.
- Các dạng câu hỏi trực tiếp, paraphrase và condition-based.
- 4 câu có metadata filter.
- Có câu hỏi filter trong tập test.

Mỗi câu hỏi có expected answer, gold source, gold page và evidence quote được đối chiếu với nội dung PDF.

### 2.3. Tiêu chí đánh giá

1. Retrieval accuracy: Hit@1 và Evidence Hit@3.
2. Chunk coherence: đánh giá khả năng giữ nguyên ngữ cảnh.
3. Metadata usefulness: đánh giá tính đầy đủ và chính xác của metadata.
4. Grounding quality: đánh giá mức độ evidence hỗ trợ câu trả lời.
5. Strategy impact: so sánh hiệu quả và tài nguyên giữa các cấu hình.

Tiêu chí lựa chọn chiến lược cuối cùng cần được cố định trước khi đánh giá tập test.

## 3. Kết quả thực nghiệm

### 3.1. Retrieval benchmark

| Strategy | Split | Hit@1 | Evidence Hit@3 | Trạng thái |
|---|---|---|---|---|
| A | Dev | Chưa đo | Chưa đo | Chưa chạy |
| A | Test | Chưa đo | Chưa đo | Chưa chạy |
| B | Dev | Chưa đo | Chưa đo | Chưa chạy |
| B | Test | Chưa đo | Chưa đo | Chưa chạy |
| C | Dev | Chưa đo | Chưa đo | Chưa chạy |
| C | Test | Chưa đo | Chưa đo | Chưa chạy |

### 3.2. Manual audit

| Strategy | Chunk coherence | Metadata usefulness | Filter compliance | Grounding quality |
|---|---|---|---|---|
| A | Chưa đánh giá | Chưa đánh giá | N/A | Chưa đánh giá |
| B | Chưa đánh giá | Chưa đánh giá | N/A | Chưa đánh giá |
| C | Chưa đánh giá | Chưa đánh giá | Chưa đánh giá | Chưa đánh giá |

Chưa có kết quả retrieval trên corpus thật nên chưa thể thực hiện manual audit có căn cứ.

### 3.3. Tài nguyên và chi phí

- Đã kiểm tra batch embedding bằng dữ liệu tổng hợp.
- Đã kiểm tra thao tác upsert và search trên PostgreSQL + pgvector.
- Đã kiểm tra metadata filtering bằng dữ liệu tổng hợp.
- Chưa có thời gian ingestion corpus thật.
- Chưa có chi phí embedding corpus thật.
- Token usage chưa khả dụng trong phản hồi API của lần kiểm thử.

## 4. Phân tích và kết luận

Ở giai đoạn hiện tại, dự án đã hoàn thành phần chuẩn bị corpus, xây dựng các phương pháp chunking, embedding adapter, vector store và cấu trúc benchmark.

Các kiểm thử tổng hợp xác nhận những thành phần chính có thể phối hợp với nhau, bao gồm tạo embedding, lưu vector, truy xuất top-k và metadata filtering.

Tuy nhiên, chưa có kết quả retrieval trên corpus thật. Vì vậy, chưa thể kết luận chiến lược A, B hay C có hiệu quả tốt hơn.

### Hạn chế

1. Chưa xác nhận quyền sử dụng corpus cho external AI embedding.
2. Chưa chạy real-corpus ingestion và retrieval.
3. Chưa thực hiện manual audit.
4. Chưa có số liệu thực tế về thời gian và chi phí embedding.
5. Tập test chỉ có 5 câu hỏi nên cần thận trọng khi diễn giải kết quả khi benchmark được thực hiện.

### Hướng hoàn thiện

- Xác nhận quyền sử dụng dữ liệu.
- Thực hiện ingestion trên corpus thật khi đủ điều kiện.
- Lưu trace top-3 cho cả ba chiến lược.
- Hoàn thành manual audit theo năm tiêu chí.
- So sánh A–B và B–C dựa trên kết quả dev/test.
- Hoàn thiện kết luận dựa trên số liệu thực nghiệm.

## 5. Tệp và khả năng tái lập

- Source code: `starter/src/`
- Corpus manifest: `starter/data/manifest.json`
- Benchmark: `benchmark/questions.json`
- Experiment log: `artifacts/experiment_log.csv`
- Preparation summary: `starter/results/preparation_summary.json`

Các kiểm thử đã thực hiện gồm Python compilation, synthetic embedding, PostgreSQL vector insertion, top-3 retrieval và metadata filtering.