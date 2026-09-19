# Báo cáo nhóm — Lab 7: Embedding & Vector Store

**Nhóm:** phoboi

**Thành viên:** Trịnh Đức Huy — 2A202602865; Trịnh Hoàng Tùng — 2A202602937; Nguyễn Hoàng Sơn — 2A202602457; Đỗ Quốc An — 2A202602892

**Ngày cập nhật:** 19/09/2026

Báo cáo sử dụng dữ liệu mới trong `data/library/`: 8 tài liệu thư viện Đại học Bách khoa Hà Nội (HUST), thay thế danh mục và câu hỏi VNU của bản cũ. Nguồn đối chiếu là bản Markdown đã lưu trong repo; chưa kiểm tra lại website trực tiếp trong lần cập nhật này.

**Phạm vi bằng chứng:** Đã chạy `bench.py` cho cả bốn chiến lược bằng model local `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. File [benchmark_results.json](benchmark_results.json) đã được ghi đè bằng kết quả local (`"mock": false`) của đúng năm câu hỏi mới. Benchmark đo truy xuất top-3; phần trả lời dưới đây được đối chiếu thủ công với gold answer, không được trình bày như đầu ra của một LLM.

## 1. Lựa chọn tài liệu — 10 điểm

### Chủ đề và lý do lựa chọn

Chủ đề là dịch vụ thư viện HUST: mượn sách, gia hạn, tài khoản, giờ phục vụ, phòng học nhóm và tài nguyên số. Tài liệu có nhiều mục, quy trình và con số cần giữ đúng ngữ cảnh, phù hợp so sánh chunking. Hai tài liệu mượn sách cùng dành cho sinh viên nhưng khác hạn mức và thời hạn giúp kiểm tra khả năng phân biệt loại dịch vụ.

### Danh mục dữ liệu mới

Độ dài tính bằng `len(content)` của Python sau khi bỏ frontmatter, chuẩn hóa xuống dòng khi đọc và `strip()` phần thân; không phải số token. Tổng **10.952 ký tự**. Không đưa tài liệu mẫu ngoài `data/library/` vào benchmark.

| File trong data/library | Tài liệu | Nguồn ghi trong metadata | Ký tự | audience | category |
|---|---|---|---:|---|---|
| `digital-resource-faq.md` | Câu hỏi thường gặp về tài nguyên số | [HUST node/528](https://library.hust.edu.vn/vi/node/528) | 1.537 | all | digital-resources |
| `document-renewal.md` | Hướng dẫn gia hạn tài liệu | [HUST node/183](https://library.hust.edu.vn/vi/node/183) | 759 | all | renewal |
| `group-study-room.md` | Dịch vụ sử dụng phòng học nhóm | [HUST node/1363](https://library.hust.edu.vn/vi/node/1363) | 1.243 | all | study-room |
| `information-on-demand.md` | Dịch vụ cung cấp thông tin theo yêu cầu | [HUST node/41](https://library.hust.edu.vn/vi/node/41) | 1.414 | all | research-support |
| `library-account.md` | Hướng dẫn quản lý tài khoản bạn đọc | [HUST node/49](https://library.hust.edu.vn/vi/node/49) | 2.010 | all | library-account |
| `service-hours.md` | Giờ phục vụ thư viện | [HUST node/416](https://library.hust.edu.vn/vi/node/416) | 787 | all | hours |
| `student-reference-book-borrowing.md` | Hướng dẫn mượn sách tham khảo phòng 102 | [HUST node/502](https://library.hust.edu.vn/vi/node/502) | 1.071 | student | borrowing |
| `student-textbook-borrowing.md` | Quy trình mượn trả giáo trình phòng 111 | [HUST node/483](https://library.hust.edu.vn/vi/node/483) | 2.131 | student | borrowing |

Tất cả tài liệu ghi `retrieved_at: 2026-09-19`, `document_version: not-stated`, `department: library`, `language: vi`. Không thay `not-stated` bằng ngày hiệu lực tự suy đoán. Danh sách URL đầu vào nằm tại [data/urls.csv](../data/urls.csv).

### Metadata và quản trị dữ liệu

| Trường | Vai trò |
|---|---|
| `doc_id`, `title` | Định danh và tên tài liệu; `doc_id` khớp tên file bỏ đuôi `.md`. |
| `source_url`, `retrieved_at`, `document_version` | Truy vết nguồn, ngày thu thập và tình trạng phiên bản. |
| `audience` | Corpus có 2 tài liệu `student`, 6 tài liệu `all`. |
| `department`, `category`, `language` | Lọc đơn vị, loại dịch vụ và ngôn ngữ. |
| `chunk_index`, `strategy`, `source` | Benchmark bổ sung chỉ số chunk, chiến lược và URL trích dẫn. |

Danh mục ghi nguồn công khai (`public-source`); nhãn này không đồng nghĩa với giấy phép tái phân phối. Một số tài liệu chứa tên, email và số điện thoại đầu mối hỗ trợ công khai, vì vậy không khẳng định corpus hoàn toàn không có thông tin cá nhân. Không dùng thông tin đăng nhập hay tài khoản thật của sinh viên để thử nghiệm.

## 2. Thiết kế bốn chiến lược — 15 điểm

### Phân công đề xuất và thiết kế

Phân công dưới đây phục vụ trình bày và chạy lại, không khẳng định từng thành viên đã chạy thử nghiệm riêng.

| Thành viên | Chiến lược | Cấu hình và lý do chọn | Hạn chế |
|---|---|---|---|
| Đỗ Quốc An | **Fixed-size Chunking** | `FixedSizeChunker(500, 50)`: cửa sổ 500 ký tự, overlap 50; đường cơ sở đơn giản, giới hạn kích thước rõ. | Có thể cắt giữa câu hoặc bước thực hiện; overlap tăng lượng văn bản nhúng. |
| Nguyễn Hoàng Sơn | **Recursive Chunking** | `RecursiveChunker(chunk_size=500)`: ưu tiên đoạn, dòng, câu rồi khoảng trắng; phù hợp quy trình có nhiều đoạn ngắn. | Mã hiện tại có bước gộp nên có chunk vượt 500 ký tự; đây không phải giới hạn cứng. |
| Trịnh Hoàng Tùng | **Heading-based Chunking** | `heading_chunks()` chia tại tiêu đề `##`, giữ `###` trong mục cha; gộp tiêu đề tài liệu và mở đầu vào mục đầu tiên. | Mục dài sinh chunk dài; các chunk sau chỉ giữ tiêu đề tài liệu trong metadata. |
| Trịnh Đức Huy | **Semantic Chunking** | `semantic_chunks()` nhúng từng câu/dòng; tách khi cosine giữa hai đơn vị kế tiếp < 0,5 hoặc khi ghép vượt 500 ký tự. Dùng `LocalEmbedder`, cùng backend cho truy xuất. | Tốn lượt nhúng; ngưỡng 0,5 chưa được tối ưu. Dòng quá dài vẫn phải chia cứng. |

Semantic quyết định ranh giới bằng độ tương tự embedding, không phải gom cố định số câu. Không dùng vector hash của MockEmbedder để tạo ranh giới ngữ nghĩa: `--strategy all --backend mock` báo bỏ qua Semantic; `--strategy semantic --backend mock` báo lỗi rõ ràng.

`SentenceChunker` vẫn thuộc mã bài lab trong `src`, nhưng không thuộc bốn chiến lược của báo cáo. Benchmark có danh sách chiến lược riêng, không sử dụng đầu ra ba chiến lược của `ChunkingStrategyComparator` như kết quả bốn chiến lược.

### Kết quả đo cấu trúc chunk

Mỗi ô là **số chunk / độ dài trung bình (ký tự)**. Số liệu lấy từ lần chạy benchmark trên toàn bộ 8 file; frontmatter không được đưa vào embedding.

| doc_id | Fixed-size | Recursive | Heading-based | Semantic |
|---|---:|---:|---:|---:|
| digital-resource-faq | 4 / 421,8 | 4 / 381,0 | 6 / 254,5 | 8 / 190,0 |
| document-renewal | 2 / 404,5 | 2 / 375,5 | 3 / 251,7 | 13 / 56,9 |
| group-study-room | 3 / 447,7 | 3 / 411,7 | 5 / 247,0 | 20 / 60,6 |
| information-on-demand | 4 / 391,0 | 3 / 468,7 | 5 / 281,2 | 23 / 60,3 |
| library-account | 5 / 442,0 | 5 / 397,8 | 4 / 501,0 | 35 / 56,0 |
| service-hours | 2 / 418,5 | 2 / 390,5 | 2 / 392,5 | 8 / 96,9 |
| student-reference-book-borrowing | 3 / 390,3 | 2 / 531,5 | 3 / 355,7 | 21 / 49,5 |
| student-textbook-borrowing | 5 / 466,2 | 5 / 423,6 | 4 / 531,2 | 28 / 74,7 |
| **Tổng số chunk** | **28** | **26** | **32** | **156** |
| **Chunk dài nhất** | **500** | **619** | **794** | **433** |

Semantic với ngưỡng 0,5 tạo nhiều đoạn rất ngắn: 156 chunk, gấp khoảng 5–6 lần ba chiến lược còn lại. Điều này giúp tìm đúng `doc_id` nhưng thường chỉ trả về tiêu đề hoặc một dòng riêng lẻ, làm mất các dữ kiện cần kết hợp.

**Kết luận chiến lược:** Fixed-size cho kết quả tổng thể ổn định nhất trên năm câu hỏi mới: các dữ kiện cần thiết đều xuất hiện trong top-3, bốn câu có phần hỗ trợ chính ở top-1. Heading-based tốt nhất cho câu hỏi reset mật khẩu vì giữ trọn sáu bước trong một mục. Recursive hợp với câu hỏi phòng học nhóm vì lấy quy trình ở top-1 và thời gian ở top-2. Semantic mặc định chưa phù hợp corpus Markdown này; cần hạ ngưỡng hoặc gom các dòng trong cùng mục trước khi so độ tương tự.

## 3. Năm câu hỏi và chất lượng truy xuất — 10 điểm

### Bộ câu hỏi chung và gold answer

Các câu hỏi được định nghĩa tại `QUERIES` trong [bench.py](../bench.py). Gold answer chỉ dựa trên bản dữ liệu lưu trong repo.

| # | Câu hỏi (Query) | Câu trả lời chuẩn (Gold Answer) | Chunk nào chứa thông tin? |
|---|-------|-------------------------------|--------------------------|
| 1 | Với giáo trình tại phòng 111, được mượn tối đa bao nhiêu cuốn, trong bao lâu và được gia hạn thế nào? | Tối đa 8 cuốn trong 90 ngày, được gia hạn một lần thêm 30 ngày, tổng tối đa 120 ngày. Câu này chạy với metadata_filter={"audience": "student"}. | student-textbook-borrowing, mục Chính sách mượn |
| 2 | Chính sách mượn sách tham khảo tại phòng 102 quy định số lượng, thời hạn và gia hạn như thế nào? | Tối đa 5 cuốn, mượn từ 1 đến 30 ngày và được gia hạn một lần thêm 7 ngày. | student-reference-book-borrowing, mục Chính sách mượn |
| 3 | Phòng học nhóm phục vụ vào thời gian nào và quy trình nhận trả chìa khóa ra sao? | Thứ Hai–thứ Sáu 08:00–21:00, cuối tuần 08:00–16:00; đăng ký, để lại thẻ tại P411 để nhận chìa khóa rồi trả chìa khóa để nhận lại thẻ. | group-study-room, mục Thời gian sử dụng và Quy trình đăng ký |
| 4 | Khi quên mật khẩu tài khoản thư viện, bạn đọc cần thực hiện các bước nào? | Truy cập libopac, chọn Đăng nhập, chọn Quên mật khẩu/Quên mã PIN, nhập mã số thẻ, mở liên kết trong email rồi đặt mật khẩu mới. | library-account, mục Đặt lại mật khẩu |
| 5 | Bạn đọc ngoài HUST có thể đọc toàn văn tài nguyên số không và cần điều kiện gì? | Có, nếu đăng ký thẻ hoặc tài khoản thư viện; họ có thể đọc toàn văn tài liệu số và tải học liệu mở. | digital-resource-faq, mục Quyền truy cập của bạn đọc ngoài HUST |

### Quy trình benchmark tái lập

1. Đọc từng file `.md`, tách frontmatter thành metadata và phần thân thành content. Bộ đọc hỗ trợ metadata phẳng `key: value` của corpus, không phải YAML tổng quát; file lỗi hoặc thiếu trường bắt buộc sẽ báo lỗi.
2. Chunk phần thân; tạo `Document(id=f"{path.stem}#{i}", content=chunk, metadata={**frontmatter, "doc_id": path.stem, ...})`, chỉ số bắt đầu từ 0.
3. Tạo `EmbeddingStore` riêng cho từng chiến lược, nạp chunk và chạy đúng 5 query bằng `search_with_filter(..., top_k=3, metadata_filter=...)`.
4. In câu hỏi, bộ lọc, gold answer, top-3 với `score`, `doc_id`, `chunk_id` và toàn bộ nội dung để đối chiếu. File JSON giữ kết quả chi tiết.

Chạy kiểm tra luồng, không cần API key:

```powershell
$env:PYTHONIOENCODING = 'utf-8'
.\.venv\Scripts\python.exe bench.py --strategy all --backend mock --output report/benchmark_results_mock.json
```

Chạy đủ bốn chiến lược bằng embedding đa ngữ thật sau khi cài dependencies tùy chọn và có model:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-local.txt
.\.venv\Scripts\python.exe bench.py --strategy all --backend local --output report/benchmark_results_local.json
```

Local dùng `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`; lần đầu có thể cần tải model. Benchmark không tự hạ xuống Mock khi local lỗi. Dùng cùng corpus, câu hỏi, bộ lọc và model để so sánh; tham số được ghi trong JSON kết quả.

### Kết quả local: top-3 và đối chiếu gold answer

Cả bốn chiến lược đều đưa đúng `doc_id` lên top-1 ở cả năm câu. Tuy nhiên, đúng tài liệu chưa chắc đủ dữ kiện. Bảng sau chọn chiến lược có ngữ cảnh hữu ích nhất cho từng câu dựa trên nội dung chunk, thứ hạng và score.

| Câu | Chiến lược tốt nhất | Kết quả top-3 | Score chính | Đối chiếu gold answer |
|---|---|---|---:|---|
| 1 | Fixed-size | `student-textbook-borrowing#1` ở top-1 | 0,664974 | Một chunk chứa đủ 8 cuốn, 90 ngày, gia hạn 30 ngày và tổng 120 ngày. |
| 2 | Fixed-size / Recursive | `student-reference-book-borrowing#0` ở top-1 | 0,679896 | Hai chiến lược trả cùng nội dung, đủ 5 cuốn, 1–30 ngày và gia hạn 7 ngày. |
| 3 | Recursive | `group-study-room#1` top-1 và `#0` top-2 | 0,555067; 0,515690 | Top-1 chứa quy trình nhận/trả chìa khóa; top-2 chứa đầy đủ hai khung giờ. |
| 4 | Heading-based | `library-account#0` ở top-1 | 0,633434 | Giữ trọn sáu bước reset mật khẩu trong cùng một chunk. |
| 5 | Fixed-size | `digital-resource-faq#2` ở top-1 | 0,715633 | Một chunk chứa cả điều kiện đăng ký thẻ/tài khoản và quyền đọc, tải học liệu mở. |

Đánh giá dữ kiện thực sự có trong top-3:

| Chiến lược | Đủ gold answer | Một phần | Thiếu dữ kiện | Nhận xét |
|---|---:|---:|---:|---|
| Fixed-size | 5/5 | 0/5 | 0/5 | Câu 3 và 4 cần ghép hai chunk; bốn câu có chunk hỗ trợ chính ở top-1. |
| Recursive | 4/5 | 0/5 | 1/5 | Câu 4 lấy đúng tài liệu nhưng không lấy các bước reset mật khẩu. |
| Heading-based | 4/5 | 1/5 | 0/5 | Câu 3 có quy trình nhưng thiếu chunk thời gian trong top-3. |
| Semantic | 2/5 | 1/5 | 2/5 | Đủ câu 1 và 5; câu 4 chỉ có một phần; câu 2 và 3 chủ yếu trả tiêu đề. |

Theo rubric, kết quả truy xuất tốt nhất của nhóm bao phủ đủ 5/5 gold answer trong top-3. Benchmark chưa gọi LLM nên báo cáo chỉ chấm phần truy xuất và trình bày câu trả lời được tổng hợp trực tiếp từ các chunk; không coi đây là phép đo chất lượng sinh văn bản của agent.

### Vai trò và giới hạn của metadata filter

Câu 1 dùng `audience=student`, thu hẹp từ 8 xuống 2 tài liệu. Kết quả local đưa `student-textbook-borrowing#1` lên top-1 với score 0,664974. Cả giáo trình và sách tham khảo đều có nhãn `student`, nên vẫn cần phân biệt phòng 111/102 và loại sách bằng nội dung. Corpus chưa có chính sách mượn riêng cho `faculty` hoặc `staff`, do đó phép chạy này chứng minh pre-filter hoạt động nhưng chưa đo được khả năng loại một chính sách mượn dành riêng cho đối tượng khác.

Không dùng bộ lọc `student` cho mọi câu: hướng dẫn gia hạn, giờ phục vụ và phòng học nhóm mang nhãn `all`. Bộ lọc hiện tại so khớp chính xác nên sẽ loại các nguồn chung nếu dùng sai. `KnowledgeBaseAgent.answer()` hiện gọi `search()`; benchmark gọi trực tiếp `search_with_filter()` theo yêu cầu, không khẳng định agent tự hỗ trợ lọc metadata.

## 4. Demo và bài học nhóm — 5 điểm

### Kịch bản trình bày

1. Mở file nguồn, chỉ ra frontmatter/phần thân; chạy benchmark và đối chiếu `doc_id#chunk_index` với nội dung gốc.
2. So sánh mục chính sách mượn giáo trình giữa Fixed-size và Heading-based: số liệu có nằm trọn trong chunk không? Trình bày số chunk, độ dài và trường hợp Recursive vượt kích thước mục tiêu.
3. Chạy câu 1 với bộ lọc `student`, giải thích tại sao sách tham khảo vẫn là ứng viên. Sau đó mở `benchmark_results.json` để so sánh top-3 của đủ bốn chiến lược và chỉ ra hiện tượng Semantic bị chia quá nhỏ ở ngưỡng 0,5.

Đây là kịch bản đã chuẩn bị, không phải xác nhận nhóm đã thuyết trình.

### Bài học và hướng cải thiện

Cùng corpus nhưng ranh giới chunk khác nhau làm thay đổi thông tin đưa vào ngữ cảnh. Tìm đúng tài liệu chưa đủ: câu hỏi hạn mức cần đúng mục chính sách, không chỉ một đoạn quy trình trong cùng file. Metadata phải phản ánh đối tượng và loại dịch vụ; không thể bù hoàn toàn cho chunk thiếu ý.

Bước cải thiện tiếp theo là thử Heading-based kết hợp Recursive cho mục dài, bổ sung metadata loại tài liệu/phòng và nguồn công khai có chính sách khác nhau theo đối tượng. Với Semantic, cần chọn threshold trên một tập câu hỏi phát triển riêng hoặc gom dòng theo đoạn/mục trước khi tính cosine; điều chỉnh trực tiếp để thắng năm câu đánh giá sẽ làm sai lệch phép so sánh.

## 5. Tự đánh giá theo bằng chứng hiện có

| Tiêu chí | Bằng chứng và tình trạng | Điểm |
|---|---|---|
| Lựa chọn tài liệu /10 | Đủ 8 tài liệu HUST, metadata và nguồn công khai; `document_version` chưa được nguồn nêu. | 9 / 10 |
| Thiết kế chiến lược /15 | Có mã, cấu hình, ưu/nhược điểm và số liệu local cho đủ 4 chiến lược. | 14 / 15 |
| Chất lượng truy xuất /10 | Chiến lược tốt nhất bao phủ đủ 5/5 gold trong top-3; chưa đánh giá đầu ra LLM. | 9 / 10 |
| Thuyết trình /5 | Có kịch bản demo và file kết quả tái lập; chưa có bằng chứng buổi trình bày thực tế. | 4 / 5 |
| **Tổng /40** | **Tự đánh giá theo bằng chứng trong repo.** | **36 / 40** |

Kiểm thử mã: `python -m pytest tests/ -q` đạt **48/48 tests** (42 tests hiện có và 6 tests benchmark mới). Các test mới kiểm tra tách frontmatter, giữ mục con, ranh giới Semantic với vector kiểm soát, xử lý dòng dài, lọc metadata và xuất kết quả; không thay thế đánh giá embedding thật.
