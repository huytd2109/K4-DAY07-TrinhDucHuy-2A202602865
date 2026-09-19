# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Trịnh Đức Huy
**Nhóm:** phoboi
**Ngày:** 19/09/2026

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Độ tương tự cosine cao (tiến gần đến 1.0) nghĩa là hai vector embedding cùng hướng trong không gian vector đa chiều (góc giữa hai vector xấp xỉ 0 độ). Về mặt ngữ nghĩa, điều này thể hiện hai đoạn văn bản có độ tương đồng ý nghĩa rất cao, diễn đạt cùng một thông điệp cốt lõi bất kể độ dài hay số lượng từ vựng khác nhau.

**Ví dụ có độ tương tự CAO:**
- Câu A: "Thư viện mở cửa phục vụ bạn đọc từ thứ Hai đến thứ Sáu hàng tuần."
- Câu B: "Thời gian đón độc giả của trung tâm thông tin thư viện là các ngày trong tuần trừ thứ Bảy và Chủ Nhật."
- Tại sao tương đồng: Hai câu sử dụng các từ ngữ khác nhau (bạn đọc vs độc giả, thứ Hai đến thứ Sáu vs các ngày trong tuần trừ cuối tuần) nhưng cùng diễn đạt trọn vẹn một sự thật thực tế về lịch hoạt động của thư viện.

**Ví dụ có độ tương tự THẤP:**
- Câu A: "Sinh viên chính quy được mượn tối đa 8 cuốn giáo trình trong thời hạn 90 ngày."
- Câu B: "Hôm nay thời tiết Hà Nội chuyển lạnh và có mưa rào rải rác trên diện rộng."
- Tại sao khác: Hai câu thuộc hai lĩnh vực tri thức hoàn toàn độc lập (quy định học vụ thư viện vs dự báo thời tiết), không chia sẻ ngữ cảnh hay khái niệm ngữ nghĩa chung nào.

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Khoảng cách Euclid bị chi phối bởi độ dài (magnitude) của vector — vốn tỉ lệ thuận với độ dài văn bản hoặc tần suất xuất hiện của từ — khiến hai đoạn văn bản cùng chủ đề nhưng khác độ dài bị xem là xa nhau. Ngược lại, Cosine similarity chỉ đo góc định hướng giữa hai vector (chuẩn hóa độ dài về 1), phản ánh bản chất ngữ nghĩa thuần túy độc lập với độ dài văn bản.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:*
> - Bước nhảy giữa các chunk liên tiếp: $\text{step} = \text{chunk\_size} - \text{overlap} = 500 - 50 = 450$ ký tự.
> - Chunk đầu tiên bao phủ đoạn từ ký tự 0 đến 500.
> - Số ký tự còn lại cần bao phủ: $10.000 - 500 = 9.500$ ký tự.
> - Số chunk kế tiếp: $\lceil 9.500 / 450 \rceil = \lceil 21,11 \rceil = 22$ chunk.
> - Tổng số chunk tạo thành: $1 + 22 = 23$ chunk.
> - Kiểm chứng theo công thức tổng quát: $N = \lceil (L - \text{overlap}) / (\text{chunk\_size} - \text{overlap}) \rceil = \lceil (10.000 - 50) / (500 - 50) \rceil = \lceil 9.950 / 450 \rceil = \lceil 22,11 \rceil = 23$ chunk (khớp hoàn toàn với hàm `FixedSizeChunker(chunk_size=500, overlap=50).chunk("a" * 10000)` trong code).
> *Đáp án:* **23 chunks**.

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> Khi overlap tăng lên 100, bước nhảy giảm xuống còn $500 - 100 = 400$ ký tự, số lượng chunk tăng từ 23 lên $\lceil (10.000 - 100) / 400 \rceil = \lceil 9.900 / 400 \rceil = 25$ chunks (tăng thêm 2 chunks). Ta muốn tăng độ chồng chéo vì overlap lớn hơn giúp hạn chế tối đa nguy cơ một câu văn, định nghĩa hoặc số liệu quan trọng bị cắt đứt gãy ở ranh giới giữa hai chunk kế tiếp, đảm bảo bảo toàn ngữ cảnh ngữ nghĩa trọn vẹn cho quá trình truy xuất.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Thuật toán sử dụng biểu thức chính quy positive lookbehind `r"(?<=[.!?])\s+|(?<=\n)\s*"` để tách văn bản tại vị trí ngay sau dấu kết câu (`.`, `!`, `?`) hoặc ký tự xuống dòng mà không làm mất dấu câu ở cuối mỗi câu. Sau khi tách và làm sạch các câu rỗng, hàm gom tuần tự từng nhóm `max_sentences_per_chunk` câu lại với nhau bằng `join`. Xử lý tốt các edge case chuỗi rỗng/chỉ chứa khoảng trắng (trả về `[]`), đồng thời nhận diện được hạn chế chưa xử lý hoàn hảo đối với các từ viết tắt có dấu chấm (`TS.`, `ThS.`, `v.v.`) hoặc số thập phân (`3.14`) khi theo sau là khoảng trắng.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> Thuật toán hoạt động theo nguyên lý đệ quy 2 chiều: ưu tiên tách theo cấu trúc ngữ nghĩa lớn trước theo danh sách separator `["\n\n", "\n", ". ", " ", ""]`; nếu mảnh văn bản sau khi tách vẫn vượt quá `chunk_size` thì gọi đệ quy `_split` với bộ separator cấp con còn lại, sau đó có pha gom (merge) các mảnh nhỏ kế tiếp nhau để giữ kích thước chunk sát với mục tiêu, tránh sinh ra chunk quá vụn (5–10 ký tự). Các trường hợp cơ sở (base cases) gồm: chuỗi rỗng trả về `[]`, chuỗi có độ dài `<= chunk_size` trả về `[cleaned]`, và khi hết separator (`remaining_separators == []`) thì fallback chia cứng thành các đoạn có độ dài `chunk_size`.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> Lưu trữ in-memory đơn giản và tin cậy dưới dạng danh sách `self._store` chứa các dict record chuẩn hóa gồm `id`, `content`, `metadata` (sao chép dictionary để tránh side-effect và bảo đảm luôn có trường `doc_id` của tài liệu gốc) cùng vector `embedding` sinh từ `self._embedding_fn`. Khi tìm kiếm (`search`), hàm gọi `_search_records` để nhúng query, tính cosine similarity với từng record qua hàm `compute_similarity`, sắp xếp giảm dần theo điểm `score` và trả về `top_k` kết quả (lược bỏ trường vector embedding để output sạch sẽ).

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> Áp dụng cơ chế lọc trước (pre-filtering) trong `search_with_filter`: duyệt qua `self._store` và chỉ giữ lại các record thỏa mãn toàn bộ các cặp key-value trong `metadata_filter` trước khi tính similarity, đảm bảo tài liệu không phù hợp không chiếm mất các vị trí trong `top_k`. Hàm `delete_document` dùng list comprehension để lọc bỏ toàn bộ các bản ghi có `metadata["doc_id"] == doc_id`, so sánh kích thước store trước và sau khi xóa để trả về `True` (nếu có chunk bị xóa) hoặc `False` (nếu không tìm thấy).

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> Hàm kiểm tra tính hợp lệ của câu hỏi và kích thước kho dữ liệu (trả lời thông báo phù hợp nếu rỗng); sau đó gọi `store.search(question, top_k)` để thu thập các đoạn trích liên quan nhất. Ngữ cảnh được định dạng thành các khối có đánh số `[{index}] Source: {source}\n{content}` nhằm tối ưu hóa khả năng truy vết nguồn gốc (source traceability). Prompt chỉ thị rõ ràng: "Use only the following retrieved context to answer... If the answer is not present, say that the information is not available", giúp ngăn chặn tình trạng mô hình tự bịa thông tin ngoài ngữ cảnh được cung cấp.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```
============================= test session starts =============================
platform win32 -- Python 3.13.15, pytest-9.1.1, pluggy-1.6.0 -- D:\aiinaction\K4-DAY07-TrinhDucHuy-2A202602865\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: D:\aiinaction\K4-DAY07-TrinhDucHuy-2A202602865
collecting ... collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED   [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED    [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED   [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================= 42 passed in 0.09s ==============================
```

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Thư viện mở cửa phục vụ sinh viên từ thứ Hai đến thứ Sáu. | Thời gian đón bạn đọc của thư viện là các ngày trong tuần trừ thứ Bảy và Chủ Nhật. | cao | 0.892 | Đúng |
| 2 | Sinh viên chính quy được mượn tối đa 8 cuốn giáo trình. | Hạn mức mượn tài liệu học tập của sinh viên là 8 quyển sách. | cao | 0.865 | Đúng |
| 3 | Mức phạt tiền khi trả sách quá hạn tại thư viện là 5.000 đồng một ngày. | Thư viện trang bị hệ thống máy tính hiện đại tại tầng 2. | thấp | 0.148 | Đúng |
| 4 | Bạn đọc không được tự ý cài đặt phần mềm lạ vào máy tính thư viện. | Hôm nay thời tiết Hà Nội chuyển lạnh và có mưa rào rải rác. | thấp | 0.038 | Đúng |
| 5 | Quy trình gia hạn tài liệu thư viện trực tuyến qua hệ thống OPAC. | Hướng dẫn thực hiện kéo dài thời gian mượn sách qua cổng thông tin. | cao | 0.877 | Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Điểm bất ngờ và thú vị nhất là ở Cặp 1 và Cặp 5: mặc dù hai câu hoàn toàn không lặp lại các từ khóa cốt lõi (ví dụ "từ thứ Hai đến thứ Sáu" so với "các ngày trong tuần trừ thứ Bảy và Chủ Nhật", hay "gia hạn tài liệu" so với "kéo dài thời gian mượn sách"), mô hình embedding ngữ nghĩa vẫn cho điểm tương tự cosine rất cao (> 0.86). Điều này khẳng định rằng text embeddings biểu diễn ngữ nghĩa không dựa trên so khớp chuỗi ký tự hay từ vựng bề mặt (lexical matching), mà ánh xạ văn bản vào một không gian hình học đa chiều liên tục (latent semantic space), nơi các khái niệm đồng nghĩa và tương đương ngữ nghĩa được định vị gần nhau về phương hướng vector.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân của bạn trong gói `src`. **5 câu hỏi này phải trùng với các thành viên cùng nhóm** (xem `REPORT_NHOM.md`).

**Chiến lược cá nhân:** Semantic Chunking, `chunk_size=500`, `semantic_threshold=0.5`, dùng model local `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. Chiến lược sinh 156 chunk trên 8 tài liệu. Bảng dưới dùng kết quả thật trong `report/benchmark_results.json`; cột trả lời là phần có thể tổng hợp từ top-3, không phải đầu ra của một LLM.

| # | Câu hỏi (Query) | Top-1 | Top-2 | Top-3 | Đánh giá và câu trả lời từ context |
|---|-------|-------|-------|-------|------------------------|
| 1 | Với giáo trình tại phòng 111, được mượn tối đa bao nhiêu cuốn, trong bao lâu và được gia hạn thế nào? | `student-textbook-borrowing#0` — 0,650083 — chỉ có tiêu đề tài liệu. | `student-textbook-borrowing#9` — 0,591399 — hạn mức 8 cuốn. | `student-textbook-borrowing#10` — 0,552258 — 90 ngày, gia hạn 30 ngày, tổng 120 ngày. | **Đủ dữ kiện khi ghép top-2 và top-3:** 8 cuốn; 90 ngày; gia hạn 1 lần thêm 30 ngày, tổng tối đa 120 ngày. |
| 2 | Chính sách mượn sách tham khảo tại phòng 102 quy định số lượng, thời hạn và gia hạn như thế nào? | `student-reference-book-borrowing#7` — 0,646336 — chỉ có tiêu đề chính sách. | `student-reference-book-borrowing#0` — 0,633998 — chỉ có tiêu đề tài liệu. | `document-renewal#1` — 0,579539 — tiêu đề quy trình gia hạn, sai tài liệu. | **Failure:** không có các dữ kiện 5 cuốn, 1–30 ngày và gia hạn 7 ngày trong top-3. |
| 3 | Phòng học nhóm phục vụ vào thời gian nào và quy trình nhận trả chìa khóa ra sao? | `group-study-room#7` — 0,744599 — chỉ có tiêu đề quy trình. | `group-study-room#0` — 0,551507 — chỉ có giới thiệu dịch vụ. | `student-reference-book-borrowing#2` — 0,533177 — tiêu đề vị trí/giờ phòng 102, sai tài liệu. | **Failure:** không có khung giờ hoặc các bước nhận/trả chìa khóa trong top-3. |
| 4 | Khi quên mật khẩu tài khoản thư viện, bạn đọc cần thực hiện các bước nào? | `library-account#10` — 0,666802 — tiêu đề đăng nhập và đổi mật khẩu. | `library-account#2` — 0,663417 — tiêu đề reset mật khẩu. | `library-account#7` — 0,640889 — email đặt lại và bước 5. | **Có một phần:** biết hệ thống gửi liên kết và người dùng mở email, nhưng thiếu bước 1–4 và bước 6. |
| 5 | Bạn đọc ngoài HUST có thể đọc toàn văn tài nguyên số không và cần điều kiện gì? | `digital-resource-faq#6` — 0,671166 — quyền đọc toàn văn và tải học liệu mở. | `digital-resource-faq#5` — 0,661946 — điều kiện đăng ký thẻ/tài khoản. | `digital-resource-faq#4` — 0,639515 — quyền của người học HUST, liên quan thấp. | **Đủ dữ kiện từ top-1 và top-2:** có; cần đăng ký thẻ/tài khoản, sau đó được đọc toàn văn và tải học liệu mở. |

**Bao nhiêu câu hỏi trả về chunk có dữ kiện liên quan trong top-3?** 3 / 5. Trong đó câu 1 và 5 đủ gold answer; câu 4 chỉ đủ một phần. Câu 2 và 3 trả đúng tài liệu nhưng chỉ lấy tiêu đề, nên không tính là truy xuất thành công về nội dung.

Kết quả console rút gọn của cá nhân được lưu tại [`ket_qua_benchmark.txt`](../ket_qua_benchmark.txt).

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> So sánh với Fixed-size và Heading-based cho thấy đúng `doc_id` không đồng nghĩa với có đủ ngữ cảnh trả lời. Ngưỡng Semantic 0,5 tách Markdown thành 156 mảnh rất nhỏ, khiến tiêu đề có score cao nhưng các con số và bước thực hiện bị đẩy ra ngoài top-3. Với corpus này, nên gom các dòng trong cùng section trước khi xét semantic boundary hoặc chọn threshold trên một tập phát triển riêng.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 4 / 10 |
| **Tổng phần cá nhân** | **54 / 60** |
