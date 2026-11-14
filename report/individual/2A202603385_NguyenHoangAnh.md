# Báo cáo cá nhân — Day 07: Retrieval & Context Engineering (Agentic RAG)

**Học viên**: Nguyễn Hoàng Anh  
**MSSV**: 2A202603385  
**Lớp**: K5-L3A  
**Corpus**: *U.S. History* (OpenStax, Đại học Rice)  
**Repository**: [K5-L3A-DAY07-NguyenHoangAnh-2A202603385-RetrievalContext](https://github.com/hoanganh-nguyen/K5-L3A-DAY07-NguyenHoangAnh-2A202603385-RetrievalContext)

---

## 1. Dữ liệu và Benchmark

### 1.1. Nguồn dữ liệu, bản quyền, số trang và số chunk
- **Nguồn tài liệu**: Tác phẩm giáo khoa *U.S. History* do Nhà xuất bản OpenStax (Đại học Rice) phát hành.
- **Bản quyền**: Giấy phép mở Creative Commons Attribution Non-Commercial ShareAlike 4.0 International License (`CC BY-NC-SA 4.0`).
- **Phạm vi trích xuất**: Trang vật lý 23 đến 100 trong file `data/US_History_-_WEB.pdf`, gồm trọn vẹn 3 chương đầu (giai đoạn tiền Columbus 1492 đến năm 1700), tổng cộng **78 trang** với 231,021 ký tự nội dung sạch.
- **Kế thừa Chunking Day 06**: Tái sử dụng nguyên vẹn Strategy B (Structure / Paragraph-aware) gồm **349 chunks** có metadata chi tiết (`document_id`, `source`, `page`, `chapter`, `section`). Dữ liệu được quản lý tập trung tại `data/chunks.json`.

### 1.2. Vector store, Embedding model và chi phí
- **Embedding Model**: OpenAI `text-embedding-3-small`, số chiều **1,536 dimensions**, metric tính độ tương đồng là Cosine Similarity.
- **Lưu trữ & Truy xuất**: 349 vector embeddings đã được tính toán thật qua API và lưu trữ đệm trong `data/chunk_embeddings.json` nhằm đảm bảo tính tái lập 100%, tốc độ truy vấn mili-giây và độc lập môi trường.
- **Chi phí Embedding**: Tiêu tốn ~70,000 tokens (~$0.0014 USD).

### 1.3. Agent Model, cơ chế điều phối và phân chia Dev / Test
- **Agent LLM**: OpenAI `gpt-4o-mini` (official API), tham số `temperature = 0.0` và kích hoạt chế độ `response_format={"type": "json_object"}` để đảm bảo kiểm soát hoàn toàn tính tất định.
- **Bộ benchmark**: Tổng cộng **24 câu hỏi** được đóng băng tại `data/questions.json` và `data/agent_cases.json`, phân bổ theo quy định chuẩn:
  - **Tập Dev (17 câu)**: Gồm **15 câu có đáp án** (`q01`–`q15`) và **2 câu thiếu bằng chứng** (`q21`, `q22`).
  - **Tập Test (7 câu)**: Gồm **5 câu có đáp án** (`q16`–`q20`) và **2 câu thiếu bằng chứng** (`q23`, `q24`).

### 1.4. Tiêu chí xác định câu thiếu bằng chứng và quy tắc chấm
- **Câu thiếu bằng chứng (`unanswerable`)**: Được thiết kế liên quan chặt chẽ đến chủ đề lịch sử Hoa Kỳ nhưng thuộc các mốc sự kiện cuối thế kỷ 18 và đầu thế kỷ 19 (sau năm 1700, ví dụ: Tuyên ngôn Độc lập 1776, Hiệp ước Paris 1783, Hiến pháp 1787, Thỏa thuận Louisiana 1803). Toàn bộ từ khóa của các câu này đều được quét kiểm tra vắng mặt trong 349 chunks.
- **Tiêu chí chấm**:
  - *Tool use*: Tỷ lệ câu có gọi công cụ tìm kiếm trên tổng số câu.
  - *Grounded answer* (0–2): 0 = sai, ảo giác hoặc từ chối nhầm câu có bằng chứng; 1 = đúng một phần; 2 = đúng, đủ và có bằng chứng hỗ trợ trực tiếp.
  - *Citation hợp lệ*: Tỷ lệ trích dẫn `[chunk-id]` vừa xuất hiện trong context vừa thực sự hỗ trợ cho khẳng định.
  - *Abstention đúng*: Khả năng từ chối trả lời ("Không đủ bằng chứng để trả lời.") trên câu thiếu evidence, không đưa ra khẳng định vô căn cứ.

---

## 2. Retrieval và Thử nghiệm

### 2.1. Bảng kết quả so sánh các phương pháp Retrieval trên Dev và Test

| Cấu hình | Phương pháp, Top-k, Filter, Context Budget | Giả thuyết / Thay đổi | Split | Hit@1 (n/N) | Evidence Hit@3 (n/N) | Context Evidence Rate (n/N) | Thời gian median / Chi phí |
|---|---|---|:---:|:---:|:---:|:---:|:---:|
| **Baseline** | TF-IDF (Lexical), top_k=3, budget=800 | Khớp từ khóa truyền thống | dev | 9/15 (60.0%) | 11/15 (73.33%) | 6/15 (40.0%) | 0.98 ms / $0.00 |
| **Phương pháp 2** | Dense (`text-embedding-3-small`), top_k=3, budget=800 | Tìm kiếm theo khoảng cách ngữ nghĩa | dev | 8/15 (53.33%) | **13/15 (86.67%)** | 3/15 (20.0%) | 498.7 ms / <$0.0001 |
| **Phương pháp 3** | Hybrid RRF (Dense + TF-IDF), top_k=3, budget=800 | Kết hợp điểm hạng RRF ($k=60$) | dev | **10/15 (66.67%)** | 12/15 (80.0%) | 4/15 (26.67%) | 516.0 ms / <$0.0001 |
| **Cải thiện 1** | Dense, top_k=3, **budget=1600** | Mở rộng budget để chứa đủ top-3 chunks | dev | 8/15 (53.33%) | **13/15 (86.67%)** | **13/15 (86.67%)** | 509.4 ms / <$0.0001 |
| **Cải thiện 2** | Hybrid RRF, top_k=3, **budget=1600** | Kết hợp ưu thế Hit@1 của RRF & budget lớn | dev | **10/15 (66.67%)** | 12/15 (80.0%) | **12/15 (80.0%)** | 498.9 ms / <$0.0001 |
| **Baseline Test** | Dense, top_k=3, budget=800 | Baseline đánh giá trên tập độc lập | test | 4/5 (80.0%) | 4/5 (80.0%) | 1/5 (20.0%) | 473.0 ms / <$0.0001 |
| **Cấu hình chọn** | **Hybrid RRF, top_k=3, budget=1600** | **Cấu hình tối ưu được chốt từ Dev** | **test** | **5/5 (100%)** | **5/5 (100%)** | **5/5 (100%)** | **462.8 ms / <$0.0001** |

---

### 2.2. Phân tích 2 lỗi thực nghiệm trên Dev, nguyên nhân và cách khắc phục

#### Lỗi 1: Nghẽn ngữ cảnh do giới hạn Context Budget (Context Bottleneck)
- **Hiện tượng**: Tại cấu hình mặc định (`budget = 800`), mặc dù Dense Retrieval đạt `Hit@3` tới **86.67%** (13/15 câu), nhưng `Context Evidence Rate` lại giảm mạnh xuống chỉ còn **20.0%** (3/15 câu).
- **Nguyên nhân**: Các chunk trích xuất theo đoạn văn (Strategy B) dài trung bình ~600 ký tự. Cùng với cấu trúc định danh nguồn `[{cid}] (trang {page}) ...`, một chunk đơn lẻ chiếm ~650–700 ký tự. Do đó, hàm `pack_context` chỉ nhét vừa chunk xếp hạng 1, toàn bộ các chunk xếp hạng 2 và 3 chứa evidence đúng đều bị cắt bỏ.
- **Biện pháp & Đo lường sau sửa đổi**: Tăng Context Budget lên **1,600 ký tự** (đủ cho 2–3 chunks). Kết quả: `Context Evidence Rate` của Dense tăng vọt từ **20.0% lên 86.67%** (13/15 câu có evidence trong top-3 đều vào trọn context), giải phóng hoàn toàn nút thắt cổ chai.

#### Lỗi 2: Loãng từ khóa truy vấn (Lexical Dilution) trên câu hỏi `q12`
- **Hiện tượng**: Câu hỏi `q12`: *"What is recognized as the oldest continuously inhabited European city in North America, established in 1565?"*. Cả Dense và Hybrid đều xếp các đoạn văn về Pueblo societies và thành phố cổ Cahokia (trang 44) lên trên St. Augustine (trang 75).
- **Nguyên nhân**: Tần suất xuất hiện cao của cụm từ rộng *"North America"* và *"city"* trong tài liệu đã làm loãng vector truy vấn, kéo kết quả về các nền văn hóa bản địa Bắc Mỹ thay vì định vị vào từ khóa bản lề mang tính phân loại cao là *"1565"* và *"European settlement"*.
- **Bài học kinh nghiệm**: Việc kết hợp Hybrid RRF đã nâng điểm các câu có danh từ riêng đặc thù (giúp Hit@1 đạt 66.67%), tuy nhiên với câu chứa nhiều khái niệm bao trùm, cần chiến lược viết lại truy vấn (Query Rewriting) hoặc metadata filtering theo thế kỷ.

---

### 2.3. Tiêu chí lựa chọn cấu hình cuối cùng
**Cấu hình được chọn: Hybrid RRF Retriever + Context Budget 1,600 ký tự**
- Đạt `Hit@1` cao nhất (66.67% trên Dev, 100% trên Test).
- Tận dụng sức mạnh đa dạng: Dense nắm bắt ngữ nghĩa câu hỏi, TF-IDF giữ độ chuẩn xác cho các thuật ngữ thực thể cổ (như *quipu*, *patroonships*, *encomienda*).
- Đảm bảo Context Evidence Rate cao mà không làm bùng nổ token ngữ cảnh (chỉ ~1,300 ký tự context).

---

## 3. Agent và Đánh giá

### 3.1. Thiết kế Agent, Tool use, điều kiện dừng và quản lý ngữ cảnh
- **Cơ chế hoạt động**: Sử dụng `RealLLMController` điều phối vòng lặp `run_agent`:
  1. *Lượt 1*: Nhận diện câu hỏi, trích xuất từ khóa trọng tâm và phát lệnh `search(query)`.
  2. *Lượt 2*: Kiểm tra context trả về qua `pack_context(candidates, budget=1600)`.
     - Nếu có bằng chứng: Tổng hợp câu trả lời khách quan, chèn mã trích dẫn `[chunk-id]` ngay sau từng luận điểm và trả về danh sách `citations`.
     - Nếu không có bằng chứng (câu hỏi ngoài giai đoạn 1492–1700): Chủ động từ chối với câu chuẩn *"Không đủ bằng chứng để trả lời."* và trả về `citations: []`.
- **Bảo vệ chống lặp & Giới hạn**: Khóa số lượt tìm kiếm tối đa `max_searches = 3`. Nếu sau 3 lần tìm không đủ context, Agent tự động dừng và abstain.

---

### 3.2. Bảng chỉ số Agent trên Dev và Test

| Cấu hình / Split | Tool use | Số lượt gọi TB | Grounded answer (0 / 1 / 2) | Điểm TB Grounded (/2.0) | Citation hợp lệ | Câu trả lời có citation | Abstention đúng | Từ chối nhầm (False Abstention) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline (Dense 800) / DEV** | 17/17 (100%) | 2.0 | [8, 0, 7] | 0.93 / 2.0 | 7 / 7 (100%) | 7 / 7 (100%) | 2 / 2 (100%) | 8 / 15 (53.3%) |
| **Cải thiện 1 (Dense 1600) / DEV** | 17/17 (100%) | 2.0 | [2, 0, 13] | 1.73 / 2.0 | 17 / 17 (100%) | 13 / 13 (100%) | 2 / 2 (100%) | 2 / 15 (13.3%) |
| **Cải thiện 2 (Hybrid 1600) / DEV**| 17/17 (100%) | 2.0 | [2, 0, 13] | 1.73 / 2.0 | 15 / 15 (100%) | 13 / 13 (100%) | 2 / 2 (100%) | 2 / 15 (13.3%) |
| **Baseline (Dense 800) / TEST** | 7/7 (100%) | 2.0 | [3, 0, 2] | 0.80 / 2.0 | 2 / 2 (100%) | 2 / 2 (100%) | 2 / 2 (100%) | 3 / 5 (60.0%) |
| **Cấu hình chọn (Hybrid 1600) / TEST** | **7/7 (100%)** | **2.0** | **[0, 0, 5]** | **2.00 / 2.0 (100%)** | **5 / 5 (100%)** | **5 / 5 (100%)** | **2 / 2 (100%)** | **0 / 5 (0%)** |

---

### 3.3. Bảng Audit chi tiết toàn bộ 7 câu trên tập Test (Cấu hình đã chọn)

Toàn bộ 7 câu trên tập Test được kiểm tra đối chiếu trực tiếp giữa câu trả lời, chunk trích dẫn và nội dung sách gốc:

| Query ID | Loại câu | Tool call và Kết quả | Câu trả lời và Citation | Nhãn & Lý do kiểm chứng |
|:---:|:---:|---|---|:---:|
| **`q16`** | Có đáp án | `search("Mayflower Compact 1620 English Pilgrims agreement")` $\rightarrow$ 3 chunks | The fundamental governance agreement drafted and signed by the English Pilgrims aboard the Mayflower in 1620 is known as the Mayflower Compact, which has been called the first American governing document `[openstax-us-history_p088_para_001]`. | **2 / 2**: Đúng và đủ. Chunk trang 88 ghi rõ thỏa thuận Mayflower Compact năm 1620. |
| **`q17`** | Có đáp án | `search("religious figures expelled Massachusetts Bay 1630s Puritan")` $\rightarrow$ 3 chunks | The outspoken religious figures expelled from the Massachusetts Bay colony during the 1630s for defying Puritan orthodoxy included Roger Williams and Anne Hutchinson `[openstax-us-history_p090_para_001]`. | **2 / 2**: Khớp hoàn toàn. Chunk trang 90 xác nhận Roger Williams và Anne Hutchinson bị trục xuất. |
| **`q18`** | Có đáp án | `search("Wampanoag leader Metacom King Philip War grievance 1675")` $\rightarrow$ 3 chunks | Wampanoag leader Metacom launched King Philip's War primarily to stop the encroachment of Puritans into Native lands, as he was determined to drive the English from the territory `[openstax-us-history_p091_para_002]`. | **2 / 2**: Khớp chuẩn xác. Chunk trang 91 nêu rõ nguyên nhân ngăn chặn sự lấn chiếm đất đai. |
| **`q19`** | Có đáp án | `search("epidemic disease casualties Huron Iroquois Great Lakes")` $\rightarrow$ 3 chunks | The epidemic disease that caused massive casualties among indigenous peoples such as the Huron and Iroquois around the Great Lakes was smallpox `[openstax-us-history_p095_para_002]`. | **2 / 2**: Khớp nguyên văn. Chunk trang 95 xác định dịch bệnh đậu mùa (smallpox). |
| **`q20`** | Có đáp án | `search("1680 Pueblo Revolt military outcomes New Mexico")` $\rightarrow$ 3 chunks | The immediate military outcomes of the 1680 Pueblo Revolt included the killing of over four hundred Spaniards and the expulsion of the remaining settlers, with perhaps as many as two thousand fleeing south toward Mexico `[openstax-us-history_p076_para_002]`. | **2 / 2**: Đầy đủ số liệu. Chunk trang 76 ghi nhận >400 lính Tây Ban Nha thiệt mạng và ~2,000 người tháo chạy. |
| **`q23`** | Thiếu bằng chứng | `search("primary author Federalist Papers Alexander Hamilton John Jay")` $\rightarrow$ 3 chunks | *"Không đủ bằng chứng để trả lời."* | **Đạt (Abstain đúng)**: Sự kiện soạn thảo Federalist Papers diễn ra năm 1787–1788 (Chương 7), không có trong 78 trang đầu. Agent từ chối chuẩn xác. |
| **`q24`** | Thiếu bằng chứng | `search("United States pay France 1803 Louisiana Purchase")` $\rightarrow$ 3 chunks | *"Không đủ bằng chứng để trả lời."* | **Đạt (Abstain đúng)**: Thỏa thuận Louisiana Purchase diễn ra năm 1803, không có trong corpus. Agent không bịa đặt số liệu. |

---

### 3.4. Dữ liệu kiểm chứng, lệnh thực thi, thời gian và chi phí
- **Đường dẫn artifacts kiểm chứng**:
  - Traces và Summary Test đã chọn: [results/agent-test-selected/traces.json](../../results/agent-test-selected/traces.json) & [summary.json](../../results/agent-test-selected/summary.json)
  - Traces và Summary Test Baseline: [results/agent-test-baseline/traces.json](../../results/agent-test-baseline/traces.json) & [summary.json](../../results/agent-test-baseline/summary.json)
  - Traces Dev các cấu hình: [results/agent-dev-hybrid-1600/](../../results/agent-dev-hybrid-1600/)
- **Lệnh thực thi tái lập**:
  ```powershell
  # Chạy benchmark retrieval 3 phương pháp
  python starter/benchmark_retrieval.py --method all --split dev --budget 1600
  python starter/benchmark_retrieval.py --method all --split test --budget 1600

  # Chạy Agent trên tập test với cấu hình đã chọn
  python starter/run_agent_benchmark.py --split test --retriever hybrid --budget 1600 --output results/agent-test-selected
  ```
- **Tổng tài nguyên tiêu hao**:
  - Toàn bộ quá trình chạy thử nghiệm cho cả Dev (17 câu) và Test (7 câu) chỉ tiêu tốn **0.0067 USD** (~170 VNĐ).
  - Tốc độ xử lý trung bình đạt **3.5 giây/câu hỏi**.

### 3.5. Đánh đổi (Trade-offs) và Giới hạn hệ thống
- **Đánh đổi giữa Context Budget và Chi phí token**: Nâng budget từ 800 lên 1,600 ký tự làm tăng nhẹ số lượng prompt tokens (~15%), tuy nhiên hiệu quả mang lại vượt trội: giảm False Abstention từ 53.3% xuống 13.3% trên Dev và đạt 0% trên Test. Đây là sự đánh đổi hoàn toàn xứng đáng.
- **Giới hạn**: Hệ thống hiện tại áp dụng trích xuất đơn chặng (single-hop search). Với những câu hỏi đa chặng (multi-hop) phức tạp đòi hỏi tổng hợp từ các chương cách xa nhau, cần phát triển thêm cơ chế Multi-turn Query Decomposition.

---