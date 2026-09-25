# RAG evaluation results

## Run information

| Field                              | Value |
| ---------------------------------- | ----- |
| Evaluation date                    | 2026-09-25 |
| Framework and version              | Ragas 0.4.3 |
| Evaluator model                    | gemini-2.0-flash |
| Generator model                    | gemini-2.0-flash |
| Embedding model                    | BAAI/bge-m3 (1024-dim) |
| Corpus version/commit              | commit f02d145 |
| Golden dataset size                | 15 Q&A pairs |
| `top_k`                            | 5 |
| Fallback threshold and calibration | 0.30 (calibrated on in-domain vs OOD queries) |

## Configurations

- **Config A — dense-only:** ChromaDB cosine similarity search, lấy top 5 chunks điểm cao nhất gửi cho LLM sinh câu trả lời.
- **Config B — hybrid + RRF:** Kết hợp Dense Retrieval (ChromaDB) và Lexical Retrieval (BM25Okapi) qua thuật toán Reciprocal Rank Fusion (k=60), reorder chunks giảm lost-in-the-middle trước khi đưa vào LLM.

Hai config sử dụng cùng golden dataset (15 câu), generator (Gemini 2.0 Flash), prompt và `top_k=5`; chỉ thay retrieval strategy.

## Overall scores

| Metric            | Config A | Config B | Delta B−A |
| ----------------- | -------: | -------: | --------: |
| Faithfulness      |     0.87 |     0.94 |     +0.07 |
| Answer relevance  |     0.84 |     0.92 |     +0.08 |
| Context recall    |     0.80 |     0.93 |     +0.13 |
| Context precision |     0.78 |     0.89 |     +0.11 |
| **Average**       | **0.82** | **0.92** | **+0.10** |

## A/B comparison

- **Cấu hình tốt hơn:** Config B (Hybrid + RRF).
- **Evidence:** 
  - Context Recall tăng vượt bậc từ 0.80 lên 0.93 (+13%) nhờ BM25 bắt chính xác các từ khóa thực thể, tên súng (LMG, RPD, M249, MG3), phím thao tác (phím H, phím F) và số liệu cập nhật bản 43.1.
  - Faithfulness cải thiện (+7%) do thông tin context được xếp hạng chính xác hơn, hạn chế hiện tượng mô hình suy diễn ngoài tài liệu.
- **Trade-off về latency/cost:** 
  - Latency của Config B tăng nhẹ khoảng 35ms (do tính toán BM25 và fusion RRF), hoàn toàn không đáng kể so với thời gian sinh văn bản của LLM (~1.2s).
  - Chi phí token gần như tương đương vì cùng giữ `top_k=5`.

## Worst performers

|   # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage | Root cause |
| --: | -------- | ------ | -----------: | --------: | -----: | --------: | ------------- | ---------- |
|   1 | Độ cao bung dù khẩn cấp tối thiểu thay đổi thế nào? | Config A | 0.75 | 0.80 | 0.60 | 0.70 | retrieval | Dense embedding nhầm lẫn giữa độ cao bung dù khẩn cấp (9m) và độ cao kích hoạt dù tự động sau khi dùng Cứu trợ (250m). |
|   2 | Phím H có thể dùng để làm gì với các vật phẩm đã triển khai? | Config A | 0.80 | 0.85 | 0.65 | 0.72 | retrieval | Từ khóa "Phím H" bị phân tán ngữ nghĩa trong không gian vector dày đặc. BM25 ở Config B đã khắc phục triệt để lỗi này. |
|   3 | Đòn đá cận chiến mới có thể phá hủy những vật thể nào? | Config B | 0.90 | 0.88 | 0.85 | 0.82 | generation | Ngữ cảnh liệt kê nhiều loại vật thể, câu trả lời bỏ sót chi tiết chai lọ nhỏ dù context có đề cập. |

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
| -------: | ------ | ------------------------------ | --------------- | ------------- |
|        1 | Bổ sung từ điển từ đồng nghĩa viết tắt (ADS, LMG, Scope) | Câu hỏi về súng máy LMG đôi khi truy xuất lẫn súng trường | Tăng Context Precision thêm ~5% | Đánh giá lại trên tập câu hỏi về vũ khí |
|        2 | Tinh chỉnh chunk overlap ở các bảng thông số bản cập nhật | Một số bảng cập nhật bị tách giữa chừng khiến ngữ cảnh đứt quãng | Tăng Context Recall thêm ~4% | Kiểm tra các chunk thuộc mục Patch Note 43.1 |
|        3 | Thêm few-shot example vào system prompt để trích xuất đầy đủ danh sách | Câu hỏi liệt kê vật thể bị sót 1 vài chi tiết nhỏ | Tăng Faithfulness và Answer Relevance | Chạy lại eval với 3 câu hỏi dạng liệt kê |

## Bonus experiments

| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |
| ---------- | -------- | -----------: | -----------------: | ---------- |
| Reranker với Cross-Encoder | Config B | Average +0.02 | Latency +120ms | Cải thiện nhẹ độ chính xác nhưng tăng độ trễ đáng kể |
