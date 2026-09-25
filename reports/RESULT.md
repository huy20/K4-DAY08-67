# RAG evaluation results

## Run information

| Field                              | Value |
| ---------------------------------- | ----- |
| Evaluation date                    | 2026-09-25 |
| Framework and version              | Python 3.10, ChromaDB 1.5.9, Google GenAI SDK 2.25.0, rank-bm25 0.2.2 |
| Evaluator model                    | gemini-flash-lite-latest (Gemini API LLM-as-a-Judge) |
| Generator model                    | gemini-flash-lite-latest |
| Embedding model                    | gemini-embedding-001 (dimension 3072) |
| Corpus version/commit              | c17c060 (488 chunks từ 9 tài liệu pháp lý và tin tức chuẩn hóa) |
| Golden dataset size                | 18 test cases grounded |
| `top_k`                            | 5 |
| Fallback threshold and calibration | SCORE_THRESHOLD = 0.3 (calibrated cosine similarity với PageIndex fallback) |

## Configurations

- **Config A — dense-only:** Truy xuất ngữ nghĩa thuần túy (Semantic Search) qua ChromaDB Vector Store sử dụng cosine distance chuyển đổi sang similarity qua `1.0 - distance`, lấy top-5 chunks có điểm số tương đồng cao nhất.
- **Config B — hybrid + RRF:** Kết hợp đồng thời Semantic Search (ChromaDB) và Lexical Search (BM25Okapi với IDF ổn định RobustBM25Okapi), sau đó dung hợp thứ hạng bằng thuật toán Reciprocal Rank Fusion (RRF với k=60) để chọn ra top-5 chunks tối ưu.

Hai config phải dùng cùng golden dataset, generator, evaluator, prompt và `top_k`; chỉ thay retrieval strategy.

## Overall scores

| Metric            | Config A | Config B | Delta B−A |
| ----------------- | -------: | -------: | --------: |
| Faithfulness      |   0.9111 |   0.9444 |   +0.0333 |
| Answer relevance  |   0.8722 |   0.9111 |   +0.0389 |
| Context recall    |   0.8500 |   0.8500 |   +0.0000 |
| Context precision |   0.4444 |   0.4111 |   -0.0333 |
| **Average**       |   0.7694 |   0.7792 |   +0.0098 |

## A/B comparison

- **Cấu hình tốt hơn:** Config B (hybrid + RRF) cho hiệu quả tổng thể vượt trội hơn Config A (dense-only) với điểm trung bình đạt 0.7792 so với 0.7694 (+0.0098).
- **Evidence:** 
  - Điểm độ trung thực (Faithfulness) tăng từ 0.9111 lên 0.9444 (+0.0333), chứng minh các đoạn ngữ cảnh dung hợp từ BM25 giúp LLM có căn cứ vững chắc hơn, giảm thiểu nguy cơ suy diễn ngoài tài liệu.
  - Điểm phù hợp câu trả lời (Answer Relevance) cải thiện từ 0.8722 lên 0.9111 (+0.0389). RRF phát huy thế mạnh rõ rệt ở các câu hỏi chứa tên riêng (tuyển thủ, sự kiện) và từ khóa chuyên môn (thông số súng LMG, RPD, M249, quy định DMA, mod policy), nơi dense search dễ bị phân tán bởi các vector ngữ nghĩa tương đồng trong cùng miền tài liệu game.
- **Trade-off về latency/cost:** 
  - Khâu tính toán BM25 và RRF score diễn ra trên RAM rất nhanh, chỉ tăng thêm ~15-20ms cho mỗi lượt truy xuất, hoàn toàn không đáng kể so với thời gian gọi API LLM (~1.5s - 2.5s).
  - Chi phí token LLM giữa 2 cấu hình là tương đương nhau do cùng cố định `top_k=5` chunks và cùng định dạng ngữ cảnh đưa vào generator prompt.

## Worst performers

|   # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage | Root cause |
| --: | -------- | ------ | -----------: | --------: | -----: | --------: | ------------- | ---------- |
|   1 | Hai tuyển thủ Himass và TanVuu bị xử lý kỷ luật như thế nào sau sự cố vận hành giải đấu PUBG Asia Stars? | Config A / Config B | 1.0000 | 0.0000 / 0.4000 | 0.0000 | 0.0000 | Retrieval | Thông tin về mức kỷ luật cụ thể (khóa vĩnh viễn tài khoản và cấm thi đấu vĩnh viễn) nằm rải rác ở các chunk khác nhau; top-5 chunks lấy về chỉ chứa hành vi vi phạm chung chung. Generator tuân thủ quy tắc Safe Refusal nên từ chối trả lời bịa đặt. |
|   2 | Tại Đảo Khởi hành trong sự kiện PUBG x Jujutsu Kaisen, người chơi tương tác với Hộp Jujutsu Kaisen như thế nào và hiện tượng gì xảy ra? | Config A / Config B | 1.0000 | 0.3000 / 0.3000 | 0.1000 / 0.2000 | 0.2000 | Retrieval | Bài viết sự kiện chứa nhiều đoạn đề cập từ khóa "Jujutsu Kaisen". Semantic search bị nhiễu và trả về các chunk nói về Nhà kho sự kiện và máy bán hàng tự động thay vì đoạn mô tả tương tác Hộp ở Đảo Khởi hành. |
|   3 | Người chơi có được phép bán các bản Mod của mình để kiếm tiền trong PUBG theo Chính sách về Mod không? | Config A / Config B | 0.8000 | 1.0000 / 1.0000 | 0.8000 | 0.2000 / 0.4000 | Retrieval / Generation | Tài liệu chính sách Mod gồm nhiều điều khoản pháp lý tương tự nhau khiến context precision thấp (nhiều chunk nhiễu). Model trả lời đúng trọng tâm cấm kinh doanh nhưng bị trừ nhẹ điểm faithfulness do chưa trích dẫn đầy đủ điều khoản tước quyền sở hữu. |

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
| -------: | ------ | ------------------------------ | --------------- | ------------- |
|        1 | Section-aware & Semantic Chunking | Case 1 và Case 2 bị trượt thông tin do chunking cố định 500 ký tự cắt rời chi tiết phán quyết và tương tác sự kiện khỏi bối cảnh chính. | Tăng Context Recall từ 0.85 lên > 0.92, giải quyết triệt để các câu hỏi cần liên kết thông tin đa đoạn. | Chạy lại `run_evaluation.py` và kiểm tra điểm Context Recall của Case 1 và 2. |
|        2 | Tinh chỉnh tham số RRF và Entity Boosting | Case 1 chứa tên riêng thực thể (Himass, TanVuu, PUBG Asia Stars) nhưng RRF k=60 chưa đủ đẩy chunk chứa hình phạt lên top đầu. | Tăng Context Precision (+0.10) và độ liên quan của các câu hỏi tra cứu thực thể cụ thể. | Đánh giá độ chính xác xếp hạng (MRR / Hit@3) trên tập các câu hỏi chứa tên thực thể. |
|        3 | Pre-retrieval Query Expansion / HyDE | Ở Case 2, câu hỏi diễn đạt tự nhiên ("hiện tượng gì xảy ra") không khớp từ khóa nguyên văn trong bài báo. | Cải thiện độ bao phủ ngữ nghĩa cho các truy vấn phức tạp của người dùng. | Kiểm thử qua tập câu hỏi out-of-vocabulary hoặc câu hỏi ngữ cảnh mở rộng. |

## Bonus experiments

| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |
| ---------- | -------- | -----------: | -----------------: | ---------- |
| Reorder context for LLM (`reorder_for_llm` mitigation lost-in-the-middle) | Giữ nguyên thứ tự rank chuẩn | Faithfulness: +0.022 | Latency: 0 ms / Cost: 0 token | Đưa các chunk có độ liên quan cao nhất về đầu và cuối context giúp LLM nắm bắt bằng chứng tốt hơn và trích dẫn chuẩn xác hơn. |
| Fallback PageIndex vectorless search khi cosine score < 0.3 | Dense search đơn thuần không fallback | Context Recall trên out-of-domain: +0.15 | Thêm 1 API call khi fallback kích hoạt | Giúp hệ thống có cơ chế phòng thủ vững chắc khi vector search không tìm thấy tài liệu phù hợp. |
