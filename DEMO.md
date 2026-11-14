# Demo Day 7 — kịch bản 4 phút

**Mở demo:** `.venv/Scripts/python demo/server.py` → http://127.0.0.1:8770
(Mục **3 ca chạy thật** chạy **agent thật** (claude-haiku-5-5): từng lượt tìm hiện ngay khi xảy ra. Model được nạp sẵn lúc server khởi động; badge "agent thật đã sẵn sàng" báo xong. Cần khóa API và mạng; nếu thiếu, có nút phát lại trace đã lưu để dự phòng. Kết quả có thể khác nhẹ giữa các lần chạy.)

**Menu trong demo (mỗi mục ứng với một hạng mục rubric):** Tổng quan · 1 Dữ liệu (13đ) · 2 Retrieval (17đ) · 3 Agent & context (17đ) · **3 ca chạy thật** · **Hỏi đáp test** (hỏi agent thật, so sánh với baseline, chấm tay, tải nhật ký) · 4 Đánh giá (17đ) · 5 Thử nghiệm (13đ) · 6 Tái lập & kết luận (8đ) · Đối chiếu rubric (85đ, có ô tự chấm để so với điểm cuối). Số liệu đọc trực tiếp từ `results/`.

**Trang 1 để cầm khi nói:** `demo/demo-1-trang.png` (hoặc `.pdf`) — sơ đồ tổng + 3 ca + 3 con số. Sửa nội dung ở `demo/demo-1-trang.html`.

## Mạch nói
| Phút | Làm gì | Nói gì (1 câu) |
|---|---|---|
| 0:00 | Chỉ sơ đồ `assets/day7-end-to-end.svg` | "Kho luật nạp ở Day 6, hôm nay thêm agent đứng giữa người hỏi và kho: tìm → ghép ngữ cảnh → trả lời có nguồn." |
| 0:45 | Bấm **Có bằng chứng** (`q14`) | "Agent tìm 2 lần, ra Điều 20, trả lời *phải xác minh tuổi trẻ em* và bấm vào `[NĐ 13 · a020:000]` là thấy đúng đoạn luật." |
| 1:45 | Bấm **Thiếu bằng chứng** (`u03`) | "Hỏi đăng ký kết hôn — ngoài hai văn bản. Agent tìm đủ lượt, thấy toàn mẫu biểu không liên quan nên **từ chối**, không dẫn nguồn." |
| 2:30 | Bấm **Có lỗi** (`q16`) | "Câu hỏi mơ hồ. Lượt hai tìm đúng đoạn Nghị định (hạng 2), nhưng **ngữ cảnh đã đầy** vì 3 đoạn Luật, nên đoạn đúng bị bỏ → trả lời sai. Sửa: xếp theo điểm thay vì thứ tự đến." |
| 3:30 | Chỉ bảng kết quả trong báo cáo | "Tăng budget 800 → 2400 đưa evidence trong ngữ cảnh từ 4/15 lên 12/15; test: 0,00 → 1,60 trên thang 2." |

## Câu hỏi giám khảo hay hỏi
- **Vì sao hybrid?** Dense tìm theo nghĩa, BM25 bắt đúng con số/cụm cố định; trộn theo hạng (RRF) khỏi chuẩn hóa điểm. Dev: Hit@3 13/15 so với dense 12/15.
- **Sao không tin hết số?** Nhãn là bản nháp AI, test nhỏ (5 + 2), mỗi cấu hình chạy 1 lần, filter chưa kích hoạt. Đã ghi ở phần Giới hạn.
- **Làm gì tiếp?** Sửa cách ghép ngữ cảnh (xếp theo điểm), thử rerank và top-k lớn hơn cho `q07`, `q13`.

## Nếu demo trục trặc
- Mất mạng/khóa lỗi → dùng 3 ca mẫu (không gọi API). Giao diện hiện lỗi rõ và có nút thử lại.
- Cổng 8770 bận → đổi `PORT` đầu `demo/server.py`.
