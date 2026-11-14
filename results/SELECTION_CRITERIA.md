# Tiêu chí chọn cấu hình — chốt TRƯỚC khi chạy agent và TRƯỚC khi chạy test

Ngày: 2026-11-14.

## Retrieval (đã chạy trên dev, không LLM)
Tiêu chí: Evidence Hit@3 cao nhất; hòa thì Hit@1 cao nhất; còn hòa thì ít tài nguyên hơn.
Kết quả dev: xem `results/retrieval/dev.json`.

## Agent (chạy trên dev trước khi chốt)
Cấu hình được chọn theo thứ tự ưu tiên trên **dev**:
1. Context Evidence Rate (số câu có evidence trong ngữ cảnh cuối / 15 câu có đáp án) cao nhất.
2. Hòa thì tỷ lệ câu trả lời đúng (grounded answer mean, nhãn thủ công) cao nhất.
3. Còn hòa thì ít token đầu vào hơn.

Nhãn chất lượng câu trả lời (0–2) và citation là nhãn thủ công; nếu chưa có nhãn thủ công cho dev, chọn theo (1) và báo rõ.

## Quy tắc tập test
- Chạy test đúng một lần, với cấu hình đã chốt ở trên, và các cấu hình baseline để so sánh.
- Không đổi tiêu chí hay cấu hình sau khi xem kết quả test.
- Test Day 6 (5 câu) đã được Day 6 dùng để báo cáo, không dùng để chỉnh Day 7; ghi chú này nằm trong báo cáo.
