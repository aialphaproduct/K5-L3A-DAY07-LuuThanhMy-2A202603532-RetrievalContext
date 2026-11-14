# Quyết định chọn cấu hình — ghi TRƯỚC khi chạy test (2026-11-14)

Áp dụng đúng tiêu chí trong `SELECTION_CRITERIA.md`, chỉ dựa trên **dev** (`results/agent/dev/`, `results/agent/dev_summary.json`).

| Cấu hình | Retriever | Budget | Filter | Context Evidence Rate | Grounded mean (/2, AI-audit) | Input tokens |
|---|---|---:|---|---:|---:|---:|
| base | dense | 800 | không | 4/15 | chưa chấm | 70 096 |
| imp1 | hybrid_rrf | 800 | không | 4/15 | chưa chấm | 75 117 |
| imp2 | hybrid_rrf | 2400 | không | 12/15 | 1.60 | 122 817 |
| imp3 | hybrid_rrf | 2400 | document_id suy từ câu hỏi | 12/15 | 1.60 | 102 479 |

1. Tiêu chí 1 (Context Evidence Rate): imp2 = imp3 = 12/15 > base = imp1 = 4/15 → loại base, imp1.
2. Tiêu chí 2 (grounded mean): imp2 = imp3 = 1.60 (nhãn do AI đọc trace và chấm, anh phải kiểm lại) → hòa.
3. Tiêu chí 3 (ít token đầu vào hơn): imp3 (102 479) < imp2 (122 817).

**Cấu hình chốt: imp3.** Baseline đối chiếu trên test: base.
Không đổi cấu hình hay tiêu chí sau khi xem test.
