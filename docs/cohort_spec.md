# Cohort spec — liều IV furosemide tại T0 ở bệnh nhân CKD (MIMIC-IV 3.1, hosp + icu)

Trạng thái: mục 1–6 và cờ đã chốt (2026-10-29), trừ 4.1 chờ probe.
Số liệu lấy từ Probe 1 (`kg/ehr/cohort_funnel_probe.py`), chỉ là số tổng hợp.

## 1. Đơn vị mẫu, T0, nhãn, input

- **Mẫu**: 1 bệnh nhân = 1 mẫu = **lượt nhập viện cuối đủ tiêu chuẩn** (qua S9).
- **T0**: mũi bolus/push IV furosemide **đầu tiên tại ICU** của lượt đó
  (`inputevents` itemid 221794, mg, amount > 0, bỏ `statusdescription = Rewritten`).
- **Nhãn**: mg của đúng mũi T0, giá trị gốc, không cộng dồn.
  Ladder: 10/20/40/60/80/100/120/140/160/200 (phủ 99,43% liều đầu ICU). Ladder nằm trong system prompt.
- Lý do chọn liều đầu: là quyết định duy nhất chưa thấy đáp ứng; từ liều 2 là chỉnh liều theo đáp ứng
  (bài toán khác); guideline (DOSE, ESC) định nghĩa liều khởi đầu.
- **Input = toàn bộ lịch sử trước T0**:
  - Lượt nhập viện trước (n−1, n−2, …): **mọi** mũi furosemide (ICU bolus/truyền, IV khoa thường, uống)
    với thời điểm, liều, route; đáp ứng sau mỗi mũi (nước tiểu 6 h, creatinine, tăng liều);
    liều uống lúc ra viện; khoảng cách thời gian tới lượt hiện tại.
  - Lượt hiện tại trước T0: mũi IV ở khoa thường, liều uống ở nhà, xét nghiệm, sinh hiệu.
  - Lượt nhập viện **sau** T0 không bao giờ vào input.
  - Thiếu nguồn (vd. `emar` trước ~2011) ghi rõ "không có dữ liệu", không ngầm hiểu là "không tiêm".
  - Cách nén lịch sử dài: để C2 quyết định.
- Route của emar lấy từ `pharmacy.route` → `emar_detail.route` → tên chế phẩm
  (`emar_detail.route` không bao giờ ghi IV cho furosemide). Khí dung (IH/NEB) không tính là IV.

## 2. Các bước lọc

Không bước lọc nào được dùng thông tin sau T0, trừ ngoại lệ ghi ở cuối mục.

| Bước | Tiêu chí | Còn lại |
|---|---|---|
| S1 | Mũi bolus/push IV furosemide tại ICU | 95.799 mũi |
| S2 | Mũi ICU đầu tiên mỗi lượt nhập viện = T0 | 24.761 lượt |
| S3 | Liều T0 trên ladder; lệch thì bỏ cả lượt (không lấy liều kế) | 24.621 |
| S4 | Tuổi ≥ 18 tại T0 | 24.621 |
| S5 | Không đang truyền furosemide liên tục tại T0 | 24.123 |
| S6 | Bằng chứng CKD (a/b/c, mục 3) | 8.249 |
| S7 | Không lọc máu trong 24 h trước T0 | 8.165 |
| S8 | Phụ thuộc lọc máu (Z99.2/V45.11/N18.6/585.6) → cohort 5D riêng (645) | 7.520 |
| S9 | Creatinine **đã có kết quả** (`storetime`) trong [T0 − 48 h, T0) | 7.424 lượt / 6.176 BN |
| S10 | Lượt cuối đủ tiêu chuẩn mỗi BN | 6.176 mẫu |

- Chia theo bệnh nhân 70/15/15 ≈ 4.323 / 926 / 926.
- **Ngoại lệ**: S6(c) và S8 dùng mã ICD của lượt hiện tại (ghi lúc ra viện) — chỉ để chọn ca,
  không làm feature. Kiểm tra bằng phân tích độ nhạy trên nhóm `confirmed`.
- Không lọc theo tử vong, thời gian nằm viện, hay đáp ứng sau T0.
- Cần kiểm tra (Probe 2): tử vong trong viện của lượt mục tiêu so với các lượt khác
  (lấy lượt cuối có thể nghiêng về ca nặng).

## Cờ

Quy tắc: cờ nhóm A **không bao giờ vào input**. Cờ mới phải xếp được vào A1–A3, B hoặc C.

**A. Trục phân tích** (metric tính riêng từng nhóm)

| Trục | Nhóm | Gộp từ |
|---|---|---|
| A1 `ckd_evidence` | `confirmed` / `index_only` | `ckd_tier` (giữ 5 mức chi tiết trong file) |
| A2 `decision_context` | `routine` / `protocol` / `loading_for_infusion` / `comfort_care` | truyền máu, albumin, sau mổ tim (cửa sổ [T0−2 h, T0+1 h]); bắt đầu truyền ≤ 1 h sau T0; Code Status. Ưu tiên khi trùng: comfort > loading > protocol |
| A3 `prior_exposure` | `naive` / `prior_admission` / `ward_this_admission` / `unverifiable` | `emar_group` + lịch sử lượt trước |

Kèm dải liều của nhãn (≤20 / 40–60 / ≥80) cho metric macro — suy từ nhãn, không phải cờ.

**B. Cờ lâm sàng** (Bảng 1; metric riêng nếu nhóm đủ lớn)

| Cờ | Vào input |
|---|---|
| `anuric` (nước tiểu trước T0) | có |
| `transplant` | có, chỉ từ mã ICD lượt **trước**; mã lượt hiện tại chỉ dùng cho Bảng 1 |

**C. Mô tả** (Bảng 1): `via_ed`, `first_careunit`, `service_at_t0`, `admission_type` (được vào input);
`anchor_year_group` (chỉ mô tả).

## 3. Xác định CKD

Mọi bằng chứng (a), (b) đều nằm **trước** lượt nhập viện mục tiêu. eGFR theo CKD-EPI 2021
(không hệ số chủng tộc), tuổi tại thời điểm đo.

| Nguồn | Định nghĩa |
|---|---|
| (a) | Mã N18.x / 585.x ở lượt nhập viện trước. Mã I12/I13 (ICD-10), 403/404 (ICD-9) thêm vào nếu probe cho thấy đáng kể |
| (b) | 2 lần eGFR < 60 cách nhau ≥ 90 ngày, **chỉ từ creatinine ngoại trú** (`labevents.hadm_id` trống) và `omr` eGFR, **không có lần eGFR ≥ 60 xen giữa** |
| (c) | Chỉ có mã CKD ở lượt hiện tại (chỉ để chọn ca; nhóm `index_only` ở A1) |

- **Input**: eGFR nền (lần gần nhất trước nhập viện, ưu tiên ngoại trú), creatinine tại T0,
  chênh lệch giữa hai giá trị (AKI chồng CKD). Không đưa giai đoạn suy từ mã ICD.
- **Giai đoạn để phân tích**: từ eGFR nền (G3a/G3b/G4/G5); không có eGFR nền thì dùng eGFR tại T0
  và gắn cờ `stage_from_t0`.
- Probe cần đo: số ca đổi tầng khi siết (b); số ca chỉ có I12/I13/403/404.
- Giới hạn: CKD G1–G2 (định nghĩa bằng albumin niệu) chỉ bắt được qua mã ICD.
## 4. Quy tắc chống rò rỉ

- Feature chỉ từ dữ liệu trước T0. Bảng có `storetime` (`labevents`, `chartevents`, `outputevents`,
  `inputevents`, `emar`) dùng `storetime < T0`; bảng không có thì `charttime < T0`.
- Trường chỉ có ngày (`omr.chartdate`, `procedures_icd.chartdate`): chỉ lấy khi ngày < ngày của T0.
- Cấm làm feature: mã ICD/DRG lượt hiện tại, `dischtime`, `deathtime`, `discharge_location`,
  `hospital_expire_flag`, thời gian nằm viện, `transfers`/`services` sau T0, mọi tổng hợp tính trên
  cả lượt nhập viện. Mọi tổng hợp phải tính trong cửa sổ trước T0.
- Dữ liệu sau T0 lưu file riêng, chỉ cho đánh giá.
- **4.1 (chờ kiểm tra)**: y lệnh furosemide IV trước T0 (`pharmacy`/`prescriptions`, thời điểm hiển thị
  = `pharmacy.entertime`) có thể chính là y lệnh của mũi T0 → lộ nhãn. Probe đo tỉ lệ ca có y lệnh IV
  trong 1/6/24 h trước T0 và tỉ lệ liều y lệnh gần nhất = nhãn. Nếu cao: lượt hiện tại chỉ dùng bản ghi
  **đã tiêm**, không dùng y lệnh. `poe` chưa kiểm (không có liều).
- **Liều uống**: MIMIC-IV-ED không có nên không có liều tại nhà. Ước lượng theo thứ tự: đơn uống lúc
  ra viện lượt trước (kèm khoảng cách) → furosemide uống đã tiêm trong lượt hiện tại trước T0 →
  "không rõ". Gọi là "liều uống gần nhất được ghi nhận".
- **DUA PhysioNet**: mọi lần chạy có dữ liệu MIMIC chỉ dùng `local_vllm` trên GreenNode hoặc provider
  đã xác minh zero-retention. Không dùng Gemini API công khai (`provider: gemini` mặc định hiện tại).
## 5. Chia tập

- **Chính**: theo bệnh nhân, 70/15/15 (≈ 4.323 / 926 / 926), phân tầng theo dải liều × `ckd_evidence`,
  seed cố định.
- **Độ nhạy theo thời gian**: train/dev = BN `anchor_year_group` 2008–2016, test = 2017–2022
  (≈ 1.000 BN). Giới hạn: `anchor_year_group` là nhóm 3 năm theo BN, không phải năm nhập viện.
- Vai trò: train = kho truy hồi ca tương tự, few-shot, thống kê tri thức; dev = chỉnh prompt/cấu hình;
  test = chạy **một lần** sau khi chốt, không truy hồi hay lấy ví dụ từ test.
- File ID các tập chỉ lưu trên GreenNode, không commit (DUA).
## 6. Baseline và metric

**Baseline** (cùng tập test):

| | Baseline |
|---|---|
| B0 | Hằng số: luôn 20 mg (mode), luôn 40 mg (median) — mức sàn, luôn in trong bảng kết quả |
| B1 | Luật guideline: có liều uống gần nhất → 1–2,5 lần (DOSE/ESC); không có → bảng theo giai đoạn eGFR |
| B2 | ML dạng bảng (LightGBM, phân loại thứ tự trên ladder), cùng bộ feature |
| B4 | LLM zero-shot, không truy hồi, không tri thức (ablation C3) |

**Metric**:

| Metric | Ghi chú |
|---|---|
| MAE (mg) | In kèm skill = 1 − MAE / MAE_B0 (B0 = luôn 40 mg, MAE ≈ 25 mg trên Probe 1) |
| MAPE | Phát hiện quá liều; lưu ý MAPE chặn sai số đoán thấp ở 100% |
| ±20% | So sánh với tài liệu. Trên ladder: ≤ 80 mg tương đương đúng bậc |
| Đường REC | Tỉ lệ ca theo ngưỡng sai số |
| Similarity | Giải thích vs đoạn guideline đúng cho ca (ánh xạ theo luật B1). Metric phụ, không dùng để kết luận chất lượng giải thích (MedThink-Bench: Pearson −0,17 đến 0,45 với chuyên gia) |
| Nhất quán giải thích–liều | Hướng (giảm/giữ/tăng) và liều nêu trong giải thích so với liều xuất ra |

Bỏ khỏi phạm vi hiện tại: kNN baseline, đúng bậc / ±1 bậc / 2 lần, tầng outcome, κ chuyên gia.
