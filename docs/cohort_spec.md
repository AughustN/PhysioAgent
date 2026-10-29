# Cohort spec — liều IV furosemide tại T0 ở bệnh nhân CKD (MIMIC-IV 3.1, hosp + icu)

Trạng thái: mục 1–6, cờ và 4.6 đã chốt (2026-10-29).
Số liệu lấy từ Probe 1 (`kg/ehr/cohort_funnel_probe.py`) và Probe 2 (`kg/ehr/cohort_feature_probe.py`),
chỉ là số tổng hợp. Funnel dưới đây là theo Probe 1; luật (b) strict (mục 3) sẽ làm giảm S6–S10.

## 1. Đơn vị mẫu, T0, nhãn, input

- **Mẫu**: 1 bệnh nhân = 1 mẫu = **lượt nhập viện cuối đủ tiêu chuẩn** (qua S9).
- **T0**: mũi bolus/push IV furosemide **đầu tiên tại ICU** của lượt đó
  (`inputevents` itemid 221794, mg, amount > 0, bỏ `statusdescription = Rewritten`).
- **Nhãn**: mg của đúng mũi T0, giá trị gốc, không cộng dồn.
  Ladder: 10/20/40/60/80/100/120/140/160/200 (phủ 99,43% liều đầu ICU). Ladder nằm trong system prompt.
- Lý do chọn liều đầu: là quyết định duy nhất chưa thấy đáp ứng; từ liều 2 là chỉnh liều theo đáp ứng
  (bài toán khác); guideline (DOSE, ESC) định nghĩa liều khởi đầu.
- **Mốc cắt input = `t_dec`** (mục 4.6), không phải T0.
- **Input = toàn bộ lịch sử trước `t_dec`**:
  - Lượt nhập viện trước (n−1, n−2, …): **mọi** mũi furosemide (ICU bolus/truyền, IV khoa thường, uống)
    với thời điểm, liều, route; đáp ứng sau mỗi mũi (nước tiểu 6 h, creatinine, tăng liều);
    liều uống lúc ra viện; khoảng cách thời gian tới lượt hiện tại.
  - Lượt hiện tại trước `t_dec`: mũi IV ở khoa thường, liều uống gần nhất được ghi nhận, xét nghiệm,
    sinh hiệu, nước tiểu.
  - Lượt nhập viện **sau** T0 không bao giờ vào input.
  - Nguồn thưa (Probe 2): albumin 30%, lactate 70%, NT-proBNP 9,6% — prompt không được phụ thuộc vào chúng.
  - Thiếu nguồn (vd. `emar` trước ~2011) ghi rõ "không có dữ liệu", không ngầm hiểu là "không tiêm".
  - Cách nén lịch sử dài: để C2 quyết định (Probe 2: 9,7% BN có ≥ 10 lượt trước; số mũi trước p90 = 44).
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
| S9 | Creatinine **đã có kết quả** (`storetime`) trong [`t_dec` − 48 h, `t_dec`), nối theo `subject_id` | 7.424 lượt / 6.176 BN |
| S10 | Lượt cuối đủ tiêu chuẩn mỗi BN | 6.176 mẫu |

- Chia theo bệnh nhân 70/15/15 ≈ 4.323 / 926 / 926.
- Builder: `kg/ehr/cohort_build.py` áp luật (b) strict ở S6 và mốc `t_dec` ở S9 (mốc cắt input phải
  có creatinine); số mới thay bảng trên sau lần chạy đầu.
- **Ngoại lệ**: S6(c) và S8 dùng mã ICD của lượt hiện tại (ghi lúc ra viện) — chỉ để chọn ca,
  không làm feature. Kiểm tra bằng phân tích độ nhạy trên nhóm `confirmed`.
- Không lọc theo tử vong, thời gian nằm viện, hay đáp ứng sau T0.
- Tử vong trong viện (Probe 2): 20,3% ở lượt mục tiêu so với 16,9% trên toàn S9. Lượt không được chọn
  luôn là lượt BN còn quay lại nên không tử vong; lấy lượt cuối nghiêng về ca nặng. Ghi vào giới hạn,
  giữ phương án A.

## Cờ

Quy tắc: cờ nhóm A **không bao giờ vào input**. Cờ mới phải xếp được vào A1–A3, B hoặc C.

**A. Trục phân tích** (metric tính riêng từng nhóm)

| Trục | Nhóm | Gộp từ |
|---|---|---|
| A1 `ckd_evidence` | `confirmed` / `index_only` | `ckd_tier` (giữ 5 mức chi tiết trong file) |
| A2 `decision_context` | `routine` / `protocol` / `loading_for_infusion` | truyền máu, albumin, sau mổ tim (cửa sổ [T0−2 h, T0+1 h]); bắt đầu truyền ≤ 1 h sau T0 (Probe 2: 2,7%). Ưu tiên khi trùng: loading > protocol. Bỏ `comfort_care` (Probe 2: 1 ca) |
| A3 `prior_exposure` | `naive` / `prior_admission` / `ward_this_admission` / `unverifiable` | `emar_group` + lịch sử lượt trước. Probe 2: naive 30,3%, prior_admission 27,6%, ward 10,7%, unverifiable 31,4% (giả thuyết: do thiếu emar 2008–2010, builder in crosstab với `anchor_year_group`) |

Kèm dải liều của nhãn (≤20 / 40–60 / ≥80) cho metric macro — suy từ nhãn, không phải cờ.

**B. Cờ lâm sàng** (Bảng 1; metric riêng nếu nhóm đủ lớn)

| Cờ | Vào input |
|---|---|
| `transplant` | có, chỉ từ mã ICD lượt **trước**; mã lượt hiện tại chỉ dùng cho Bảng 1 |

Bỏ `anuric` (Probe 2: 39/2.336 ca đủ 24 h ICU có nước tiểu 24 h < 100 mL). Nước tiểu 6 h / 24 h vẫn vào
input dạng số.

**C. Mô tả** (Bảng 1): `via_ed`, `first_careunit`, `service_at_t0`, `admission_type`, `code_status`
(Code Status itemid 223758 gần nhất trước `t_dec`; Probe 2: 59% không có bản ghi → "không có dữ liệu")
(được vào input); `anchor_year_group` (chỉ mô tả).

## 3. Xác định CKD

Mọi bằng chứng (a), (b) đều nằm **trước** lượt nhập viện mục tiêu. eGFR theo CKD-EPI 2021
(không hệ số chủng tộc), tuổi tại thời điểm đo.

| Nguồn | Định nghĩa |
|---|---|
| (a) | Mã N18.x / 585.x ở lượt nhập viện trước. Không thêm I12/I13/403/404 (Probe 2: 1 BN có mà không có N18/585) |
| (b) | 2 lần eGFR < 60 cách nhau ≥ 90 ngày, **chỉ từ creatinine ngoại trú** (`labevents.hadm_id` trống) và `omr` eGFR, **không có lần eGFR ≥ 60 xen giữa** |
| (c) | Chỉ có mã CKD ở lượt hiện tại (chỉ để chọn ca; nhóm `index_only` ở A1) |

- **Input**: eGFR nền (lần gần nhất trước nhập viện, ưu tiên ngoại trú), creatinine tại T0,
  chênh lệch giữa hai giá trị (AKI chồng CKD). Không đưa giai đoạn suy từ mã ICD.
- **Giai đoạn để phân tích**: từ eGFR nền (G3a/G3b/G4/G5); không có eGFR nền thì dùng eGFR tại T0
  và gắn cờ `stage_from_t0`.
- Siết (b) theo luật trên (Probe 2, trên 6.176 BN của funnel cũ): 669 BN (10,8%) mất hết bằng chứng CKD
  → ra khỏi cohort; 335 `confirmed_a_and_b` → `confirmed_a`; 116 `confirmed_b` → `index_only`.
  `index_only` lên 34,7%. Cohort ước còn ≈ 5,5 nghìn BN, số chính xác do builder tính lại S6–S10.
- eGFR nền (lần ngoại trú gần nhất ≤ 365 ngày trước nhập viện): Probe 2 có ở 82,3% BN.
- Giới hạn: CKD G1–G2 (định nghĩa bằng albumin niệu) chỉ bắt được qua mã ICD. Probe 2: 17,5% BN có
  eGFR nền ≥ 60.

## 4. Quy tắc chống rò rỉ

- Feature chỉ từ dữ liệu trước `t_dec` (4.6). Bảng có `storetime` (`labevents`, `chartevents`,
  `outputevents`, `inputevents`, `emar`) dùng `storetime < t_dec`; bảng không có thì `charttime < t_dec`.
- Trường chỉ có ngày (`omr.chartdate`, `procedures_icd.chartdate`): chỉ lấy khi ngày < ngày của `t_dec`.
- Xét nghiệm nối theo `subject_id` + cửa sổ thời gian, **không** theo `hadm_id`: mẫu lấy ở ED trước khi
  nhập viện thường để trống `hadm_id` (Probe 2 nối theo `hadm_id` chỉ thấy creatinine 48 h ở 90% BN,
  dù S9 bắt buộc có).
- Cấm làm feature: mã ICD/DRG lượt hiện tại, `dischtime`, `deathtime`, `discharge_location`,
  `hospital_expire_flag`, thời gian nằm viện, `transfers`/`services` sau T0, mọi tổng hợp tính trên
  cả lượt nhập viện. Mọi tổng hợp phải tính trong cửa sổ trước T0.
- Dữ liệu sau T0 lưu file riêng, chỉ cho đánh giá.
- **4.1 Cấm y lệnh furosemide của lượt hiện tại** (`prescriptions`, `pharmacy`, `poe`). Lượt hiện tại
  chỉ dùng bản ghi **đã tiêm** (`emar` Administered, `inputevents`); y lệnh lượt trước được dùng.
  Bằng chứng (Probe 1, `pharmacy.entertime` < T0, liều từ `prescriptions`):

  | Cửa sổ trước T0 | Có y lệnh IV | Liều y lệnh gần nhất = nhãn |
  |---|---|---|
  | 1 h | 60,6% cohort | 94,6% |
  | 6 h | 85,7% | 94,4% |
  | 24 h | 88,8% | 93,9% |

- **4.6 Mốc quyết định `t_dec`** = `pharmacy.entertime` của y lệnh IV furosemide gần nhất trong 24 h
  trước T0 có liều = nhãn; không có thì `t_dec` = T0. Bác sĩ quyết định lúc ra y lệnh; cắt tại T0 sẽ
  đưa vào input dữ liệu ghi sau quyết định. Probe 2: 85,6% BN có y lệnh khớp; y lệnh → tiêm median
  0,59 h, p90 2,34 h (68,6% ≤ 1 h, 93,4% ≤ 3 h); tần suất ONCE 82%, BID 16%. `t_dec` là một cột,
  không phải cờ. Phân tích độ nhạy: cắt tại T0.
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
| B0 | Hằng số = median liều của tập train (hiện 20 mg); in kèm luôn 20 mg và luôn 40 mg — mức sàn, luôn in trong bảng kết quả. Probe 2 (6.176 BN): luôn 20 mg MAE 25,06 / MAPE 39,7% / ±20% 42,3%; luôn 40 mg MAE 25,29 / MAPE 80,4% / ±20% 25,2% |
| B1 | Luật guideline: có liều uống gần nhất → 1–2,5 lần (DOSE/ESC); không có → bảng theo giai đoạn eGFR |
| B2 | ML dạng bảng (LightGBM, phân loại thứ tự trên ladder), cùng bộ feature |
| B4 | LLM zero-shot, không truy hồi, không tri thức (ablation C3) |

**Metric**:

| Metric | Ghi chú |
|---|---|
| MAE (mg) | In kèm skill = 1 − MAE / MAE_B0 (B0 = median train) |
| MAPE | Phát hiện quá liều; lưu ý MAPE chặn sai số đoán thấp ở 100% |
| ±20% | So sánh với tài liệu. Trên ladder: ≤ 80 mg tương đương đúng bậc |
| Đường REC | Tỉ lệ ca theo ngưỡng sai số |
| Similarity | Giải thích vs đoạn guideline đúng cho ca (ánh xạ theo luật B1). Metric phụ, không dùng để kết luận chất lượng giải thích (MedThink-Bench: Pearson −0,17 đến 0,45 với chuyên gia) |
| Nhất quán giải thích–liều | Hướng (giảm/giữ/tăng) và liều nêu trong giải thích so với liều xuất ra |

Bỏ khỏi phạm vi hiện tại: kNN baseline, đúng bậc / ±1 bậc / 2 lần, tầng outcome, κ chuyên gia.
