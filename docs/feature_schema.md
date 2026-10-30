# Feature schema — input cho mô hình liều IV furosemide tại T0

Trạng thái: **đã duyệt** (2026-10-29). Đi kèm `docs/cohort_spec.md`. Builder: `kg/ehr/cohort_features.py`;
loader và danh sách cột: `kg/ehr/dataset.py` (`FEATURE_COLUMNS`, `FORBIDDEN_COLUMNS`).
Mục đích: teammate code mô hình ML (B2) theo schema này bằng dữ liệu giả, trước khi có file thật.

## 0. Quy ước chung

- **Khóa**: `subject_id`, `hadm_id` (mỗi BN đúng 1 dòng trong `features.csv.gz`, khớp `cohort.csv.gz`).
- **Nhãn**: `cohort.csv.gz` → `dose_mg` (một trong 10/20/40/60/80/100/120/140/160/200).
- **Mốc cắt**: mọi giá trị đều có trước `t_dec` (`storetime < t_dec`; trường chỉ có ngày thì ngày < ngày
  của `t_dec`). Cửa sổ "6 h", "48 h" tính lùi từ `t_dec`.
- **Thiếu**: số để trống (NaN); phân loại ghi `unknown`. Thiếu ≠ 0: "không ghi nước tiểu" khác "0 mL".
  Nguồn có thể vắng cả khối (emar trước ~2011) có cờ `src_*` riêng để mô hình phân biệt
  "không tiêm" với "không có dữ liệu".
- **Kiểu**: bool ghi `0/1`; thời gian ghi dạng số giờ/ngày trước `t_dec`, không ghi timestamp tuyệt đối.
- **Đơn vị**: mg, mL, kg, mmHg, °C, mg/dL (creatinine, BUN, glucose), mEq/L (Na, K, Cl, HCO3), g/dL (Hb,
  albumin), mmol/L (lactate), pg/mL (NT-proBNP), eGFR mL/min/1,73 m² (CKD-EPI 2021).
- **File** (trên server, `kg/data/ehr/cohort/`, gitignore, DUA): `features.csv.gz` (bảng phẳng, ML dùng),
  `history.csv.gz` (bảng dài các mũi furosemide, LLM dùng; ML dùng bản tổng hợp ở nhóm H).

## 1. `features.csv.gz` — mỗi BN một dòng

### D. Nhân khẩu, bối cảnh

| Cột | Kiểu | Định nghĩa |
|---|---|---|
| `age` | số | `anchor_age` + năm(`t0`) − `anchor_year` |
| `sex_female` | bool | |
| `weight_kg` | số | Cân nặng gần nhất lượt này (chartevents 226512, 224639, 226531 lb → kg) |
| `weight_source` | phân loại | `icu_chart` / `omr` (omr gần nhất ≤ 365 ngày trước nhập viện) / `unknown` |
| `via_ed` | bool | |
| `admission_type` | phân loại | |
| `first_careunit` | phân loại | |
| `service_at_t_dec` | phân loại | |
| `code_status` | phân loại | `full` / `dnr` / `dni` / `dnr_dni` / `cmo` / `unknown` (itemid 223758 gần nhất) |
| `transplant_prior` | bool | Z94.0 / V42.0 ở lượt trước |
| `hosp_hours_before` | số | `t_dec` − `admittime` |
| `icu_hours_before` | số | `t_dec` − `intime` |
| `fluid_in_24h_ml` | số | Tổng `amount` (mL) của inputevents bắt đầu trong 24 h trước `t_dec` (tính theo `starttime`) |

### V. Sinh hiệu (6 h)

| Cột | Định nghĩa |
|---|---|
| `hr`, `sbp`, `dbp`, `map`, `spo2`, `resp_rate`, `temp_c` | Giá trị gần nhất trong 6 h |
| `sbp_min_6h`, `map_min_6h` | Thấp nhất trong 6 h (huyết áp thấp → thận trọng liều) |

Itemid: HR 220045; SBP 220179/220050; DBP 220180/220051; MAP 220181/220052; SpO2 220277; RR 220210;
nhiệt độ 223762 (°C), 223761 (°F → °C).

### L. Xét nghiệm (48 h, nối theo `subject_id`)

| Cột | Itemid | Định nghĩa |
|---|---|---|
| `creatinine` | 50912 | Gần nhất |
| `creatinine_hours_ago` | | Giờ từ lúc lấy mẫu tới `t_dec` |
| `creatinine_change_48h` | | Gần nhất − sớm nhất trong 48 h (AKI) |
| `bun`, `sodium`, `potassium`, `chloride`, `bicarbonate`, `magnesium`, `glucose`, `hemoglobin` | 51006, 50983, 50971, 50902, 50882, 50960, 50931, 51222 | Gần nhất |
| `albumin`, `lactate`, `ntprobnp` | 50862, 50813, 50963 | Gần nhất; thưa (30%, 70%, 10%) |

### K. Thận

| Cột | Định nghĩa |
|---|---|
| `egfr_t_dec` | eGFR từ `creatinine` |
| `egfr_baseline` | eGFR ngoại trú gần nhất ≤ 365 ngày trước nhập viện (creatinine không `hadm_id` + omr) |
| `egfr_baseline_days` | Số ngày trước nhập viện của lần đo đó |
| `egfr_delta` | `egfr_t_dec` − `egfr_baseline` |
| `ckd_stage` | G1…G5 từ `egfr_baseline`, không có thì từ `egfr_t_dec` |
| `stage_from_t_dec` | bool: `ckd_stage` lấy từ `egfr_t_dec` |

### U. Nước tiểu (outputevents, trừ irrigant 227488)

| Cột | Định nghĩa |
|---|---|
| `uo_6h_ml`, `uo_24h_ml` | Tổng trong cửa sổ; NaN nếu không có bản ghi nào |
| `uo_window_6h_h` | min(6, `icu_hours_before`): số giờ thực có để đo |
| `uo_rate_6h_ml_kg_h` | `uo_6h_ml` / `weight_kg` / `uo_window_6h_h` |

### F. Furosemide lượt hiện tại trước `t_dec` (chỉ bản ghi **đã tiêm**, rule 4.1)

| Cột | Định nghĩa |
|---|---|
| `src_emar_this_adm` | bool: lượt này có dữ liệu emar |
| `ward_iv_n` | Số mũi IV ở khoa thường (emar Administered, route IV) |
| `ward_iv_last_mg`, `ward_iv_max_mg` | Liều mũi gần nhất / lớn nhất |
| `ward_iv_total_24h_mg` | Tổng liều IV 24 h trước `t_dec` |
| `ward_iv_last_hours_ago` | |
| `oral_this_adm_last_mg` | Liều uống gần nhất đã cho lượt này |
| `icu_infusion_before` | bool: đã từng truyền liên tục và đã ngưng trước T0 (S5 loại ca đang truyền) |

Liều emar lấy từ `emar_detail.dose_given` (+ `dose_given_unit` = mg). Emar lượt hiện tại chỉ tính bản ghi
**trước `intime`**: T0 là mũi ICU đầu tiên trong inputevents, nên mũi IV emar giữa `intime` và T0 là ghi
trùng mũi T0 (lộ nhãn) hoặc mâu thuẫn định nghĩa T0. Kiểm tra trên cohort thật: 11/5.473 ca có mũi emar
< 1 h trước T0 cùng liều nhãn.

### H. Lịch sử các lượt trước (tổng hợp từ `history.csv.gz`)

| Cột | Định nghĩa |
|---|---|
| `n_prior_admissions` | |
| `days_since_last_discharge` | |
| `src_emar_prior` | bool: có ít nhất một lượt trước có dữ liệu emar |
| `prior_furo_admissions` | Số lượt trước có tiêm furosemide |
| `prior_iv_n` | Tổng số mũi IV ở các lượt trước (ICU + khoa thường) |
| `prior_iv_last_mg`, `prior_iv_median_mg`, `prior_iv_max_mg` | |
| `prior_icu_first_bolus_mg` | Liều T0-tương đương (mũi ICU đầu) của lượt trước gần nhất có ICU bolus |
| `prior_icu_first_bolus_days` | Cách bao nhiêu ngày |
| `prior_icu_first_bolus_uo_6h_ml` | Đáp ứng: nước tiểu 6 h sau mũi đó |
| `prior_icu_first_bolus_escalated` | bool: có mũi IV liều cao hơn trong 24 h sau |
| `prior_infusion` | bool: từng truyền liên tục ở lượt trước |
| `prior_last_furo_days` | Ngày từ lần tiêm furosemide gần nhất (mọi lượt trước) |
| `oral_recorded_mg` | Liều uống gần nhất được ghi nhận (spec 4.4) |
| `oral_recorded_daily_mg` | Tổng liều uống một ngày của nguồn đó: đơn ra viện = liều × số lần/ngày theo `frequency` (QID/Q6H 4, TID/Q8H 3, BID/Q12H 2, DAILY/QD/Q24H/QAM/QPM/QHS 1, Q48H/QOD 0,5; một lần hoặc PRN → trống); uống lượt này = tổng mg trong 24 h kết thúc ở liều uống cuối. Thêm 2026-10-30 cho luật ESC 2021 "1–2 lần liều uống mỗi ngày" |
| `oral_recorded_source` | `prior_discharge_rx` / `this_admission` / `unknown` (ưu tiên theo thứ tự này, spec 4.4) |
| `oral_recorded_days` | |
| `prior_hf` | bool: I50 / 428 ở lượt trước |
| `prior_cirrhosis` | bool: K70.3, K74.6 / 571.2, 571.5 ở lượt trước |
| `prior_diabetes` | bool: E08–E13 / 250 ở lượt trước |

Mã ICD chỉ lấy từ **lượt trước** (mã lượt hiện tại bị cấm, spec 4).

## 2. `history.csv.gz` — mỗi mũi furosemide một dòng

| Cột | Định nghĩa |
|---|---|
| `hadm_id` | Lượt mục tiêu |
| `event_hadm_id` | Lượt chứa mũi đó |
| `scope` | `prior_admission` / `this_admission` |
| `source` | `icu_inputevents` / `emar` / `discharge_rx` |
| `kind` | `bolus` / `infusion` / `oral` / `discharge_rx` / `other` (emar không rõ route hoặc không phải IV/uống) |
| `hours_before_t_dec` | |
| `dose_mg` | Bolus, oral: liều; infusion: tổng liều |
| `rate_mg_h`, `duration_h` | Infusion |
| `frequency` | `discharge_rx` |
| `uo_6h_after_ml` | Chỉ ICU (outputevents); NaN ở khoa thường |
| `creatinine_before`, `creatinine_after_48h` | Gần nhất trước mũi, gần nhất trong 48 h sau (chỉ lượt trước) |
| `escalated_24h` | bool: mũi IV liều cao hơn trong 24 h sau |

Đáp ứng chỉ tính khi cả cửa sổ (6 h / 24 h / 48 h) đóng trước `t_dec`, ở mọi lượt; creatinine trước/sau
chỉ cho lượt trước. Emar lượt hiện tại chỉ lấy trước `intime` (như nhóm F).

## 3. Cột cấm làm input

Loader (`load_dataset`) báo lỗi nếu bất kỳ cột nào dưới đây lọt vào X:

- Nhãn và suy từ nhãn: `dose_mg`, `dose_band`.
- Cờ nhóm A: `ckd_tier`, `ckd_evidence`, `emar_group`, `prior_exposure`, `decision_context`,
  `protocol_transfusion`, `protocol_albumin`, `protocol_post_cardiac_surgery`, `loading_for_infusion`.
- Chỉ mô tả: `transplant_any`, `anchor_year_group`, `anchor_year`, `anchor_age` (dùng `age`).
- Thời điểm, split: `t0`, `t_dec`, `admittime`, `intime`, `split`, `split_temporal`.
- Mọi cột của `outcomes.csv.gz`.

Cột A vẫn đọc được để **chia nhóm khi tính metric**, không bao giờ vào X.
