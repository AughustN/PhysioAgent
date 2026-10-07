from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class PatientProfile(BaseModel):
    age: int = Field(ge=0, le=120)
    sex: str
    weight_kg: float = Field(gt=0)
    height_cm: float = Field(gt=0)
    ckd_stage: Optional[str] = None
    egfr: Optional[float] = Field(default=None, ge=0)
    renal_stenosis_severity: Optional[float] = Field(default=None, ge=0, le=1)
    systolic_bp: float = Field(gt=0)
    diastolic_bp: float = Field(gt=0)
    heart_rate: Optional[float] = Field(default=None, gt=0)
    potassium: Optional[float] = None
    sodium: Optional[float] = None
    creatinine: Optional[float] = None
    bun: Optional[float] = None
    fluid_overload: bool = False
    edema: bool = False
    current_furosemide_dose: Optional[float] = None
    route: str = "IV"
    comorbidities: list[str] = Field(default_factory=list)
    clinical_question: str

    urine_prior24h_ml: Optional[float] = Field(
        default=None,
        ge=0,
        description="Urine output over t-24h..t0. Strictly pre-dose (report.md S2.7 SAFE).",
    )
    care_unit: Optional[str] = Field(
        default=None,
        description="ICU care unit at the dosing decision (MICU, CCU, ...). Categorical.",
    )
    admission_type: Optional[str] = Field(
        default=None,
        description="MIMIC admission_type (EW EMER., ELECTIVE, ...). Categorical.",
    )

    predose_sbp: Optional[float] = Field(default=None, gt=0)
    predose_dbp: Optional[float] = Field(default=None, gt=0)
    predose_hr: Optional[float] = Field(default=None, gt=0)

    has_chf: Optional[bool] = None
    has_diabetes: Optional[bool] = None
    has_htn: Optional[bool] = None
    aki_creat_safe: Optional[bool] = None

    loop_naive: Optional[bool] = None
    instance_id: Optional[str] = Field(
        default=None,
        description=(
            "Stable, unique case id (e.g. MIMIC '<subject_id>_<hadm_id>'). When set it "
            "becomes patient_id, so every run artifact is unique and traceable back to "
            "the real patient/admission. Left unset (web app, ad-hoc patients) it "
            "falls back to the descriptive '<ckd_stage>_<age>_<sex>' label, which is NOT unique."
        ),
    )

    @property
    def patient_id(self) -> str:
        if self.instance_id:
            return str(self.instance_id)
        stage = (self.ckd_stage or "ckd").replace(" ", "_")
        return f"{stage}_{self.age}_{self.sex}".lower()
