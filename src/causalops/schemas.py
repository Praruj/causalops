from __future__ import annotations

from pydantic import BaseModel, Field


class DriftMetric(BaseModel):
    """How much one column moved between the reference and the incident window."""

    feature: str
    psi: float = Field(ge=0)
    ks: float = Field(ge=0, le=1)
    wasserstein: float = Field(ge=0, description="on values scaled by the reference std")
    null_rate_ref: float = Field(ge=0, le=1)
    null_rate_cur: float = Field(ge=0, le=1)


class Hypothesis(BaseModel):
    """A suspect and the fixed test that will be run on it."""

    cause: str
    test: str
    reason: str


class CausalValidation(BaseModel):
    """Result of undoing one suspected change and re-scoring the model."""

    cause: str
    test: str
    loss_after_repair: float
    recovered_fraction: float = Field(description="share of the extra loss that went away")
    ci_low: float
    ci_high: float
    verified: bool
    label_shift_explained: float | None = Field(
        default=None, description="share of the label-rate move that this repair also undoes"
    )


class RCAReport(BaseModel):
    loss_reference: float
    loss_incident: float
    auc_reference: float
    auc_incident: float
    performance_dropped: bool
    drift: list[DriftMetric]
    hypotheses: list[Hypothesis]
    validations: list[CausalValidation]
    root_cause: str | None
    cause_type: str | None
    summary: str
