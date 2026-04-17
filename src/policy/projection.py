from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class WeeklyActionProjection:
    projected_contract_adjustment_mwh: float
    projected_exposure_band_mwh: float
    projection_gap_mwh: float
    projection_active: bool
    projection_reason_codes: list[str]


def project_weekly_actions(
    *,
    raw_contract_adjustment_mwh: float,
    raw_exposure_band_mwh: float,
    forecast_weekly_load_mwh: float,
    contract_adjustment_ratio_min: float,
    contract_adjustment_ratio_max: float,
    exposure_band_ratio_min: float,
    exposure_band_ratio_max: float,
    bound_reason_code: str,
) -> WeeklyActionProjection:
    contract_min = float(forecast_weekly_load_mwh) * float(contract_adjustment_ratio_min)
    contract_max = float(forecast_weekly_load_mwh) * float(contract_adjustment_ratio_max)
    exposure_min = float(forecast_weekly_load_mwh) * float(exposure_band_ratio_min)
    exposure_max = float(forecast_weekly_load_mwh) * float(exposure_band_ratio_max)
    projected_contract = min(max(float(raw_contract_adjustment_mwh), contract_min), contract_max)
    projected_exposure = min(max(float(raw_exposure_band_mwh), exposure_min), exposure_max)
    gap = abs(float(raw_contract_adjustment_mwh) - projected_contract) + abs(float(raw_exposure_band_mwh) - projected_exposure)
    reason_codes: list[str] = []
    if abs(float(raw_contract_adjustment_mwh) - projected_contract) > 1.0e-9:
        reason_codes.append("contract_adjustment_clip")
    if abs(float(raw_exposure_band_mwh) - projected_exposure) > 1.0e-9:
        reason_codes.append("exposure_band_clip")
    if bound_reason_code:
        reason_codes.append(str(bound_reason_code))
    return WeeklyActionProjection(
        projected_contract_adjustment_mwh=projected_contract,
        projected_exposure_band_mwh=projected_exposure,
        projection_gap_mwh=gap,
        projection_active=gap > 1.0e-9,
        projection_reason_codes=reason_codes,
    )


def project_hourly_hedge_tensor(
    raw_hourly_hedge_mwh: torch.Tensor,
    *,
    projected_exposure_band_mwh: torch.Tensor,
    hourly_share_cap: torch.Tensor,
    hourly_ramp_share: torch.Tensor,
    hour_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid_hours = hour_mask.sum(dim=-1).clamp_min(1.0)
    per_hour_limit = (projected_exposure_band_mwh.unsqueeze(-1) / valid_hours.unsqueeze(-1)) * hourly_share_cap
    projected = torch.clamp(raw_hourly_hedge_mwh, min=-per_hour_limit, max=per_hour_limit) * hour_mask
    ramp_limit = per_hour_limit * hourly_ramp_share
    for hour_index in range(1, projected.shape[-1]):
        current_mask = hour_mask[..., hour_index]
        upper = torch.minimum(per_hour_limit[..., hour_index], projected[..., hour_index - 1] + ramp_limit[..., hour_index])
        lower = torch.maximum(-per_hour_limit[..., hour_index], projected[..., hour_index - 1] - ramp_limit[..., hour_index])
        clipped = torch.maximum(torch.minimum(projected[..., hour_index], upper), lower)
        projected[..., hour_index] = torch.where(current_mask > 0.0, clipped, torch.zeros_like(clipped))
    projection_gap = torch.abs(raw_hourly_hedge_mwh - projected) * hour_mask
    return projected, projection_gap.sum(dim=-1)
