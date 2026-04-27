from __future__ import annotations

from src.policy.projection import project_weekly_actions


def test_project_weekly_actions_clips_raw_values_and_records_gap() -> None:
    projection = project_weekly_actions(
        raw_contract_adjustment_mwh=60.0,
        raw_exposure_band_mwh=40.0,
        forecast_weekly_load_mwh=100.0,
        contract_adjustment_ratio_min=-0.20,
        contract_adjustment_ratio_max=0.20,
        exposure_band_ratio_min=0.10,
        exposure_band_ratio_max=0.25,
        bound_reason_code="lt_price_linked|ancillary_tight",
    )

    assert projection.projected_contract_adjustment_mwh == 20.0
    assert projection.projected_exposure_band_mwh == 25.0
    assert projection.projection_gap_mwh == 55.0
    assert projection.projection_active is True
    assert projection.projection_reason_codes == ["contract_adjustment_clip", "exposure_band_clip", "lt_price_linked|ancillary_tight"]

