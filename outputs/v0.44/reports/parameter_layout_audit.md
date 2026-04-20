> version: v0.44
> experiment_id: v0.44-27074b97
> config_hash: 27074b97aa4356c7e77a229df8467731880d6d750a2f01c2121fbaa50ffffed7
> run_timestamp: 2026-04-19T13:17:53+08:00
> device: cuda
> data_range: 2025-11-01 00:00:00 -> 2026-03-20 23:45:00

# 参数布局审计

## 上层参数块

- `weekly_feature_weights`
  起止索引: [0, 152)
  维度: 152
  字段: prev_da_price_mean, prev_da_price_std, prev_da_price_max, prev_da_price_min, prev_da_price_q25, prev_da_price_q50, prev_da_price_q75, prev_id_price_mean, prev_id_price_std, prev_id_price_max, prev_id_price_min, prev_id_price_q25, prev_id_price_q50, prev_id_price_q75, prev_spread_mean, prev_spread_std, prev_spread_max, prev_spread_min, prev_spread_q25, prev_spread_q50, prev_spread_q75, prev_price_spread_abs_mean, prev_price_spread_abs_std, prev_price_spread_abs_max, prev_price_spread_abs_min, prev_price_spread_abs_q25, prev_price_spread_abs_q50, prev_price_spread_abs_q75, prev_net_load_da_mean, prev_net_load_da_std, prev_net_load_da_max, prev_net_load_da_min, prev_net_load_da_q25, prev_net_load_da_q50, prev_net_load_da_q75, prev_net_load_id_mean, prev_net_load_id_std, prev_net_load_id_max, prev_net_load_id_min, prev_net_load_id_q25, prev_net_load_id_q50, prev_net_load_id_q75, prev_load_dev_mean, prev_load_dev_std, prev_load_dev_max, prev_load_dev_min, prev_load_dev_q25, prev_load_dev_q50, prev_load_dev_q75, prev_wind_dev_mean, prev_wind_dev_std, prev_wind_dev_max, prev_wind_dev_min, prev_wind_dev_q25, prev_wind_dev_q50, prev_wind_dev_q75, prev_solar_dev_mean, prev_solar_dev_std, prev_solar_dev_max, prev_solar_dev_min, prev_solar_dev_q25, prev_solar_dev_q50, prev_solar_dev_q75, prev_renewable_dev_abs_mean, prev_renewable_dev_abs_std, prev_renewable_dev_abs_max, prev_renewable_dev_abs_min, prev_renewable_dev_abs_q25, prev_renewable_dev_abs_q50, prev_renewable_dev_abs_q75, prev_renewable_ratio_da_mean, prev_renewable_ratio_da_std, prev_renewable_ratio_da_max, prev_renewable_ratio_da_min, prev_renewable_ratio_da_q25, prev_renewable_ratio_da_q50, prev_renewable_ratio_da_q75, prev_renewable_ratio_id_mean, prev_renewable_ratio_id_std, prev_renewable_ratio_id_max, prev_renewable_ratio_id_min, prev_renewable_ratio_id_q25, prev_renewable_ratio_id_q50, prev_renewable_ratio_id_q75, prev_business_hour_spread_mean, prev_business_hour_spread_std, prev_business_hour_spread_max, prev_business_hour_spread_min, prev_business_hour_spread_q25, prev_business_hour_spread_q50, prev_business_hour_spread_q75, prev_peak_hour_spread_mean, prev_peak_hour_spread_std, prev_peak_hour_spread_max, prev_peak_hour_spread_min, prev_peak_hour_spread_q25, prev_peak_hour_spread_q50, prev_peak_hour_spread_q75, prev_valley_hour_spread_mean, prev_valley_hour_spread_std, prev_valley_hour_spread_max, prev_valley_hour_spread_min, prev_valley_hour_spread_q25, prev_valley_hour_spread_q50, prev_valley_hour_spread_q75, prev_tieline_da_mean, prev_tieline_da_std, prev_tieline_da_max, prev_tieline_da_min, prev_tieline_da_q25, prev_tieline_da_q50, prev_tieline_da_q75, prev_hydro_da_mean, prev_hydro_da_std, prev_hydro_da_max, prev_hydro_da_min, prev_hydro_da_q25, prev_hydro_da_q50, prev_hydro_da_q75, prev_nonmarket_da_mean, prev_nonmarket_da_std, prev_nonmarket_da_max, prev_nonmarket_da_min, prev_nonmarket_da_q25, prev_nonmarket_da_q50, prev_nonmarket_da_q75, weekofyear_sin, weekofyear_cos, is_partial_week, forecast_weekly_net_demand_mwh, actual_weekly_net_demand_mwh, lt_price_w, da_id_cross_corr_w, extreme_price_spike_flag_w, extreme_event_flag_w, lt_settlement_base, spot_marginal_exposure, ancillary_peak_shaving_pause, ancillary_freq_reserve_tight, renewable_mechanism_active, mechanism_price_floor, mechanism_price_ceiling, mechanism_volume_ratio_max, lt_price_linked_active, fixed_price_ratio_max, linked_price_ratio_min, forward_price_linkage_days, forward_mechanism_execution_days, forward_ancillary_coupling_days, forward_info_forecast_boundary_days, lt_price_fixed_ratio, lt_price_linked_ratio
- `policy_feature_weights`
  起止索引: [152, 158)
  维度: 6
  字段: renewable_mechanism_active, lt_price_linked_active, forward_price_linkage_days, forward_mechanism_execution_days, forward_ancillary_coupling_days, forward_info_forecast_boundary_days
- `contract_curve_latent`
  起止索引: [158, 170)
  维度: 12
  字段: latent_0, latent_1, latent_2, latent_3, latent_4, latent_5, latent_6, latent_7, latent_8, latent_9, latent_10, latent_11
- `action_head`
  起止索引: [170, 176)
  维度: 6
  字段: latent_0, latent_1, latent_2, latent_3, latent_4, latent_5

## 下层参数块

- `spread_response`
  起止索引: [0, 8)
  维度: 8
  字段: price_spread
- `load_deviation_response`
  起止索引: [8, 16)
  维度: 8
  字段: load_dev
- `renewable_response`
  起止索引: [16, 24)
  维度: 8
  字段: renewable_dev
- `policy_shrink_response`
  起止索引: [24, 32)
  维度: 8
  字段: ancillary_freq_reserve_tight

## 前瞻制度状态消费位置

- `forward_ancillary_coupling_days` -> `policy_feature_weights`
- `forward_info_forecast_boundary_days` -> `policy_feature_weights`
- `forward_mechanism_execution_days` -> `policy_feature_weights`
- `forward_price_linkage_days` -> `policy_feature_weights`
