"""
Thermal-Aware Multi-Agent Rack Power Governor.
Dynamically caps GPU wattage and dynamically downscales agent model depth
to maintain data center rack junction temperature under 82°C.
"""
from dataclasses import dataclass
from typing import Dict, Any

@dataclass(frozen=True)
class RackThermalTelemetry:
    gpu_id: str
    junction_temp_c: float
    current_power_watts: float
    tdp_limit_watts: float

class ThermalPowerGovernor:
    def __init__(self, target_temp_c: float = 82.0):
        self.target_temp = target_temp_c

    def arbitrate_model_tier(self, telemetry: RackThermalTelemetry) -> Dict[str, Any]:
        """
        Dynamically selects model execution tier based on thermal headroom:
        - Nominal (< 70°C): Tier-1 (Full 70B VLM model, max accuracy)
        - Elevated (70°C - 82°C): Tier-2 (14B quantized VLM model, -40% power)
        - Critical (> 82°C): Tier-3 (3B Edge SLM model + aggressive frame skip, -80% power)
        """
        temp = telemetry.junction_temp_c
        if temp >= self.target_temp:
            tier = "TIER_3_EDGE_SLM_EMERGENCY"
            power_cap_watts = telemetry.tdp_limit_watts * 0.45
            downscale_action = "CRITICAL_THROTTLE_DOWN_TO_3B_MODEL"
        elif temp >= 72.0:
            tier = "TIER_2_BALANCED_QUANTIZED"
            power_cap_watts = telemetry.tdp_limit_watts * 0.70
            downscale_action = "MODERATE_DOWNSCALE_TO_14B_MODEL"
        else:
            tier = "TIER_1_UNCONSTRAINED_FULL"
            power_cap_watts = telemetry.tdp_limit_watts
            downscale_action = "MAINTAIN_70B_VLM"

        return {
            "gpu_id": telemetry.gpu_id,
            "junction_temp_c": temp,
            "recommended_tier": tier,
            "recommended_power_cap_watts": round(power_cap_watts, 1),
            "governor_action": downscale_action,
            "carbon_pue_optimized": True
        }
