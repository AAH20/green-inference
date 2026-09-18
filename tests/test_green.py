import unittest
from green_inference.saccadic.motion_gradient import SaccadicPixelPruner
from green_inference.cache.dynamic_kv_eviction import DynamicKVEvictionEngine
from green_inference.thermal.power_governor import ThermalPowerGovernor, RackThermalTelemetry

class TestGreenInference(unittest.TestCase):
    def test_saccadic_pruning_ratio(self):
        pruner = SaccadicPixelPruner(patch_size=16, gradient_threshold=10.0)
        # 32x32 image = 4 patches
        f1 = [[0 for _ in range(32)] for _ in range(32)]
        pruner.prune_frame(f1, width=32, height=32)

        # Frame 2: Only 1 patch moves
        f2 = [[0 for _ in range(32)] for _ in range(32)]
        for y in range(16):
            for x in range(16):
                f2[y][x] = 200

        res = pruner.prune_frame(f2, width=32, height=32)
        self.assertEqual(res["total_patches"], 4)
        self.assertEqual(res["active_patches"], 1)
        self.assertEqual(res["pruning_ratio_pct"], 75.0)

    def test_kv_cache_eviction(self):
        engine = DynamicKVEvictionEngine(max_retained_tokens=5)
        tokens = [{"id": f"t{i}", "initial_importance": float(i)} for i in range(10)]
        res = engine.ingest_tokens(tokens)
        self.assertEqual(res["retained_tokens"], 5)
        self.assertEqual(res["evicted_tokens"], 5)
        # Ensure highest importance tokens kept
        retained_ids = [t["token_id"] for t in engine.kv_tokens]
        self.assertIn("t9", retained_ids)
        self.assertIn("t8", retained_ids)

    def test_thermal_governor_tier_downscale(self):
        gov = ThermalPowerGovernor(target_temp_c=82.0)
        t_cool = RackThermalTelemetry("GPU-1", 60.0, 300.0, 450.0)
        r_cool = gov.arbitrate_model_tier(t_cool)
        self.assertEqual(r_cool["recommended_tier"], "TIER_1_UNCONSTRAINED_FULL")

        t_hot = RackThermalTelemetry("GPU-2", 85.0, 440.0, 450.0)
        r_hot = gov.arbitrate_model_tier(t_hot)
        self.assertEqual(r_hot["recommended_tier"], "TIER_3_EDGE_SLM_EMERGENCY")
        self.assertTrue(r_hot["recommended_power_cap_watts"] < 250.0)

if __name__ == "__main__":
    unittest.main()
