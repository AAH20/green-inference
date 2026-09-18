"""
GreenInference CLI: Green AI, Saccadic Pruning & Thermal Capping Suite.
"""
import argparse
from .saccadic.motion_gradient import SaccadicPixelPruner
from .cache.dynamic_kv_eviction import DynamicKVEvictionEngine
from .thermal.power_governor import ThermalPowerGovernor, RackThermalTelemetry

def main():
    parser = argparse.ArgumentParser(
        prog="green-inference",
        description="Sub-Watt Saccadic Vision & Power-Aware KV-Cache Engine for Hyperscale Agent Data Centers."
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # prune-saccadic
    subparsers.add_parser("prune-saccadic", help="Simulate saccadic optical gradient background pruning on 4K stream")

    # evict-kv
    subparsers.add_parser("evict-kv", help="Benchmark dynamic temporal KV-cache eviction and memory savings")

    # govern-thermal
    subparsers.add_parser("govern-thermal", help="Evaluate rack thermal power capping across GPU temperatures")

    args = parser.parse_args()

    if args.command == "prune-saccadic":
        pruner = SaccadicPixelPruner(patch_size=16, gradient_threshold=10.0)
        # Create 160x120 dummy luma frame (simulating scaled 4K video)
        f1 = [[50 for _ in range(160)] for _ in range(120)]
        pruner.prune_frame(f1, width=160, height=120)
        # Frame 2: Only 5% of pixels moving (e.g. car driving in parking lot)
        f2 = [[50 for _ in range(160)] for _ in range(120)]
        for y in range(20, 30):
            for x in range(20, 30):
                f2[y][x] = 180  # Moving target

        res = pruner.prune_frame(f2, width=160, height=120)
        print("[GreenInference] Saccadic Visual Pruning Benchmark:")
        print(f"  Total Video Patches : {res['total_patches']}")
        print(f"  Active Neural Patches: {res['active_patches']}")
        print(f"  Background Pruning  : {res['pruning_ratio_pct']}% (Discarded Static Pixels)")
        print(f"  Inference Wattage Cut: {res['wattage_reduction_pct']}% GPU Power Saved")

    elif args.command == "evict-kv":
        engine = DynamicKVEvictionEngine(max_retained_tokens=1000)
        # Stream 3000 tokens with decayed importance
        tokens = [{"id": f"tok_{i}", "content": "frame_feat", "initial_importance": 0.2 + (i % 10)*0.1} for i in range(3000)]
        res = engine.ingest_tokens(tokens)
        print("[GreenInference] Dynamic KV-Cache Eviction Benchmark:")
        print(f"  Retained KV Tokens  : {res['retained_tokens']}")
        print(f"  Evicted Tokens      : {res['evicted_tokens']}")
        print(f"  HBM Bandwidth Saved : {res['memory_bandwidth_saved_mb']} MB per forward pass")
        print(f"  Compression Ratio   : {res['cache_compression_ratio']}")

    elif args.command == "govern-thermal":
        gov = ThermalPowerGovernor(target_temp_c=82.0)
        gpus = [
            RackThermalTelemetry("GPU-01", 65.0, 350.0, 450.0),
            RackThermalTelemetry("GPU-02", 76.5, 410.0, 450.0),
            RackThermalTelemetry("GPU-03", 85.0, 445.0, 450.0)
        ]
        print("[GreenInference] Data Center Rack Thermal Governor:")
        for g in gpus:
            act = gov.arbitrate_model_tier(g)
            print(f"  [{act['gpu_id']}] Temp: {act['junction_temp_c']}°C -> {act['recommended_tier']} (Cap: {act['recommended_power_cap_watts']}W)")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
