"""
ENGINE-TWIN: Master Automated Test & Validation Runner (SIH26054)
Executes all project verification suites and outputs a consolidated test report:
1. Physics & Simulation Tests (test_simulation.py)
2. AI Layer, EKF, RUL & XAI Tests (test_ai_layer.py)
3. Scientific Benchmark vs Baselines (validation/benchmark.py)
4. Architectural Ablation Studies (validation/ablation.py)
"""
import sys
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

def run_suite(name: str, script_rel_path: str):
    print("\n" + "#" * 75)
    print(f"  RUNNING SUITE: {name}")
    print("#" * 75)
    t0 = time.time()
    script_path = PROJECT_ROOT / script_rel_path
    res = subprocess.run([sys.executable, str(script_path)], cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    elapsed = time.time() - t0
    
    print(res.stdout)
    if res.returncode != 0:
        print(f"[FAIL] {name} FAILED with code {res.returncode}")
        print("ERROR OUTPUT:\n", res.stderr)
        return False, elapsed
    else:
        print(f"[PASS] {name} PASSED in {elapsed:.2f}s")
        return True, elapsed

def main():
    print("=" * 80)
    print("  ENGINE-TWIN: MASTER CONSOLIDATED TEST & VALIDATION PASS (DRDO SIH26054)")
    print("=" * 80)

    suites = [
        ("Step 1: Physics Engine, ISA Atmosphere & Sensor Dynamics", "test_simulation.py"),
        ("Step 2: AI Anomaly, EKF Fusion, Fault Classifier, RUL & XAI", "test_ai_layer.py"),
        ("Step 3: Dedicated Data Leakage & Provenance Integrity Audit", "validation/leakage_audit.py"),
        ("Step 4: Adversarial Stress & Failure Engineering Suite", "validation/stress_testing.py"),
        ("Step 5: Empirical Baseline Comparison (EIS vs ML vs ENGINE-TWIN)", "validation/benchmark.py"),
        ("Step 6: Architectural Ablation Study (Physics vs Raw ML)", "validation/ablation.py"),
        ("Step 7: Multi-Engine Scalability Proof (Rotax 914 Flat-Four)", "tools/test_multi_engine_transfer.py"),
        ("Step 8: Edge Hardware AI Compute & Memory Resource Profiler", "tools/edge_benchmark_profiler.py")
    ]

    results = []
    total_t0 = time.time()
    for name, path in suites:
        passed, elapsed = run_suite(name, path)
        results.append((name, passed, elapsed))

    total_elapsed = time.time() - total_t0

    print("\n" + "=" * 75)
    print("  FINAL CONSOLIDATED TEST SUMMARY")
    print("=" * 75)
    all_passed = True
    for name, passed, elapsed in results:
        status_str = "[PASS]" if passed else "[FAIL]"
        if not passed:
            all_passed = False
        print(f"  {status_str:<8} | {name:<55} | ({elapsed:.2f}s)")

    print("-" * 75)
    print(f"  Total Test Time: {total_elapsed:.2f}s")
    if all_passed:
        print("  [SUCCESS] ALL VERIFICATION SUITES PASSED! 100% OPERATIONAL INTEGRITY.")
    else:
        print("  [WARNING] ONE OR MORE TEST SUITES FAILED.")
    print("=" * 75)

if __name__ == "__main__":
    main()
