from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
import time
from typing import Any


def print_section(title: str) -> None:
    line = "=" * 80
    print(f"\n{line}\n{title}\n{line}")


def run_command(command: list[str]) -> str | None:
    if not shutil.which(command[0]):
        return None
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return result.stderr.strip() or None
    return result.stdout.strip() or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test a Google Colab GPU runtime and verify whether it is an A100."
    )
    parser.add_argument(
        "--size",
        type=int,
        default=4096,
        help="Matrix side length for the CUDA matmul benchmark.",
    )
    parser.add_argument(
        "--matmul-iters",
        type=int,
        default=20,
        help="Number of timed iterations for the CUDA matmul benchmark.",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=80,
        help="Number of optimization steps in the tiny training loop.",
    )
    parser.add_argument(
        "--require-a100",
        action="store_true",
        help="Exit with a non-zero code unless the detected GPU name contains 'A100'.",
    )
    return parser.parse_args()


def benchmark_matmul(torch: Any, size: int, iters: int, dtype: Any) -> tuple[float, float]:
    device = "cuda"
    a = torch.randn(size, size, device=device, dtype=dtype)
    b = torch.randn(size, size, device=device, dtype=dtype)

    for _ in range(5):
        _ = a @ b
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        _ = a @ b
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    ops = 2 * (size**3) * iters
    tflops = ops / elapsed / 1e12
    return elapsed / iters, tflops


def tiny_training_loop(torch: Any, steps: int) -> tuple[float, float]:
    device = "cuda"
    batch_size = 8192
    features = 128

    x = torch.randn(batch_size, features, device=device)
    true_w = torch.randn(features, 1, device=device)
    y = x @ true_w + 0.05 * torch.randn(batch_size, 1, device=device)

    model = torch.nn.Sequential(
        torch.nn.Linear(features, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 1),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    start = time.perf_counter()
    last_loss = 0.0
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x)
        loss = torch.nn.functional.mse_loss(prediction, y)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())

        report_every = max(steps // 4, 1)
        if (step + 1) % report_every == 0 or step == 0:
            print(f"[train] step={step + 1:03d}/{steps} loss={last_loss:.6f}")

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return last_loss, elapsed


def main() -> int:
    args = parse_args()

    print_section("Environment")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"Executable: {sys.executable}")

    print_section("nvidia-smi")
    query = run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )
    if query:
        print(query)
    else:
        print("nvidia-smi is unavailable.")

    try:
        import torch  # type: ignore
    except ImportError:
        print_section("PyTorch")
        print("PyTorch is not installed in this environment.")
        print("On Colab GPU runtimes it is normally preinstalled.")
        print("If needed, run: !pip install torch --index-url https://download.pytorch.org/whl/cu121")
        return 1

    print_section("PyTorch / CUDA")
    print(f"torch: {torch.__version__}")
    print(f"torch CUDA build: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")

    if not torch.cuda.is_available():
        print("No CUDA device is available, so this is not a Colab GPU runtime.")
        return 2

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device_name = torch.cuda.get_device_name(0)
    properties = torch.cuda.get_device_properties(0)
    total_memory_gb = properties.total_memory / (1024**3)
    capability = torch.cuda.get_device_capability(0)
    is_a100 = "A100" in device_name.upper()

    print(f"Device name: {device_name}")
    print(f"Capability: {capability[0]}.{capability[1]}")
    print(f"Total memory: {total_memory_gb:.2f} GiB")
    print(f"A100 detected: {is_a100}")

    if args.require_a100 and not is_a100:
        print("The runtime is using a GPU, but it is not an A100.")
        return 3

    print_section("Matmul Benchmark")
    fp32_avg, fp32_tflops = benchmark_matmul(
        torch=torch,
        size=args.size,
        iters=args.matmul_iters,
        dtype=torch.float32,
    )
    print(
        f"float32 matmul: size={args.size}, avg_time={fp32_avg:.4f}s, throughput={fp32_tflops:.2f} TFLOP/s"
    )

    fp16_avg, fp16_tflops = benchmark_matmul(
        torch=torch,
        size=args.size,
        iters=args.matmul_iters,
        dtype=torch.float16,
    )
    print(
        f"float16 matmul: size={args.size}, avg_time={fp16_avg:.4f}s, throughput={fp16_tflops:.2f} TFLOP/s"
    )

    print_section("Tiny Training Loop")
    final_loss, train_elapsed = tiny_training_loop(torch=torch, steps=args.train_steps)
    print(f"final_loss={final_loss:.6f}")
    print(f"elapsed={train_elapsed:.2f}s for {args.train_steps} steps")

    print_section("Result")
    if is_a100:
        print("PASS: CUDA is available and the runtime reports an NVIDIA A100 GPU.")
    else:
        print("PASS (non-A100): CUDA is available, but the GPU is not an A100.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
