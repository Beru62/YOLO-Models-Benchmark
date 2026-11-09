from pathlib import Path

import torch
from ultralytics.utils.benchmarks import benchmark

MODEL_PATH = Path(
    r"D:\path\to\best.pt"
)

DATA_YAML = Path(
    r"D:\path\to\data.yaml"
)

IMGSZ = 640
HALF = True # True = FP16, False = FP32


def main() -> None:
    gpu_available = torch.cuda.is_available()
    gpu_device = 0 if gpu_available else "cpu"

    formats: list[tuple[str, int | str]] = [
        ("-", gpu_device), #pytorch
        ("torchscript", gpu_device),
        ("onnx", gpu_device),
        ("openvino", "cpu"),
    ]

    if gpu_available:
        formats.append(("engine", 0)) # TensorRT engine

    for fmt, device in formats:
        print("\n" + "=" * 60)
        print(f"FORMAT: {fmt} | DEVICE: {device}")
        print("=" * 60)

        try:
            results = benchmark(
                model=str(MODEL_PATH),
                data=str(DATA_YAML),
                imgsz=IMGSZ,
                device=device,
                format=fmt,
                half=HALF,
            )

            print(results)

        except Exception as error:
            print(f"FAILED: {fmt}")
            print(f"Error: {error}")


if __name__ == "__main__":
    main()
