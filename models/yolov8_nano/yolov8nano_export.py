#!/usr/bin/env python3
"""Export YOLOv8-nano to ONNX for OPU benchmarking.

Uses ultralytics built-in ONNX export.

Usage:
    conda run -n merlin-dev uv run python models/yolov8_nano/yolov8nano_export.py
"""

import os


def export_yolov8n(output_dir, imgsz=320):
    """Export YOLOv8-nano to ONNX."""
    from ultralytics import YOLO

    print("Loading YOLOv8n...")
    model = YOLO("yolov8n.pt")

    print(f"Exporting to ONNX (imgsz={imgsz})...")
    onnx_path = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=18,
        simplify=True,
    )

    # Move to our directory
    target = os.path.join(output_dir, "yolov8n.onnx")
    if onnx_path != target:
        import shutil

        shutil.move(onnx_path, target)
    print(f"Exported to {target}")
    return target


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    export_yolov8n(script_dir)
