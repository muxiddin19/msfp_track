#!/bin/bash
# LITE++ Experiment Runner
# Run all experiments for ECCV 2026 paper

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "LITE++ Experiments"
echo "=========================================="

# Check for required dependencies
python -c "import ultralytics" 2>/dev/null || {
    echo "Error: ultralytics not installed. Run: pip install ultralytics"
    exit 1
}

# Download YOLO weights if needed
if [ ! -f "yolov8m.pt" ]; then
    echo "Downloading YOLOv8m weights..."
    python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"
fi

# Run ablation study
echo ""
echo "Running fusion ablation study..."
python "$PROJECT_DIR/experiments/ablation_fusion.py" \
    --yolo_weights yolov8m.pt \
    --output_dir results/ablation_fusion

# Run MOT17 experiments
echo ""
echo "Running MOT17 experiments..."
python "$PROJECT_DIR/experiments/run_mot17.py" \
    --yolo_weights yolov8m.pt \
    --fusion_type attention \
    --adaptive_threshold \
    --output_dir results/mot17_litepp

echo ""
echo "=========================================="
echo "Experiments complete!"
echo "Results saved to: results/"
echo "=========================================="
