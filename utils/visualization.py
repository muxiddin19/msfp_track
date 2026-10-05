"""
Visualization Utilities for LITE++

Functions for visualizing tracking results and analysis.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def draw_tracks(
    image: np.ndarray,
    tracks: List[Dict],
    colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
    show_id: bool = True,
    show_confidence: bool = False,
    line_thickness: int = 2,
) -> np.ndarray:
    """
    Draw tracking results on an image.

    Args:
        image: Input image (BGR, HWC)
        tracks: List of track dicts with 'id', 'bbox', optionally 'confidence'
        colors: Optional color mapping {track_id: (B, G, R)}
        show_id: Whether to display track ID
        show_confidence: Whether to display confidence
        line_thickness: Bounding box line thickness

    Returns:
        Image with drawn tracks
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV is required for visualization")

    output = image.copy()

    for track in tracks:
        track_id = track.get("id", 0)
        bbox = track.get("bbox", [0, 0, 0, 0])
        conf = track.get("confidence", 1.0)

        x1, y1, x2, y2 = map(int, bbox[:4])

        # Get color for this track
        if colors and track_id in colors:
            color = colors[track_id]
        else:
            # Generate color from ID
            np.random.seed(track_id)
            color = tuple(map(int, np.random.randint(0, 255, 3)))

        # Draw bounding box
        cv2.rectangle(output, (x1, y1), (x2, y2), color, line_thickness)

        # Draw label
        label_parts = []
        if show_id:
            label_parts.append(f"ID:{track_id}")
        if show_confidence:
            label_parts.append(f"{conf:.2f}")

        if label_parts:
            label = " ".join(label_parts)
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )

            cv2.rectangle(
                output,
                (x1, y1 - label_h - baseline - 5),
                (x1 + label_w, y1),
                color,
                -1,
            )
            cv2.putText(
                output,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

    return output


def plot_threshold_history(
    thresholds: List[float],
    save_path: Optional[str] = None,
    title: str = "Adaptive Threshold Over Time",
    figsize: Tuple[int, int] = (12, 4),
) -> None:
    """
    Plot adaptive threshold values over video frames.

    Args:
        thresholds: List of threshold values per frame
        save_path: Optional path to save the plot
        title: Plot title
        figsize: Figure size
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Matplotlib is required for plotting")

    fig, ax = plt.subplots(figsize=figsize)

    frames = np.arange(len(thresholds))
    ax.plot(frames, thresholds, "b-", linewidth=1, alpha=0.7)

    # Add moving average
    window = min(30, len(thresholds) // 10)
    if window > 1:
        ma = np.convolve(thresholds, np.ones(window) / window, mode="valid")
        ax.plot(
            frames[window - 1 :], ma, "r-", linewidth=2, label=f"MA({window})"
        )

    ax.set_xlabel("Frame")
    ax.set_ylabel("Confidence Threshold")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add statistics annotation
    stats_text = (
        f"Mean: {np.mean(thresholds):.3f}\n"
        f"Std: {np.std(thresholds):.3f}\n"
        f"Min: {np.min(thresholds):.3f}\n"
        f"Max: {np.max(thresholds):.3f}"
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_feature_tsne(
    features: np.ndarray,
    labels: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Feature t-SNE Visualization",
    perplexity: int = 30,
    figsize: Tuple[int, int] = (10, 10),
) -> None:
    """
    Create t-SNE visualization of feature embeddings.

    Args:
        features: (N, D) feature vectors
        labels: (N,) identity labels
        save_path: Optional path to save the plot
        title: Plot title
        perplexity: t-SNE perplexity parameter
        figsize: Figure size
    """
    try:
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
    except ImportError:
        raise ImportError("Matplotlib and scikit-learn are required for t-SNE")

    # Compute t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embeddings_2d = tsne.fit_transform(features)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[i]],
            label=f"ID {label}",
            alpha=0.7,
            s=20,
        )

    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    # Only show legend if few classes
    if len(unique_labels) <= 20:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def create_comparison_table(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = ["HOTA", "DetA", "AssA", "MOTA", "IDF1"],
    highlight_best: bool = True,
) -> str:
    """
    Create a formatted comparison table for tracking results.

    Args:
        results: Dict mapping method names to metric dictionaries
        metrics: Metrics to include in table
        highlight_best: Whether to highlight best values

    Returns:
        Formatted table string (Markdown format)
    """
    methods = list(results.keys())

    # Header
    header = "| Method | " + " | ".join(metrics) + " |"
    separator = "|" + "|".join(["---"] * (len(metrics) + 1)) + "|"

    # Find best values
    best_values = {}
    if highlight_best:
        for metric in metrics:
            values = [results[m].get(metric, 0) for m in methods]
            best_values[metric] = max(values)

    # Rows
    rows = []
    for method in methods:
        row_values = []
        for metric in metrics:
            value = results[method].get(metric, 0)
            if highlight_best and value == best_values[metric]:
                row_values.append(f"**{value:.1f}**")
            else:
                row_values.append(f"{value:.1f}")
        rows.append(f"| {method} | " + " | ".join(row_values) + " |")

    return "\n".join([header, separator] + rows)
