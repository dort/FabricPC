"""
Plotting utilities for continual learning benchmarks.

Provides functions for visualizing accuracy matrices,
forgetting analysis, and method comparisons.
"""

from __future__ import annotations

from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cl_benchmark.evaluation.results import BenchmarkResults


def plot_accuracy_matrix(
    matrix: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = True,
    title: str = "Accuracy Matrix",
    figsize: tuple[int, int] = (8, 6),
):
    """
    Plot accuracy matrix as a heatmap.

    Args:
        matrix: Accuracy matrix of shape (T, T)
        save_path: Optional path to save the figure
        show: Whether to display the plot
        title: Plot title
        figsize: Figure size
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    num_tasks = matrix.shape[0]

    # Create heatmap
    im = ax.imshow(matrix, cmap="YlGn", vmin=0, vmax=1, aspect="auto")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Accuracy")

    # Add value annotations
    for i in range(num_tasks):
        for j in range(num_tasks):
            if i >= j:  # Only show upper triangle where data exists
                value = matrix[i, j]
                color = "white" if value > 0.5 else "black"
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=8,
                )

    # Labels
    ax.set_xlabel("Evaluated on Task")
    ax.set_ylabel("After Training Task")
    ax.set_title(title)

    ax.set_xticks(range(num_tasks))
    ax.set_yticks(range(num_tasks))
    ax.set_xticklabels([f"T{i}" for i in range(num_tasks)])
    ax.set_yticklabels([f"T{i}" for i in range(num_tasks)])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_forgetting_analysis(
    matrix: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: tuple[int, int] = (10, 4),
):
    """
    Plot forgetting analysis with two subplots:
    1. Per-task forgetting bars
    2. Accuracy evolution over time for each task

    Args:
        matrix: Accuracy matrix of shape (T, T)
        save_path: Optional path to save the figure
        show: Whether to display the plot
        figsize: Figure size
    """
    import matplotlib.pyplot as plt

    from cl_benchmark.metrics import compute_forgetting

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    num_tasks = matrix.shape[0]
    forgetting = compute_forgetting(matrix)

    # Plot 1: Forgetting bars
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, num_tasks))
    bars = ax1.bar(range(num_tasks), forgetting, color=colors)
    ax1.set_xlabel("Task")
    ax1.set_ylabel("Forgetting")
    ax1.set_title("Per-Task Forgetting")
    ax1.set_xticks(range(num_tasks))
    ax1.set_xticklabels([f"T{i}" for i in range(num_tasks)])
    ax1.set_ylim(0, max(0.5, max(forgetting) * 1.1))

    # Add value labels on bars
    for bar, val in zip(bars, forgetting):
        if val > 0.01:
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # Plot 2: Accuracy evolution
    colors = plt.cm.tab10(np.linspace(0, 1, num_tasks))
    for task in range(num_tasks):
        accuracies = matrix[task:, task]  # Accuracy on task 'task' over time
        x = range(task, num_tasks)
        ax2.plot(
            x, accuracies, "o-", color=colors[task], label=f"Task {task}", markersize=4
        )

    ax2.set_xlabel("After Training Task")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Evolution")
    ax2.set_xticks(range(num_tasks))
    ax2.set_xticklabels([f"T{i}" for i in range(num_tasks)])
    ax2.legend(loc="lower left", fontsize=8)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_learning_curves(
    results: "BenchmarkResults",
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: tuple[int, int] = (10, 6),
):
    """
    Plot learning curves showing accuracy on each task over time.

    Args:
        results: BenchmarkResults containing accuracy matrices
        save_path: Optional path to save the figure
        show: Whether to display the plot
        figsize: Figure size
    """
    import matplotlib.pyplot as plt

    if not results.accuracy_matrices:
        print("No accuracy matrices to plot.")
        return

    fig, ax = plt.subplots(figsize=figsize)

    mean_matrix = results.get_mean_accuracy_matrix()
    std_matrix = results.get_std_accuracy_matrix()
    num_tasks = mean_matrix.shape[0]

    colors = plt.cm.tab10(np.linspace(0, 1, num_tasks))

    for task in range(num_tasks):
        # Get accuracy on this task over time (after it was learned)
        mean_accs = mean_matrix[task:, task]
        std_accs = std_matrix[task:, task]
        x = np.arange(task, num_tasks)

        ax.plot(x, mean_accs, "o-", color=colors[task], label=f"Task {task}")
        ax.fill_between(
            x,
            mean_accs - std_accs,
            mean_accs + std_accs,
            color=colors[task],
            alpha=0.2,
        )

    ax.set_xlabel("After Training Task")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Learning Curves: {results.model_name}")
    ax.set_xticks(range(num_tasks))
    ax.set_xticklabels([f"T{i}" for i in range(num_tasks)])
    ax.legend(loc="lower left")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_method_comparison(
    results_dict: Dict[str, "BenchmarkResults"],
    metric: str = "average_accuracy",
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: tuple[int, int] = (10, 6),
):
    """
    Plot comparison of multiple methods.

    Args:
        results_dict: Dictionary mapping method names to BenchmarkResults
        metric: Which metric to compare ("average_accuracy", "forgetting", "bwt")
        save_path: Optional path to save the figure
        show: Whether to display the plot
        figsize: Figure size
    """
    import matplotlib.pyplot as plt

    if not results_dict:
        print("No results to compare.")
        return

    fig, ax = plt.subplots(figsize=figsize)

    # Get metric values
    methods = list(results_dict.keys())
    means = []
    stds = []

    for name in methods:
        results = results_dict[name]
        if metric == "average_accuracy":
            means.append(results.accuracy_mean)
            stds.append(results.accuracy_std)
        elif metric == "forgetting":
            means.append(results.forgetting_mean)
            stds.append(results.forgetting_std)
        elif metric == "bwt":
            means.append(results.backward_transfer)
            stds.append(0.0)  # No std for BWT currently
        else:
            raise ValueError(f"Unknown metric: {metric}")

    # Sort by mean value
    sorted_idx = np.argsort(means)[::-1]
    methods = [methods[i] for i in sorted_idx]
    means = [means[i] for i in sorted_idx]
    stds = [stds[i] for i in sorted_idx]

    # Create bar chart
    x = np.arange(len(methods))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(methods)))

    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors)

    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        label = f"{mean:.3f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.01,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xlabel("Method")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Method Comparison: {metric.replace('_', ' ').title()}")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right")

    if metric == "average_accuracy":
        ax.set_ylim(0, 1.1)
    elif metric == "forgetting":
        ax.set_ylim(0, max(means) * 1.3 if means else 0.5)

    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()
