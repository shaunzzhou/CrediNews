import matplotlib.pyplot as plt
import numpy as np

import pandas as pd


def save_results(
    model,
    perf_original_model_val_data,
    perf_original_model_val_data_rewritten,
    perf_combined_model_val_data,
    perf_combined_model_val_data_rewritten,
    perf_rewritten_model_val_data,
    perf_rewritten_model_val_data_rewritten,
):
    perf_original_model_val_data = pd.DataFrame(perf_original_model_val_data, index=[0])
    perf_original_model_val_data_rewritten = pd.DataFrame(
        perf_original_model_val_data_rewritten, index=[0]
    )
    perf_combined_model_val_data = pd.DataFrame(perf_combined_model_val_data, index=[0])
    perf_combined_model_val_data_rewritten = pd.DataFrame(
        perf_combined_model_val_data_rewritten, index=[0]
    )
    perf_rewritten_model_val_data = pd.DataFrame(
        perf_rewritten_model_val_data, index=[0]
    )
    perf_rewritten_model_val_data_rewritten = pd.DataFrame(
        perf_rewritten_model_val_data_rewritten, index=[0]
    )
    df = pd.concat(
        [
            perf_original_model_val_data,
            perf_original_model_val_data_rewritten,
            perf_combined_model_val_data,
            perf_combined_model_val_data_rewritten,
            perf_rewritten_model_val_data,
            perf_rewritten_model_val_data_rewritten,
        ]
    )
    df["model"] = model
    df["train-data"] = [
        "original",
        "original",
        "combined",
        "combined",
        "rewritten",
        "rewritten",
    ]
    df["test-data"] = [
        "original",
        "rewritten",
        "original",
        "rewritten",
        "original",
        "rewritten",
    ]

    df2 = df[
        [
            "model",
            "train-data",
            "test-data",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
        ]
    ].reset_index(drop=True)
    df2.to_csv(f"results/{model}_metrics.csv", index=False)
    print(f"Results saved to results/{model}_metrics.csv")


def plot_results(
    model,
    perf_original_model_val_data,
    perf_original_model_val_data_rewritten,
    perf_combined_model_val_data,
    perf_combined_model_val_data_rewritten,
    perf_rewritten_model_val_data,
    perf_rewritten_model_val_data_rewritten,
    ylim=1.1,
):
    """
    Plot performance metrics for three models (trained with original, LLM-rewritten, or combined data)
    on two test sets (Original Test Data and LLM-rewritten Test Data) in one row and three columns.
    Original Test Data results are displayed using the left y-axis and LLM-rewritten Test Data results on the right y-axis.
    Separate legends are shown: the left legend (Original Test Data) at the top left,
    and the right legend (LLM-rewritten Test Data) at the top right.
    """

    # Helper function to plot each subplot
    def plot_subplot(ax, data_left, data_right, title, left_color, right_color):
        categories = list(data_left.keys())
        x = np.arange(len(categories))
        bar_width = 0.4

        # Plot Original Test Data on primary axis (left)
        bars_left = ax.bar(
            x - bar_width / 2,
            list(data_left.values()),
            width=bar_width,
            color=left_color,
        )
        ax.set_ylim(0, ylim)
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        for i, val in enumerate(data_left.values()):
            ax.text(
                x[i] - bar_width / 2,
                val + 0.01,
                f"{val*100:.2f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # Add left legend at top left for Original Test Data
        ax.legend([bars_left], ["Original Test Data"], loc="upper left", fontsize=8)

        # Create twin axis for LLM-rewritten Test Data
        ax2 = ax.twinx()
        bars_right = ax2.bar(
            x + bar_width / 2,
            list(data_right.values()),
            width=bar_width,
            color=right_color,
        )
        ax2.set_ylim(0, ylim)
        for i, val in enumerate(data_right.values()):
            ax2.text(
                x[i] + bar_width / 2,
                val + 0.01,
                f"{val*100:.2f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # Add right legend at top right for LLM-rewritten Test Data
        ax2.legend(
            [bars_right], ["LLM-rewritten Test Data"], loc="upper right", fontsize=8
        )
        ax.set_title(title, fontsize=10)

    # Create 1 row, 3 columns of subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"{model} Model Results", fontsize=16)

    # Subplot 1: Model Trained with Original Data
    plot_subplot(
        axes[0],
        perf_original_model_val_data,
        perf_original_model_val_data_rewritten,
        "Model Trained with Original Data",
        left_color="skyblue",
        right_color="steelblue",
    )

    # Subplot 2: Model Trained with LLM-rewritten Test Data
    plot_subplot(
        axes[1],
        perf_rewritten_model_val_data,
        perf_rewritten_model_val_data_rewritten,
        "Model Trained with LLM-rewritten Data",
        left_color="lightgreen",
        right_color="seagreen",
    )

    # Subplot 3: Model Trained with Combined Data
    plot_subplot(
        axes[2],
        perf_combined_model_val_data,
        perf_combined_model_val_data_rewritten,
        "Model Trained with Combined Data",
        left_color="salmon",
        right_color="tomato",
    )

    plt.tight_layout()
    plt.show()
