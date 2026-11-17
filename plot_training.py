"""Plot selected epoch statistics in a single row of panels."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


METRICS = {
    "g_loss": ("Generator Loss", "Loss"),
    "d_loss": ("Discriminator Loss", "Loss"),
    "seconds": ("Epoch training time", "Seconds"),
    "images_per_second": ("Training throughput", "Images per second"),
    "peak_allocated_gib": ("Peak allocated GPU memory", "GiB"),
    "fid": ("Validation FID", "FID"),
    "inception_score": ("Inception Score", "IS"),
}


def plot_statistics(csv_path, metrics, output_path, *, xlim=None):
    with csv_path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        missing = [name for name in ("epoch", *metrics) if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing CSV columns: {', '.join(missing)}")
        rows = sorted(reader, key=lambda row: int(row["epoch"]))

    figure, axes = plt.subplots(
        1, len(metrics), figsize=(2.8 * len(metrics), 1.95),
        sharex=True, squeeze=False, layout="constrained",
    )
    colors = ("#2166ac", "#a54b17", "#287d5b", "#805699")
    for index, (axis, metric) in enumerate(zip(axes[0], metrics)):
        # Evaluation metrics only appear in the CSV on scheduled evaluation epochs.
        points = [(int(row["epoch"]), float(row[metric])) for row in rows if row[metric]]
        if not points:
            raise ValueError(f"No recorded values for {metric}")
        epochs, values = zip(*points)
        axis.plot(epochs, values, color=colors[index % len(colors)], linewidth=1.35)
        title, ylabel = METRICS[metric]
        axis.set(title=title, xlabel="Epoch", ylabel=ylabel)
        axis.title.set_fontsize(10)
        axis.xaxis.label.set_size(8)
        axis.yaxis.label.set_size(8)
        axis.tick_params(labelsize=8)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.grid(axis="y", color="#dedede", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)

    if xlim is not None:
        axes[0, 0].set_xlim(*xlim)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, facecolor="white")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="Path to the run's epochs.csv")
    parser.add_argument("--metrics", nargs="+", choices=tuple(METRICS), default=("g_loss", "d_loss"))
    parser.add_argument("--output", type=Path, help="Output image path")
    args = parser.parse_args()
    output = args.output or args.csv_path.with_name("training_statistics.png")
    plot_statistics(args.csv_path, args.metrics, output)
    print(output.resolve())


if __name__ == "__main__":
    main()
