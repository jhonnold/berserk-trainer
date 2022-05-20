import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os


def do_plots(root_dir, output):
    """Plots loss"""
    df = pd.read_csv(os.path.join(root_dir, "loss.csv"),
                     names=["step", "val_loss", "train_loss"],
                     header=None)
    df.set_index("step").plot()
    plt.savefig(output, dpi=300)


def main():
    parser = argparse.ArgumentParser(
        description="Generate plots of losses for an experiment run",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "root_dir",
        type=str,
        help="root directory with loss.csv",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="plot.png",
        help="plot output file name",
    )

    args = parser.parse_args()
    do_plots(args.root_dir, args.output)


if __name__ == "__main__":
    main()