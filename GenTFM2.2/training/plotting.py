"""Save the final training and validation loss curves without opening a GUI."""
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator


def save_loss_curve(train_history, validation_history, path):
    """Save (step, loss) histories to a PNG; return its path.

    Training contains every optimizer step. Validation contains only the actual
    validation steps; it is plotted at those steps without resampling.
    """
    if not train_history:
        raise ValueError("train_history must contain at least one point")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(9, 5))
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    steps, losses = zip(*train_history)
    axis.plot(steps, losses, label="Train loss", color="#2563eb", linewidth=1.2,
              marker="o" if len(train_history) == 1 else None)
    if validation_history:
        steps, losses = zip(*validation_history)
        axis.plot(steps, losses, label="Validation loss", color="#ea580c",
                  linewidth=1.8, marker="o", markersize=4)
    axis.set(title="Flow matching loss", xlabel="Training step", ylabel="Velocity MSE")
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160, format="png")
    return path
