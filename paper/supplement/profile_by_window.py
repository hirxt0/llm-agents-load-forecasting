"""Reproduce the descriptive comparison in § 6.1 from the saved hourly forecasts."""
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from one_agent_forecasting.src.config import RESULTS
from one_agent_forecasting.src.summarize import load_windows, reference_predictions


def plot_examples(table, agent, profile, origins):
    figure = Figure(figsize=(10, 4.2), layout="constrained")
    axes = figure.subplots(1, 2)
    for axis, (dataset, group) in zip(axes, table.groupby("dataset", sort=True)):
        ranked = group.sort_values("gain_pp", kind="stable")
        key = ranked.index[(len(ranked) - 1) // 2]
        forecast = agent.loc[key].sort_index()
        reference = profile.loc[key].sort_index()
        axis.plot(forecast.index, forecast.y_true, color="#222222", linewidth=2.2, label="Показания счётчика")
        axis.plot(forecast.index, forecast.y_pred, color="#006b88", marker="o", markersize=3, label="GPT-5, series")
        axis.plot(reference.index, reference.y_pred, color="#ad5117", linestyle="--", label="Опорный профиль")
        axis.set(title=f"{dataset}  ·  {origins.at[key]:%d.%m.%Y}", xlabel="Час суток",
                 ylabel="Потребление, кВт·ч за час", xlim=(0, 23), xticks=[0, 4, 8, 12, 16, 20, 23])
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    figure.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=3, frameon=False)
    figure.savefig(Path(__file__).with_name("profile_examples.png"), dpi=180)


def main():
    windows, history, meta = load_windows("test", "main")
    keys = ["building_id", "window_id", "hour"]
    profile = reference_predictions(windows).query("method == 'ref_day_profile'").set_index(keys)
    agent = pd.read_csv(RESULTS / "fc_gpt-5-low_series_noisy_test" / "predictions.csv").set_index(keys)
    if not (agent.index.is_unique and profile.index.is_unique
            and agent.index.sort_values().equals(profile.index.sort_values())):
        raise ValueError("Forecasts and profiles must cover exactly the same window-hours")
    agent = agent.reindex(profile.index)
    np.testing.assert_allclose(agent.y_true, profile.y_true, rtol=0, atol=1e-10)
    squared = pd.DataFrame({
        "agent_nrmse": (agent.y_pred - profile.y_true) ** 2,
        "profile_nrmse": (profile.y_pred - profile.y_true) ** 2,
        "distance_pct": (agent.y_pred - profile.y_pred) ** 2,
    })
    errors = np.sqrt(squared.groupby(level=keys[:2]).mean())
    scale = pd.Series({key: values.mean() for key, values in history.items()})
    errors = errors.div(scale.rename_axis(keys[:2]), axis="index") * 100
    errors["gain_pp"] = errors.profile_nrmse - errors.agent_nrmse
    table = meta.join(errors, validate="one_to_one")
    table.to_csv(Path(__file__).with_name("profile_by_window.csv"))
    origins = windows.loc[windows.step == 0].set_index(keys[:2]).timestamp
    plot_examples(table, agent, profile, origins)
    print(table.groupby("dataset").agg(
        windows=("gain_pp", "size"),
        wins=("gain_pp", lambda values: (values > 0).sum()),
        ties=("gain_pp", lambda values: (values == 0).sum()),
        median_gain_pp=("gain_pp", "median"),
        median_distance_pct=("distance_pct", "median"),
    ))
    print(table.assign(win=table.gain_pp > 0).groupby(["dataset", "day_type"]).win.agg(["sum", "size"]))


if __name__ == "__main__":
    main()
