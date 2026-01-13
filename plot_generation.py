#!/usr/bin/env python3

import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ================= SETTINGS =================
INPUT_FILE = Path(
    "C:\\Users\\shruthi.devaraj\\Desktop\\output_files\\20220402_MSE_result_softearlycap_epsilon-30-3.txt"
)


WINDOW = 100
OUT_PREFIX = "softearlycap_epsilon"
# ===========================================


def parse_file(path: Path) -> pd.DataFrame:
    num = r"-?\d+(?:\.\d+)?"
    line_re = re.compile(
        rf"^episode\s+(\d+)\s+({num}|null)\s+({num}|null)\s+({num}|null)\s*$",
        re.IGNORECASE,
    )

    episodes, c2, c3, c4 = [], [], [], []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            s = raw.strip()
            m = line_re.match(s)
            if not m:
                continue

            ep = int(m.group(1))
            v2, v3, v4 = m.group(2), m.group(3), m.group(4)

            # Ignore rows containing null
            if v2.lower() == "null" or v3.lower() == "null" or v4.lower() == "null":
                continue

            try:
                episodes.append(ep)
                c2.append(float(v2))
                c3.append(float(v3))
                c4.append(float(v4))
            except ValueError:
                continue

    return pd.DataFrame(
        {
            "episode": episodes,
            "current_lp_cost": c2,
            "max_delay": c3,
            "episode_return": c4,
        }
    ).sort_values("episode").reset_index(drop=True)


def moving_avg(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def plot_pdf(x, y, ylabel, title, out_path: Path):
    plt.figure(figsize=(6, 4))   # journal-friendly size
    plt.plot(x, y, linewidth=1.5)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, format="pdf")  # VECTOR OUTPUT
    plt.close()


def main():
    if not INPUT_FILE.exists():
        raise SystemExit(f"File not found: {INPUT_FILE}")

    df = parse_file(INPUT_FILE)

    ma_df = pd.DataFrame(
        {
            "episode": df["episode"],
            "current_lp_cost_MA100": moving_avg(df["current_lp_cost"], WINDOW),
            "max_delay_MA100": moving_avg(df["max_delay"], WINDOW),
            "episode_return_MA100": moving_avg(df["episode_return"], WINDOW),
        }
    )

    # ---- PDF FIGURES ----
    plot_pdf(
        ma_df["episode"],
        ma_df["current_lp_cost_MA100"],
        "Current LP Cost (MA100)",
        "100-episode Moving Average — Current LP Cost",
        Path(f"{OUT_PREFIX}_current_lp_cost_MA{WINDOW}.pdf"),
    )

    plot_pdf(
        ma_df["episode"],
        ma_df["max_delay_MA100"],
        "Max Delay (MA100)",
        "100-episode Moving Average — Max Delay",
        Path(f"{OUT_PREFIX}_max_delay_MA{WINDOW}.pdf"),
    )

    plot_pdf(
        ma_df["episode"],
        ma_df["episode_return_MA100"],
        "Episode Return (MA100)",
        "100-episode Moving Average — Episode Return",
        Path(f"{OUT_PREFIX}_episode_return_MA{WINDOW}.pdf"),
    )

    print("Vector PDF figures saved:")
    print(f"  {OUT_PREFIX}_current_lp_cost_MA{WINDOW}.pdf")
    print(f"  {OUT_PREFIX}_max_delay_MA{WINDOW}.pdf")
    print(f"  {OUT_PREFIX}_episode_return_MA{WINDOW}.pdf")


if __name__ == "__main__":
    main()
