import os, argparse
import numpy as np
import pandas as pd

from IPython.display import display
from json import load as json_load

# ANSI escape codes per evidenziare
GREEN = "\033[92m"
RESET = "\033[0m"


def computeFileStats(file: str) -> dict[str, float]:
    if os.path.exists(file):
        with open(file, "r", encoding="utf-8") as f:
            results = json_load(f)

        # results = dict(filter(lambda x: x[0].split("_")[1]=="MSL", results.items()))

        return {
            key: [
                np.mean([results[run]['metrics'][0][key] for run in results.keys() if isinstance(results[run]['metrics'][0][key], float)]),
                np.std([results[run]['metrics'][0][key] for run in results.keys()  if isinstance(results[run]['metrics'][0][key], float)])
            ]
            for key in results[list(results.keys())[0]]['metrics'][0].keys()
        }


def build_dataframe(all_stats: dict[str, dict[str, list[float]]]) -> pd.DataFrame:
    files = list(all_stats.keys())
    metrics = list(filter(lambda x: not np.isnan(all_stats[files[0]][x][0]), all_stats[files[0]].keys()))

    data = {}

    best_values = {metric: max(all_stats[f][metric][0] for f in files) for metric in metrics}

    for f in files:
        short_name = '\\'.join(f.split(os.sep)[2:])
        data[short_name] = {}

        for metric in metrics:
            mean, std = all_stats[f][metric]
            cell = f"{mean:.4f} ± {std:.4f}"

            # Evidenzia il migliore
            if mean == best_values[metric]:
                cell = f"{GREEN}{cell}{RESET}"

            data[short_name][metric] = cell

    df = pd.DataFrame(data)
    df.index.name = "Metric"
    return df



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare results from multiple result files.")
    parser.add_argument(
        "--files",
        type=str,
        nargs="+",
        required=True,
        help="List of result files to compare.",
    )
    
    display(build_dataframe({f: computeFileStats(f) for f in parser.parse_args().files}))
