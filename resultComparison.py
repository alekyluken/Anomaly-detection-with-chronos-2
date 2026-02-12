import os, argparse
import numpy as np
import pandas as pd
import warnings

from IPython.display import display
from json import load as json_load

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ANSI escape codes per evidenziare
GREEN = "\033[92m"
RESET = "\033[0m"

def printPerClassResults(results: dict[str, dict[str, list[float]]], file: str):
    print(f"PER-CLASS RESULTS FOR {file}:")
    classes = sorted(set(run.split("_")[1].upper() for run in results.keys()))
    
    for cls in classes:
        print(f"\nClass: {cls} ({len([run for run in results.keys() if run.split('_')[1].upper() == cls])} runs)")
        for metric in results[list(results.keys())[0]]['metrics'][0].keys():
            values = [results[run]['metrics'][0][metric] for run in results.keys() if run.split("_")[1].upper() == cls and isinstance(results[run]['metrics'][0][metric], float)]
            if values:
                print(f"  {metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")


def computeFileStats(file: str, restrictToClasses:bool=False, printAllResults:bool=False) -> dict[str, float]:
    if os.path.exists(file):
        with open(file, "r", encoding="utf-8") as f:
            results = json_load(f)

        if restrictToClasses:
            if isinstance(restrictToClasses, str):
                restrictToClasses = restrictToClasses.strip().split(",")
            if isinstance(restrictToClasses, (list, tuple, set)):
                restrictToClasses = set(map(lambda x: x.strip().upper(), restrictToClasses))
            results = dict(filter(lambda x: x[0].split("_")[1].upper() in restrictToClasses, results.items()))

            print(f"Restricted to classes: {restrictToClasses}, found {len(results)} runs.")
        else:
            print(f"Processing all classes, found {len(results)} runs.")

        if printAllResults:
            printPerClassResults(results, file)

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
    parser.add_argument("--files",type=str,nargs="+",required=True,help="List of result files to compare.")
    parser.add_argument("--restrictTo", type=str, nargs="*", default=False, help="If a name or more than one class is provided (comma-separated), restrict the comparison to those classes only.")
    parser.add_argument("--printAllResults", action="store_true", help="Print all per-class results for each file.")

    args = parser.parse_args()
    display(build_dataframe({f: computeFileStats(f, restrictToClasses=args.restrictTo, printAllResults=args.printAllResults) for f in args.files}))