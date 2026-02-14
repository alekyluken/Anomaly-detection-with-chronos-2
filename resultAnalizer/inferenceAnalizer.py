import json, argparse


args = argparse.ArgumentParser(description="Evaluate inference benchmarks.")
args.add_argument("--file", type=str, help="Path to the JSON file containing the results.")


with open(args.parse_args().file, "r") as f:
    mora = json.load(f)

for key in mora.keys():
    if key in mora:
        print(key)
        print(f"{mora[key]['median']:.6f} pm {mora[key]['std_dev']:.6f}")