import argparse
from sklearn.metrics import average_precision_score
from json import load as json_load


def perClassAnalysis(results: dict):
    for cls in sorted(set(map(lambda x: x.split("_")[1].strip(), results.keys()))):
        confMat = [[0, 0], [0, 0]]
        pred, gt = [], []
        for res_key, res in results.items():
            if cls in res_key:
                confMat[0][0] += res["confusion_matrix"][0][0]
                confMat[0][1] += res["confusion_matrix"][0][1]
                confMat[1][0] += res["confusion_matrix"][1][0]
                confMat[1][1] += res["confusion_matrix"][1][1]
                pred.extend(res["pred"])
                gt.extend(res["gt"])

        print(f"Results for class: {cls}")
        print(f"True Negatives: {confMat[0][0]}")
        print(f"False Positives: {confMat[0][1]}")
        print(f"False Negatives: {confMat[1][0]}")
        print(f"True Positives: {confMat[1][1]}")
        print(confMat)
        print(f"Accuracy: {(confMat[0][0] + confMat[1][1]) / sum(map(sum, confMat)):.4f}")
        print(f"Precision: {confMat[1][1] / (confMat[1][1] + confMat[0][1]):.4f}")
        print(f"Recall: {confMat[1][1] / (confMat[1][1] + confMat[1][0]):.4f}")
        print(f"F1 Score: {2 * confMat[1][1] / (2 * confMat[1][1] + confMat[0][1] + confMat[1][0]):.4f}")
        print(f"AUC-PR: {average_precision_score(gt, pred):.4f}")
        print("-" * 50)


def totalStats(results: dict):
    confMat = [[0, 0], [0, 0]]
    pred, gt = [], []
    for res in results.values():
        confMat[0][0] += res["confusion_matrix"][0][0]
        confMat[0][1] += res["confusion_matrix"][0][1]
        confMat[1][0] += res["confusion_matrix"][1][0]
        confMat[1][1] += res["confusion_matrix"][1][1]
        pred.extend(res["pred"])
        gt.extend(res["gt"])

    print(f"True Negatives: {confMat[0][0]}")
    print(f"False Positives: {confMat[0][1]}")
    print(f"False Negatives: {confMat[1][0]}")
    print(f"True Positives: {confMat[1][1]}")
    print(confMat)
    print(f"Accuracy: {(confMat[0][0] + confMat[1][1]) / sum(map(sum, confMat)):.4f}")
    print(f"Precision: {confMat[1][1] / (confMat[1][1] + confMat[0][1]):.4f}")
    print(f"Recall: {confMat[1][1] / (confMat[1][1] + confMat[1][0]):.4f}")
    print(f"F1 Score: {2 * confMat[1][1] / (2 * confMat[1][1] + confMat[0][1] + confMat[1][0]):.4f}")
    print(f"AUC-PR: {average_precision_score(gt, pred):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate anomaly detection results.")
    parser.add_argument("--file", type=str, help="Path to the JSON file containing the results.")

    with open(parser.parse_args().file, "r") as f:
        results = json_load(f)

    perClassAnalysis(results)
    print("\nOverall Statistics:")
    totalStats(results)