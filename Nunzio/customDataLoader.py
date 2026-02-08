import torch
import numpy as np
import pandas as pd
import os


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, originalDataName:str, loadDataPath:str):
        try:
            if "WADI" not in originalDataName:
                raise ValueError("Not WADI dataset, skipping as per configuration.")
            int(originalDataName.strip(".csv").split("_")[-1].strip())
            gtFile = "_".join(originalDataName.strip(".csv").split("_")[:-1]) + ".csv"
        except ValueError:
            gtFile = originalDataName


        self.matrix = np.load(os.path.join(loadDataPath, "transition_matrices", originalDataName.strip(".csv") + "_transition_matrix.npy"), allow_pickle=True)
        self.groundTruth:np.ndarray = pd.read_csv(os.path.join(loadDataPath,"ground_truth_labels", gtFile.strip(".csv") + "_ground_truth_labels.csv"), index_col=None, header=None).values.T[0]
        self.indexes: pd.DataFrame = pd.read_csv(os.path.join(loadDataPath, "predictions", originalDataName.strip(".csv") + "_predictions.csv")).iloc[:, :2]
        self.indexes.columns = ['item_id', 'index_id']
        self.data = pd.read_csv(os.path.join(loadDataPath, "continuous_scores", originalDataName.strip(".csv") + "_scores.csv"), header=0, index_col=None)
        
        self.groupedIndex = {item: list(self.indexes.loc[self.indexes['item_id'] == item, 'index_id'].values) for item in self.indexes['item_id'].unique().tolist()}
        self.groupedIndexByIndex = {i: self.indexes[self.indexes['item_id'] == i].index.tolist() for i in self.indexes['item_id'].unique().tolist()}

        self.randomMapping = np.random.permutation(list(self.groupedIndex.keys()))
        self.minIndex = min(self.groupedIndex.keys())

    def __len__(self):
        return len(self.randomMapping)


    def __getitem__(self, idx):
        idx = int(self.randomMapping[idx]) 
        return torch.tensor(self.data.loc[self.groupedIndexByIndex[idx], :].values).float(), torch.tensor(self.matrix[idx - self.minIndex, ...]).float(), torch.tensor(self.groundTruth[self.groupedIndex[idx]]).float().unsqueeze(-1)


if __name__ == "__main__":
    for file in os.listdir("./TEST_SPLIT"):
        try:
            dataset = CustomDataset(file, "./PROCESSED_TRAIN_DATA")
            dataLoader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
        except ValueError as e:
            print(f"Error processing file {file}: {e}")

            for cont, mat, gt in dataLoader:
                assert (mat[0].sum(dim=1)/mat[0].sum(dim=1)  == 1).all(), "Transition matrix rows do not sum to 1"