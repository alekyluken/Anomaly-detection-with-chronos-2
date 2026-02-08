import torch
import numpy as np
import pandas as pd
import os


class CustomDataset(torch.utils.data.Dataset):
    """
    Dataset per caricare previsioni quantili Chronos-2 in formato wide.

    Formato CSV wide:
        v1_item_id, v1_timestep, v1_q1, ..., v1_qQ, v2_item_id, v2_timestep, ...
    
    Ogni variabile vi ha colonne metadata (item_id, timestep) + Q colonne quantile.
    item_id identifica la finestra temporale di predizione.

    Ogni sample restituisce:
        quantiles: [D, T, Q]  – previsioni quantili per D variabili
        values:    [D, T]     – valori reali osservati
        labels:    [T, 1]     – etichette anomalia ground truth
    
    Compatibile con PROCESSED_TRAIN_DATA (Q=3) e PROCESSED_TRAIN_DATAV2 (Q=21).
    """
    def __init__(self, originalDataName: str, loadDataPath: str, originalPath: str, skipNames: set[str] = {"WADI"}, context_length: int = 100):
        if any(name in originalDataName.upper() for name in skipNames):
            raise ValueError(f"Skipping dataset {originalDataName} as it is in skipNames.")

        baseName = originalDataName.rsplit('.', 1)[0]

        # ── Load predictions (wide format) ──
        self.data = pd.read_csv(os.path.join(loadDataPath, "predictions", baseName + "_predictions.csv"),header=0, index_col=None)

        # ── Load ground truth labels (aligned row-by-row with predictions) ──
        self.gt = pd.read_csv(os.path.join(loadDataPath, "ground_truth_labels", baseName + "_ground_truth_labels.csv"), index_col=None).iloc[:, 0].values.astype(float)

        # ── Load original time series ──
        self.originalTS = pd.read_csv(os.path.join(originalPath, originalDataName),header=0, index_col=None)

        # ── Parse column structure (D variabili, Q quantili) ──
        self._parse_columns()

        # Remove label / extra columns from original TS
        if self.originalTS.shape[1] > self.num_variables:
            self.originalTS = self.originalTS.iloc[:, :self.num_variables]

        # Convert to numpy for fast indexing
        self._original_values = self.originalTS.values   # [total_T, D]
        self._data_values = self.data.values              # [total_rows, total_cols]

        # ── Group rows by item_id (finestra temporale) ──
        item_col = self.data.columns[self.item_id_col_idx]
        self.indexes: dict[int, list[int]] = {}
        for iid in self.data[item_col].unique():
            self.indexes[int(iid)] = self.data.index[self.data[item_col] == iid].tolist()

        self.item_ids = sorted(self.indexes.keys())
        self.randomMapping = np.random.permutation(len(self.item_ids))
        self.context_length = context_length

    # ─────────────────────────────────────────────────────────────
    def _parse_columns(self):
        """Rileva automaticamente D variabili e Q livelli quantile dalle colonne."""
        # Trova prefissi variabili unici (v1, v2, ...)
        prefixes: list[str] = []
        seen: set[str] = set()
        for c in self.data.columns.tolist():
            parts = c.split('-', 1)
            if len(parts) == 2 and parts[0] not in seen:
                seen.add(parts[0])
                prefixes.append(parts[0])
        self.num_variables = len(prefixes)

        # Per-variable: trova indici colonne quantile e colonne metadata
        self.quantile_col_indices: list[list[int]] = []
        self.quantile_levels: list[float] = []
        self.item_id_col_idx: int = 0
        self.timestep_col_indices: list[int] = []

        for i, prefix in enumerate(prefixes):
            q_indices: list[int] = []
            q_lvls: list[float] = []
            for j, col in enumerate(self.data.columns.tolist()):
                if not col.startswith(prefix + '-'):
                    continue
                suffix = col[len(prefix) + 1:]
                if suffix == 'item_id':
                    if i == 0:
                        self.item_id_col_idx = j
                elif suffix == 'timestep':
                    self.timestep_col_indices.append(j)
                else:
                    try:
                        lvl = float(suffix)
                        q_indices.append(j)
                        q_lvls.append(lvl)
                    except ValueError:
                        pass  # 'predictions' or other non-quantile columns

            self.quantile_col_indices.append(q_indices)
            if not self.quantile_levels and q_lvls:
                self.quantile_levels = q_lvls

        self.num_quantiles = len(self.quantile_levels)

    # ─────────────────────────────────────────────────────────────
    def __len__(self):
        return len(self.item_ids)

    def __getitem__(self, idx):
        row_indices = self.indexes[self.item_ids[int(self.randomMapping[idx])]]

        raw = self._data_values[row_indices]  # [T, total_cols]

        # ── Timesteps → indici nella serie originale ──
        if self.timestep_col_indices:
            timesteps = raw[:, self.timestep_col_indices[0]].astype(int)
        else:
            # Fallback se non c'è colonna timestep
            timesteps = np.arange(len(row_indices))


        # ── Labels: [T, 1]  (allineato riga-per-riga con predictions) ──
        return (
            torch.tensor(np.stack([raw[:, qi] for qi in self.quantile_col_indices], axis=0), dtype=torch.float32),              # [D, T, Q]
            torch.tensor(self._original_values[np.clip(timesteps, 0, len(self._original_values) - 1)].T, dtype=torch.float32),                 # [D, T]
            torch.tensor(self.gt[row_indices], dtype=torch.float32).unsqueeze(-1)    # [T, 1]
        )


if __name__ == "__main__":
    # Test con entrambi i formati (V1 e V2)
    for processed_dir, label in [("./PROCESSED_TRAIN_DATAV2", "V2"), ("./PROCESSED_TRAIN_DATA", "V1")]:
        if not os.path.isdir(processed_dir):
            print(f"  [{label}] Directory {processed_dir} non trovata, skip")
            continue
        print(f"\n{'='*50}")
        print(f"Testing formato {label}: {processed_dir}")
        print('='*50)
        for file in sorted(os.listdir("./TEST_SPLIT")):
            try:
                dataset = CustomDataset(file, processed_dir, "./TEST_SPLIT", skipNames={"SKAB", "KAGGLE"})
                loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
                quantiles, values, labels = next(iter(loader))
                print(f"  {file}:")
                print(f"    D={dataset.num_variables}, Q={dataset.num_quantiles}, items={len(dataset)}")
                print(f"    quantiles: {quantiles.shape}, values: {values.shape}, labels: {labels.shape}")
            except ValueError as e:
                print(f"  {file}: skipped ({e})")
            except Exception as e:
                print(f"  {file}: ERROR {e}")