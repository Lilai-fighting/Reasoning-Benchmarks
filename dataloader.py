import os
import json
import csv
from torch.utils.data import Dataset


class GenericDataset(Dataset):
    def __init__(self, file_paths):
        self.data = []

        for path in file_paths:
            ext = os.path.splitext(path)[1].lower()

            if ext == ".json":
                self.data.extend(self._load_json(path))

            elif ext == ".csv":
                self.data.extend(self._load_csv(path))

            else:
                raise ValueError(f"Unsupported format: {ext}")

    def _load_json(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]

    def _load_csv(self, path):
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return [row for row in reader]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # normalize format here
        return self._normalize(sample)

    def _normalize(self, sample):
        pass
