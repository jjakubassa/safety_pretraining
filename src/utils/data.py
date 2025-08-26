import json
import pandas


def load_data(path: str) -> list[str]:
    if path.endswith(".txt"):
        with open(path, "r") as f:
            return f.readlines()
    elif path.endswith(".parquet"):
        df = pandas.read_parquet(path)
        data = df.get("text")
        if data is None:
            raise ValueError("No 'text' column found in parquet file")
        return data.tolist()
    elif path.endswith(".json"):
        with open(path, "r") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON data should be a list")
        return data
    else:
        raise ValueError("Unsupported file format")
