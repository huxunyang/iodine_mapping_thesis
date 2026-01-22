import pandas as pd

def save_df(df: pd.DataFrame, path):
    path.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(path, index=False)
