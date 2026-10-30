from pathlib import Path
import pandas as pd
D=Path(__file__).resolve().parents[1]/"figure_data/controlled_identity"
for name in ["near_group_metrics.csv","near_group_representation_intervals.csv","near_group_training_size_intervals.csv"]:
 print(name)
 print(pd.read_csv(D/name).to_string(index=False))
