import pandas as pd
import numpy as np
import os
import random

# Read the file 'example_data/Simulation_data0.z' as a pandas DataFrame
file_path = os.path.join('example_data', 'Simulation_data0.z')
df = pd.read_csv(file_path, sep=None, engine='python', header=None)

no_annotations = 10
for i in range(no_annotations):
    df[len(df.columns)] = np.random.normal(loc=0, scale=5, size=len(df))

output_path = os.path.join('example_data', 'Simulation_data0.v_unrelated')
df.to_csv(output_path, sep='\t', header=False, index=False)
