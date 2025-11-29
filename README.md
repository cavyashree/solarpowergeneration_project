✅ STEP 1 — Put all project files in one folder

Example folder:

C:\Users\dolas\Desktop\kavya project\


Inside this folder, keep:

solarpower.csv

solarpowergeneration.ipynb

Any images or extra scripts

✅ STEP 2 — Create a Virtual Environment (VERY IMPORTANT)

Open VS Code → Terminal → New Terminal
Then run:

python -m venv .venv


Activate it:

If you use PowerShell:
.\.venv\Scripts\Activate.ps1

If you use CMD:
.\.venv\Scripts\activate.bat


You should now see:

(.venv) PS C:\Users\dolas\Desktop\kavya project>

✅ STEP 3 — Install all required modules

While the venv is activated, run:

pip install pandas numpy matplotlib seaborn statsmodels scikit-learn openpyxl jupyterlab ipykernel


This installs everything your notebook needs.

✅ STEP 4 — Register your venv as a Jupyter Kernel

Run:

python -m ipykernel install --user --name solar-power --display-name "Solar Power Kernel"


Now Jupyter/VS Code will use your environment.

✅ STEP 5 — Open your notebook in VS Code

Open VS Code

Open the folder kavya project

Click solarpowergeneration.ipynb

VS Code will ask: Select Kernel
→ Choose Solar Power Kernel (the one you created)

This ensures no more ModuleNotFoundError.

✅ STEP 6 — Test your installation

In a cell, run:

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

print("All modules working!")


If no errors → all good 

✅ STEP 7 — Load your dataset

Run:

df = pd.read_csv(r"C:\Users\dolas\Desktop\kavya project\solarpower.csv")
df.head()


Or if an Excel file:

df = pd.read_excel(r"C:\Users\dolas\Desktop\kavya project\solarpower.xlsx")

✅ STEP 8 — Run the remaining notebook cells one by one

Click Run All or run cells manually:

Data cleaning

Visualization

Time-series decomposition

Forecasting

Model evaluation

Output graphs
