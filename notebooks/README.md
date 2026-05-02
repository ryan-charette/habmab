# Notebooks

The project is implemented as reusable modules first. These notebook slots are reserved for portfolio storytelling:

1. `01_habsos_data_exploration.ipynb`: inspect NOAA HABSOS fields, species coverage, and regional sampling density.
2. `02_sampling_simulation.ipynb`: step through one replay window and visualize selected samples.
3. `03_model_comparison.ipynb`: compare policies across budgets and bloom windows.
4. `04_final_figures.ipynb`: regenerate README/report figures.

The command-line demo already produces the final report without notebooks:

```powershell
python run_habmab.py --policies all --output reports/latest
```
