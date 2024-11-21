---
# Forcefield Validation Tool Documentation

## Overview

The **Forcefield Validation Tool** is designed to facilitate the validation of molecular dynamics (MD) force fields by performing parameter scans, running quantum mechanical (QM) and classical molecular mechanics (MM) calculations, and analyzing the results. It provides a streamlined workflow for calculating and comparing MM and QM PES energy profiles for the bonded interactions, generating key metrics, and visualizing the results.

---

## Features

- **Parameter Scanning**: Supports scanning of bond lengths and angles.
- **ORCA and GROMACS Integration**: Automates QM calculations using ORCA and MD simulations using GROMACS.
- **Flexible Workflow**:
  - Option to skip QM or MM calculations if results are already available.
  - Supports ABF (Adaptive Biasing Force), relaxed scan and constrained optimization methods.
- **Data Analysis and Plotting**: Compares MM and QM energy profiles, computes error metrics (e.g., MAE, RMSE, R²), and generates detailed plots.

---

## Dependencies

This tool requires the following software and Python libraries:

### Software
- [GROMACS](https://www.gromacs.org)
- [ORCA](https://orcaforum.cec.mpg.de/)

### Python Libraries
- `numpy`
- `matplotlib`
- `scipy`
- `tabulate`
- `jobdispatcher`

Install Python dependencies using:
```bash
pip install numpy matplotlib scipy tabulate jobdispatcher
```

---

## Usage

The tool is executed from the command line with various options to customize the workflow. Below are the key usage examples:

---

### Example Commands

#### Full Workflow (QM and MM Calculations with ABF)
```bash
python forcefield_validation_tool.py -i triagarose.xyz -t triagarose.top --total-cores 32 -tpc 8 --abf

Run the calculations, using 32 total cores and 8 cores for each calculation. Note: it will use 8 cores both for the ORCA and GROMACS calculations. If you wish to use a different number of cores depending on the type of calculation, use the --skip-XXX-calc flags to run the separately.
```

#### Skip ORCA Calculations
If QM results are already available, you can skip ORCA calculations:
```bash
python forcefield_validation_tool.py -i triagarose.xyz -t triagarose.top --skip-orca-calc --total-cores 32 -tpc 1 --abf
```

#### Skip Both QM and MM Calculations
If all calculations are completed, rerun only the data analysis and plotting:
```bash
python forcefield_validation_tool.py -i triagarose.xyz -t triagarose.top --skip-orca-calc --skip-gromacs-calc --total-cores 32 -tpc 1 --abf
```

---

## Command-Line Arguments
This section provides a detailed explanation of each argument and its intended use:

| **Argument**            | **Type**   | **Description**                                                                                      |
|-------------------------|------------|------------------------------------------------------------------------------------------------------|
| `-i`, `--input-file`    | `str`      | Input file containing the molecular structure in XYZ format. This file provides the atomic coordinates used to initiate calculations. It must be a valid XYZ file that defines the system's structure. Example: `triagarose.xyz`. |
| `-t`, `--topology`      | `str`      | Path to the topology file in GROMACS `.top` format. The topology must be self-contained with no external includes, defining molecular interactions, atom types, and bonded parameters. Example: `triagarose.top`. |
| `-tpc`, `--threads-per-calc` | `int`  | Number of threads to allocate per ORCA/GROMACS calculation. Specify how many threads each calculation can use to optimize parallel performance. Example: `-tpc 1` to use a single thread per calculation. |
| `--total-cores`         | `int`      | Total cores available for all calculations. Defines the computational resources across all running jobs. For example, if you want to run 32 jobs at once with `-tpc 1`, set `--total-cores 32`. |
| `--scan-steps`          | `int`      | Number of steps in the parameter scan (default: 8). Each step evaluates the system at different bond lengths or angles. Example: `--scan-steps 10` to divide the bond and angle scan range into 10 intervals. |
| `--abf`                 | `flag`     | Enable Adaptive Biasing Force (ABF) simulations PES generation. Performs much better then relaxed and constrained relaxation, but requires more computational time. Only compatible with GROMACS 2024.3 or newer. |
| `--range`               | `str`      | Defines the scan range for parameter exploration. Choose from: `"ff"` to focus the scan around the equilibrium value determined by the force field, or `"qm"` to use the range from QM calculations. Example: `--range qm`. |
| `--fixed-fit-eq-value`  | `flag`     | Uses a fixed equilibrium value during curve fitting taken from the MM force field.  |
| `--skip-orca-calc`      | `flag`     | Skips QM calculations using ORCA. Useful if ORCA calculations are already complete, saving time by focusing on subsequent MM steps or data analysis. No additional value required. Example: `--skip-orca-calc`. |
| `--skip-gromacs-calc`   | `flag`     | Skips MM calculations using GROMACS. Use this if MM calculations are complete, allowing the tool to rerun only data processing and plotting. No additional value required. Example: `--skip-gromacs-calc`. |
| `--include-hydrogens`   | `flag`     | Includes bonds and angles involving hydrogen atoms in the scan. By default, hydrogen atoms are excluded. Use this flag to include them in calculations. No additional value required. Example: `--include-hydrogens`. |
| `--constrained-opt`     | `flag`     | Performs constrained optimization during parameter scanning. Constrains specific interactions during MM simulations to explore geometry relaxation effects. No additional value required. Example: `--constrained-opt`. |
| `--charge`              | `int`      | Specifies the molecular charge for QM calculations. Example: `--charge 0` for neutral systems, `--charge -1` for anions, or `--charge 1` for cations. |
| `--multiplicity`        | `int`      | Specifies the spin multiplicity for QM calculations. Example: `--multiplicity 1` for singlets, `--multiplicity 2` for doublets, or higher values for other spin states. |

---

### Additional Notes

- **Optional Flags**: Many flags are optional and only need to be specified if the default behavior is not suitable.
- **Combining Flags**: Flags like `--skip-orca-calc` and `--skip-gromacs-calc` can be used together to bypass both QM and MM calculations, focusing solely on data analysis.
- **Resource Management**: Ensure `--total-cores` and `-tpc` values align with the available hardware to avoid overloading the system.

For more examples, refer to the **Usage** section above.

---

## Outputs

- **Data Files**:
  - `energy.xvg`: Extracted MM energy values.
  - `parameter_mm_data.csv` and `parameter_qm_data.csv`: MM and QM data in CSV format.
  
- **Plots**:
  - Energy comparison plots (`parameter_energy_plot.png`) for each parameter.

- **Metrics Table**:
  Printed to the console, includes:
  - `MM K` (kcal/mol/Å²): MM force constant.
  - `QM K` (kcal/mol/Å²): QM force constant.
  - `Cumulative Error`: Total energy difference.
  - `MAE`: Mean Absolute Error.
  - `RMSE`: Root Mean Square Error.
  - `R²`: Coefficient of determination.

---

## Limitations and Notes

- **Hydrogen Handling**: By default, bonds and angles involving hydrogen atoms are excluded unless `--include-hydrogens` is specified.
- **Performance**: Ensure sufficient computational resources when using many cores.
- **Customization**: Modify default settings (e.g., QM method or GROMACS MDP options) by editing the respective functions inside the script (advanced)

---

## Contact

For issues or suggestions, please contact the tool's developer.

## Future plans
It is foreseen to convert the current script into a full blown python command line tool. 