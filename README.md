# Purpose

- to try Marimo as a platform for adhoc data analysis
- to review the VBET calibration data, specifically valley width vs total drainage area to find natural breakpoints for the curve fitting

# Using

To run the notebook:

```sh
uv run marimo run plot_pg_transectpoint_newmarimo.py
```

To edit, you can use `marimo edit` in place of `marimo run`, or use VS Code - Marimo extension is helpful.

## Requirements

- uv
- the postgres configuration on your system under name NZCalibrationService
