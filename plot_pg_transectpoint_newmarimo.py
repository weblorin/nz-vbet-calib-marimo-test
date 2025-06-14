import marimo

__generated_with = "0.13.15"
app = marimo.App(width="full", app_title="VBET Calib")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import psycopg
    import plotly.express as px
    import numpy as np
    return mo, np, pl, psycopg, px


@app.cell
def _(mo):
    mo.md(r"""# Load data from Postgres""")
    return


@app.cell
def _(pl, psycopg):
    # Connect using the hardcoded service name
    conn = psycopg.connect(service="NZCalibrationService")

    # Query the database for valley widths vs drainage area, with NULL filtering
    query = '''
    SELECT 
      level_path,
      transect_points."TotDrainAreaSqKM" AS "TotDrainAreaSqKM",
      "ValleyWidthm" AS "Width_Lorin",
      "ValleyWidthm_frompoints"  AS "Width_Points"
    FROM transect_points
    WHERE level_path IS NOT NULL AND (
    "ValleyWidthm" IS NOT NULL OR "ValleyWidthm_frompoints" IS NOT NULL
    );
    '''
    df = pl.read_database(query, connection=conn, infer_schema_length=1000)

    # Ensure both columns are Float64, even if all nulls
    df = df.with_columns([
        pl.col("Width_Lorin").cast(pl.Float64),
        pl.col("Width_Points").cast(pl.Float64)
    ])

    # Query for calibration points
    query = '''
    SELECT cp.fid, cp.category, cp.transect_id, cp."HAND", cp."Slope", tp."TotDrainAreaSqKM" FROM calibration_points cp
    LEFT JOIN transect_points tp ON tp."TransectId" = cp.transect_id 
    '''
    cb_df = pl.read_database(query, connection=conn, infer_schema_length=1000)

    # Add level_path column by extracting prefix from transect_id
    cb_df = cb_df.with_columns(
        pl.col("transect_id").str.split("-").list.get(0).alias("level_path")
    )

    # Close connection
    conn.close()

    # Unpivot the DataFrame to long format for both ValleyWidthm and ValleyWidthmfrompoints
    long_df = df.unpivot(
        index=["TotDrainAreaSqKM", "level_path"],
        on=["Width_Lorin", "Width_Points"],
        variable_name="MeasuredBy",
        value_name="ValleyWidth"
    )
    return cb_df, long_df


@app.cell
def _(cb_df, long_df, mo):
    mo.accordion({"Transects with drainage and width":long_df,
                  "Calibration points with slope, hand and drainage": cb_df
                 })
    return


@app.cell
def _(mo):
    mo.md(r"""# Identifying valley bins by catchment area""")
    return


@app.cell(hide_code=True)
def _(long_df, mo):
    level_paths = sorted(long_df['level_path'].unique().to_list())
    measured_by_options = sorted(long_df['MeasuredBy'].unique().to_list())

    level_path_select = mo.ui.dropdown(
        options=["All"] + level_paths,
        label="Level Path",
        value="All"
    )
    measured_by_select = mo.ui.dropdown(
        options=["All"] + measured_by_options,
        label="Width Type",
        value="All"
    )

    return (level_path_select,)


@app.cell
def _(level_path_select):
    level_path_select
    return


@app.cell
def _(level_path_select, long_df, pl, px):
    # Filter the DataFrame based on the selected level_path
    if level_path_select.value == "All":
        filtered_df = long_df
    else:
        filtered_df = long_df.filter(pl.col("level_path") == level_path_select.value)

    fig = px.scatter(
        filtered_df,
        x="TotDrainAreaSqKM",
        y="ValleyWidth",
        color="level_path",
        symbol="MeasuredBy",
        title=f"Drainage Area vs. Valley Width{'' if level_path_select.value == 'All' else f' for {level_path_select.value}'}",
        labels={
            "TotDrainAreaSqKM": "Total Drainage Area (sq km)",
            "ValleyWidth": "Valley Width (m)",
            "MeasuredBy": "Width Type"
        }
    )

    fig.update_xaxes(range=[0, None])
    fig.update_yaxes(range=[0, None])
    fig
    return


@app.cell
def _(mo):
    mo.md(r"""# Curve fitting""")
    return


@app.cell
def _(mo):
    lineparam_m = mo.ui.number(value=-1.144, step=0.001, label="fit m")
    lineparam_b = mo.ui.number(value=8.199, step=0.001, label="fit b")
    # to do - not all hardcoded :-)
    # Sq Metres. 0 and max() are the other boundaries, So 1 break point creates 2 bins; 2 creates 3; 
    drainage_size_breaks = [25,1000] 
    # import sys
    def segments_from_breaks (): 
        # size_breaks:list[float],min:float=0,max:float=sys.float_info.max
        # sort the size_breaks 
        # check that min < smallest size; max > largest size; remove any duplicates
        # for i in size_breaks:
        # return >= lower bound and < upper bound
        return [(0,25),(25,1000),(1000,9999999)]
    mo.md(r"""## Parameters""")
    return lineparam_b, lineparam_m, segments_from_breaks


@app.cell
def _(cb_df, level_path_select, mo, pl, px, segments_from_breaks):
    # Filter the DataFrame based on the selected level_path
    if level_path_select.value == "All":
        filtered_cb_df = cb_df
    else:
        filtered_cb_df = cb_df.filter(
            pl.col("level_path") == level_path_select.value
        )
    # Add category_bin column: 0 for 'hill', 1 for 'active', 'inactive', 'channel', null otherwise
    filtered_cb_df = filtered_cb_df.with_columns(
        pl.when(pl.col("category") == "hill")
          .then(0)
          .when(pl.col("category").is_in(["active", "inactive", "channel"]))
          .then(1)
          .otherwise(None)
          .alias("category_bin")
    )

    segments = segments_from_breaks()

    slopeplots = []
    handplots = []
    for segment in segments:
        segment_df = filtered_cb_df.filter(
            (pl.col('TotDrainAreaSqKM')>=segment[0]) & 
            (pl.col('TotDrainAreaSqKM')<segment[1])
        )
        slopeplot = px.scatter(
            segment_df,
            x="Slope",
            y="category_bin",
            title=f"Drainage area {segment[0]} >= x < {segment[1]}",
            width=500,
            height=500
        )
        handplot = px.scatter(
            segment_df,
            x="HAND",
            y="category_bin",
            title=f"Drainage area {segment[0]} >= x < {segment[1]}",
            width=500,
            height=500
        )
        slopeplots.append(slopeplot)
        handplots.append(handplot)

    mo.vstack([
        mo.hstack(slopeplots, wrap=True),
        mo.hstack(handplots, wrap=True)
        ])
    return (filtered_cb_df,)


@app.cell
def _(mo):
    mo.md(r"""# Old stuff - just for examples""")
    return


@app.cell
def _(
    filtered_cb_df,
    level_path_select,
    lineparam_b,
    lineparam_m,
    mo,
    np,
    pl,
    px,
):
    # OLD STUFF - JUST KEEPING FOR EXAMPLES
    # Add theoretical line: Slope = m * ln(DrainageArea) + b using polars
    if len(filtered_cb_df) > 0:
        x_vals = filtered_cb_df['TotDrainAreaSqKM'].to_numpy()
        x_vals = x_vals[~np.isnan(x_vals)]  # Remove nan values
        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
        y_line = lineparam_m.value * np.log(x_line) + lineparam_b.value
        line_df = pl.DataFrame({
            'TotDrainAreaSqKM': x_line,
            'Slope': y_line
        })
    else:
        line_df = pl.DataFrame({'TotDrainAreaSqKM': [], 'Slope': []})

    logplot_da_slope = px.scatter(
        filtered_cb_df,
        x="TotDrainAreaSqKM",
        y="Slope",
        color="category",
        symbol="category",
        title=f"Drainage Area vs. Slope{'' if level_path_select.value == 'All' else f' for {level_path_select.value}'}",
        labels={
            "TotDrainAreaSqKM": "Total Drainage Area (sq km) at transect point",
            "Slope": "Slope at this calib point"
        }
    )
    # Set x-axis to log scale
    logplot_da_slope.update_xaxes(type="log")
    # Add the line
    if len(line_df) > 0:
        import plotly.graph_objects as go
        logplot_da_slope.add_traces(go.Scatter(
            x=line_df['TotDrainAreaSqKM'].to_numpy(),
            y=line_df['Slope'].to_numpy(),
            mode='lines',
            name=f'Slope = {lineparam_m.value} * ln(DrainageArea) + {lineparam_b.value}',
            line=dict(color='black', dash='dash')
        ))

    mo.vstack([logplot_da_slope,mo.hstack(items=[lineparam_m, lineparam_b])])
    return


if __name__ == "__main__":
    app.run()
