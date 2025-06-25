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
    from sklearn.linear_model import LogisticRegression  
    return LogisticRegression, mo, np, pl, psycopg, px


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
    WHERE "Discard" = false
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
    # Add binary valley column: 0 for 'hill', 1 for 'active', 'inactive', 'channel', null otherwise
    filtered_cb_df = filtered_cb_df.with_columns(
        pl.when(pl.col("category") == "hill")
          .then(0)
          .when(pl.col("category").is_in(["active", "inactive", "channel"]))
          .then(1)
          .otherwise(None)
          .alias("valley")
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
            y="valley",
            symbol="category",  # Add this line
            title=f"Drainage area {segment[0]} >= x < {segment[1]}",
            width=500,
            height=500
        )
        handplot = px.scatter(
            segment_df,
            x="HAND",
            y="valley",
            symbol="category",  # Add this line
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
def _(
    LogisticRegression,
    filtered_cb_df,
    go,
    mo,
    np,
    pl,
    segments_from_breaks,
):
    def _logistic_reg():
        segments = segments_from_breaks()
        results = []

        for segment in segments:
            segment_df = filtered_cb_df.filter(
                (pl.col('TotDrainAreaSqKM') >= segment[0]) &
                (pl.col('TotDrainAreaSqKM') < segment[1])
            ).select(['Slope', 'HAND', 'valley'])
            # Drop rows with missing values
            segment_df = segment_df.with_columns([
                pl.col("Slope").fill_nan(None),
                pl.col("HAND").fill_nan(None),
                pl.col("valley").fill_nan(None)
            ])
            segment_df = segment_df.drop_nulls()
            print (segment_df.shape)
            if segment_df.height == 0:
                results.append({
                    "segment": f"{segment[0]}-{segment[1]}",
                    "coef_Slope": None,
                    "coef_HAND": None,
                    "intercept": None,
                    "n_samples": 0,
                    "min_Slope": None,
                    "max_Slope": None,
                    "min_HAND": None,
                    "max_HAND": None,
                })
                continue
            X = np.stack([segment_df['Slope'].to_numpy(), segment_df['HAND'].to_numpy()], axis=1)
            y = segment_df['valley'].to_numpy()
            model = LogisticRegression()

            assert not np.isnan(X).any(), "X still contains NaN!"
            assert not np.isnan(y).any(), "y still contains NaN!"
            model.fit(X, y)
            results.append({
                "segment": f"{segment[0]}-{segment[1]}",
                "coef_Slope": model.coef_[0][0],
                "coef_HAND": model.coef_[0][1],
                "intercept": model.intercept_[0],
                "n_samples": len(y),
                "min_Slope": segment_df['Slope'].min(),
                "max_Slope": segment_df['Slope'].max(),
                "min_HAND": segment_df['HAND'].min(),
                "max_HAND": segment_df['HAND'].max()
            })
        return results


    results = _logistic_reg()

    def _lr_plots(results):
        figs = []
        """this is example for 1, but we want to do for all 3"""
        result = results[1]  
        slope_range = np.linspace(result["min_Slope"],result["max_Slope"],50)
        hand_range = np.linspace(result["min_HAND"],result["max_HAND"],50)
        intercept = result["intercept"]
        coef_slope = result["coef_Slope"]
        coef_hand = result["coef_HAND"]
        SLOPE, HAND = np.meshgrid(slope_range,hand_range)
        Z = 1 / (1 + np.exp(-(intercept + coef_slope * SLOPE + coef_hand * HAND)))
        modelfitfig =go.Figure(data=[go.Surface(x=SLOPE, y=HAND, z=Z,
                                               hovertemplate=
                "Slope: %{x:.2f}<br>HAND: %{y:.2f}<br>Probability: %{z:.3f}<extra></extra>")])
        modelfitfig.update_layout(
            title="Probability Surface: Valley=1",
            scene=dict(
            xaxis_title='Slope',
            yaxis_title='HAND',
            zaxis_title='Probability'
            ))
        return modelfitfig


    mo.vstack([
    mo.md(r"""
    The logistic regression formula is:

    $$
    p = \frac{1}{1 + \exp\left(-(\beta_0 + \beta_1 \cdot \text{Slope} + \beta_2 \cdot \text{HAND})\right)}
    $$
    """), 
    results,
    _lr_plots(results)
    ])
    return


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
    return (go,)


if __name__ == "__main__":
    app.run()
