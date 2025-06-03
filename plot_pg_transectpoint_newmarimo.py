import marimo

__generated_with = "0.13.15"
app = marimo.App(app_title="VBET Calib")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import psycopg
    import plotly.express as px
    return mo, pl, psycopg, px


@app.cell
def _(pl, psycopg):
    # Connect using the hardcoded service name
    conn = psycopg.connect(service="NZCalibrationService")

    # Query the database with explicit casting and NULL filtering
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

    # Close connection
    conn.close()

    # Unpivot the DataFrame to long format for both ValleyWidthm and ValleyWidthmfrompoints
    long_df = df.unpivot(
        index=["TotDrainAreaSqKM", "level_path"],
        on=["Width_Lorin", "Width_Points"],
        variable_name="MeasuredBy",
        value_name="ValleyWidth"
    )
    return (long_df,)


@app.cell
def _(long_df):
    long_df
    return


@app.cell
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
        color="MeasuredBy",
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
def _():
    # fig = px.scatter(
    #     long_df,
    #     x="TotDrainAreaSqKM",
    #     y="ValleyWidth",
    #     color="MeasuredBy",
    #     symbol="MeasuredBy",
    #     facet_col="level_path",  # Facet by level_path
    #     facet_col_wrap=4,
    #     title="Drainage Area vs. Valley Width by Level Path",
    #     labels={
    #         "TotDrainAreaSqKM": "Total Drainage Area (sq km)",
    #         "ValleyWidth": "Valley Width (m)",
    #         "level_path": "Level Path",
    #         "WidthType": "Width Type"
    #     }
    # )

    # # Set min x and y to zero for all facets, but let Plotly choose the max
    # fig.update_xaxes(range=[0, None])
    # fig.update_yaxes(range=[0, None])

    return


if __name__ == "__main__":
    app.run()
