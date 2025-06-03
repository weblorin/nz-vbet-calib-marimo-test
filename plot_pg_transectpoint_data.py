import polars as pl
import psycopg
import plotly.express as px

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
WHERE level_path IS NOT NULL;
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

# Create Plotly figure from Polars
fig = px.scatter(
    long_df,
    x="TotDrainAreaSqKM",
    y="ValleyWidth",
    color="MeasuredBy",
    symbol="MeasuredBy",
    facet_col="level_path",  # Facet by level_path
    facet_col_wrap=4,
    title="Drainage Area vs. Valley Width by Level Path",
    labels={
        "TotDrainAreaSqKM": "Total Drainage Area (sq km)",
        "ValleyWidth": "Valley Width (m)",
        "level_path": "Level Path",
        "WidthType": "Width Type"
    }
)

# Set min x and y to zero for all facets, but let Plotly choose the max
fig.update_xaxes(range=[0, None])
fig.update_yaxes(range=[0, None])

fig.show()
