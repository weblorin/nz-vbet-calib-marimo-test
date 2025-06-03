import polars as pl
import psycopg
import plotly.express as px

# Connect using the hardcoded service name
conn = psycopg.connect(service="NZCalibrationService")

# Query the database with explicit casting and NULL filtering
query = '''
SELECT 
  CAST(transect_points."TotDrainAreaSqKM" AS DOUBLE PRECISION) AS "TotDrainAreaSqKM",
  CAST("ValleyWidthm" AS DOUBLE PRECISION) AS "ValleyWidthm"
FROM transect_points
WHERE transect_points."TotDrainAreaSqKM" IS NOT NULL AND "ValleyWidthm" IS NOT NULL;
'''
df = pl.read_database(query, connection=conn)

# Close connection
conn.close()

# Create Plotly figure from Polars
fig = px.scatter(df, x="TotDrainAreaSqKM", y="ValleyWidthm", 
                 title="Drainage Area vs. Valley Width",
                 labels={"TotDrainAreaSqKM": "Total Drainage Area (sq km)", 
                         "ValleyWidthm": "Valley Width (m)"})
fig.show()
