from pyspark.sql import SparkSession

# Create a SparkSession
spark = SparkSession.builder \
    .appName("SimpleSparkApp") \
    .getOrCreate()

# Create a simple DataFrame
data = [("Alice", 25), ("Bob", 30), ("Charlie", 35)]
columns = ["Name", "Age"]

df = spark.createDataFrame(data, columns)

# Display the DataFrame
df.show()

# Perform basic operations
df.filter(df.Age > 26).show()

# Stop the SparkSession
spark.stop()