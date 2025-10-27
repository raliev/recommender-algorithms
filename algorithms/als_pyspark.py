# algorithms/als_pyspark.py
import numpy as np
import pandas as pd
from .base import Recommender
try:
    from pyspark.sql import SparkSession
    from pyspark.ml.recommendation import ALS
    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False

class ALSPySparkRecommender(Recommender):
    def __init__(self, k, iterations=10, lambda_reg=0.1, **kwargs):
        super().__init__(k)
        if not PYSPARK_AVAILABLE:
            raise ImportError(
                "PySpark is not installed. Please run 'pip install pyspark' to use this recommender."
            )

        self.name = "ALS (PySpark)"
        self.rank = k
        self.maxIter = iterations
        self.regParam = lambda_reg
        self.spark_model = None
        self.original_shape = None
        self.user_ids_in_train = None
        self.item_ids_in_train = None

        self.spark = SparkSession.builder.appName("StreamlitALS") \
            .config('spark.driver.memory', '2g') \
            .config('spark.executor.memory', '2g') \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .getOrCreate()
        # Set log level to WARN to reduce verbosity
        self.spark.sparkContext.setLogLevel("WARN")

    def __del__(self):
        """Destructor to stop the Spark session when the object is destroyed."""
        if hasattr(self, 'spark'):
            self.spark.stop()

    def fit(self, R, progress_callback=None, visualizer = None):
        self.original_shape = R.shape

        # 1. Convert NumPy matrix to Spark DataFrame format
        user_ids, item_ids = R.nonzero()
        ratings = R[user_ids, item_ids]

        # Store the users and items that are actually in the training data
        self.user_ids_in_train = np.unique(user_ids)
        self.item_ids_in_train = np.unique(item_ids)

        pd_df = pd.DataFrame({
            'user': user_ids.astype(np.int32),
            'item': item_ids.astype(np.int32),
            'rating': ratings.astype(np.float32)
        })

        spark_df = self.spark.createDataFrame(pd_df)

        # 2. Train the ALS model
        als = ALS(
            rank=self.rank,
            maxIter=self.maxIter,
            regParam=self.regParam,
            userCol="user",
            itemCol="item",
            ratingCol="rating",
            coldStartStrategy="drop",
            nonnegative=True
        )

        if progress_callback: progress_callback(0.1)
        self.spark_model = als.fit(spark_df)
        if progress_callback: progress_callback(1.0)

        self.P = None
        self.Q = None

        return self

    def predict(self):
        if self.spark_model is None:
            raise RuntimeError("The model has not been trained yet. Call fit() first.")

        # 1. Create a DataFrame of all possible (user, item) pairs for prediction
        # We only need to predict for users and items that were in the training set
        all_users = self.user_ids_in_train
        all_items = self.item_ids_in_train

        user_df = self.spark.createDataFrame([(int(u),) for u in all_users], ["user"])
        item_df = self.spark.createDataFrame([(int(i),) for i in all_items], ["item"])

        all_pairs = user_df.crossJoin(item_df)

        # 2. Get predictions
        predictions_spark = self.spark_model.transform(all_pairs)

        # 3. Convert predictions back to a dense NumPy matrix
        predictions_pd = predictions_spark.select("user", "item", "prediction").toPandas()

        full_matrix = np.zeros(self.original_shape)

        if not predictions_pd.empty:
            pivoted = predictions_pd.pivot(index='user', columns='item', values='prediction')

            # --- FIX: Use reindex to safely align the pivot table with the full matrix shape ---
            # Create a full index and column set from the original shape
            full_index = np.arange(self.original_shape[0])
            full_columns = np.arange(self.original_shape[1])

            # Reindex the pivoted table, filling missing values with 0
            pivoted_reindexed = pivoted.reindex(index=full_index, columns=full_columns, fill_value=0)

            full_matrix = pivoted_reindexed.to_numpy()

        self.R_predicted = full_matrix

        return self.R_predicted