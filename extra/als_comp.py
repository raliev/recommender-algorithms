# Matrix Factorisation Recommendation System

# import packages
import pandas as pd
import numpy as np
import requests
import io
import torch
from torch import tensor
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType
from pyspark.ml.recommendation import ALS



RATINGS_DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-100k/u.data"
MOVIES_DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-100k/u.item"
# TAGS_DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-100k/u.genre"

# fetch ratings data
res_1 = requests.get(RATINGS_DATA_URL)
res_2 = requests.get(MOVIES_DATA_URL)

if (
        res_1.status_code==200 and
        res_2.status_code==200
):
    data1 = res_1.content
    data2 = res_2.content
    print("Data fetch successfull...")

df_data = pd.read_csv(
    io.StringIO(data1.decode("utf-8")),
    sep='\t', header=None,
    names=["user_id", "movie_id", "rating", "timestamp"],
    index_col=False,
    engine = 'python')

df_movies = pd.read_csv(
    io.StringIO(data2.decode("latin-1")),
    sep='|', header=None,
    names=["movie_id", "title"],
    usecols=[0,1],
    index_col=False,
    engine = 'python')

# merge datasets to add Title col
df_data = pd.merge(df_data, df_movies[["movie_id", "title"]], how="left", on="movie_id")
# # type cast cols
# df_data.astype({
#     "user_id": "int32",
#     "movie_id": "int32",
#     "rating": "float",
#     "timestamp": "int32",
#     "title": 'str'
# })

print("Dataset has {} total ratings".format(df_data.shape[0]))
print("Dataset has {} total users who rated movies".format(df_data["user_id"].nunique()))
print("Dataset has {} total rated movies".format(df_data["movie_id"].nunique()))

# initiate a new spark session
spark = SparkSession.builder.appName(
    name='MovieLensALS'
).config(
    'spark.executor.memory', '2g'
).config(
    'spark.driver.memory', '2g'
).config(
    'spark.sql.shuffle.partitions', '8'
).getOrCreate()
# define the schema
schema = StructType([
    StructField("user_id", IntegerType(), True),
    StructField("movie_id", IntegerType(), True),
    StructField("rating", FloatType(), True)
])

# cast original dataframe ratings col to float
df_data['rating'] = df_data['rating'].astype('float')

# Convert with schema
dfs = spark.createDataFrame(df_data, schema=schema)

# rename cols as spark accepst 'user' & 'movie' col names
dfs = dfs.withColumnsRenamed({
    'user_id': 'user',
    'movie_id': 'movie',
})

# split data in train & test sets
dfs_train, dfs_test = dfs.randomSplit([0.8, 0.2], 42)
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS

# initiate model without hyperparams
# initialise the model
als = ALS(userCol='user', itemCol='movie', ratingCol='rating',
          coldStartStrategy="drop", nonnegative=True)

# create a hyperparameter grid
param_grid = ParamGridBuilder().addGrid(
    als.rank, [1, 15, 30, 45]
).addGrid(
    als.maxIter, [10]
).addGrid(
    als.regParam, [0.05, 0.1, 0.15]
).build()

# create an evaluation metric
eval = RegressionEvaluator(metricName='rmse', labelCol='rating', predictionCol='prediction')

# use cross validator to run models
cv = CrossValidator(
    estimator=als,
    estimatorParamMaps=param_grid,
    evaluator=eval,
    numFolds=3,
    collectSubModels=True
)

cv_model = cv.fit(dfs_train)

# load all sub models
all_models = cv_model.subModels

# initialise an array to save regression evaluation params
all_models_reg_eval = []
# load all models
all_models = cv_model.subModels
# iterate over all_models to access each model
for fold_idx, models_per_fold in enumerate(all_models):
    for model_idx, model in enumerate(models_per_fold):
        # generate predictions
        dfs_preds = model.transform(dfs_test)
        # set up regression evaluator
        model_reg_eval = RegressionEvaluator(metricName='rmse', labelCol='rating', predictionCol='prediction')
        rmse = model_reg_eval.evaluate(dfs_preds)
        r2 = model_reg_eval.evaluate(dfs_preds, {eval.metricName: 'r2'})
        mae = model_reg_eval.evaluate(dfs_preds, {eval.metricName: 'mae'})
        var = model_reg_eval.evaluate(dfs_preds, {eval.metricName: 'var'})

        all_models_reg_eval.append(
            {
                'rank': model._java_obj.parent().getRank(),
                'regParam': model._java_obj.parent().getRegParam(),
                'rmse': rmse,
                'r2': r2,
                'mae': mae,
                'var': var
            }
        )

# convert to pandas dataframe
df_all_models_reg_eval = pd.DataFrame(all_models_reg_eval)
# transform 36 models -> 12 models
# aggregate metrics with mean
df_all_models_reg_eval_grouped = df_all_models_reg_eval.groupby(['rank', 'regParam']).agg(
    avg_rmse=('rmse', 'mean'),
    avg_r2=('r2', 'mean'),
    avg_mae=('mae', 'mean'),
    avg_var=('var', 'mean')
).reset_index()

# Pivot the DataFrame to create a grid for heatmap
# change 'values' to print regression metrics like R2, MAE, VAR
heatmap_data = df_all_models_reg_eval_grouped.pivot(index='regParam', columns='rank', values='avg_rmse')

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap='viridis', cbar_kws={'label': 'RMSE'})
plt.title('Heatmap of RMSE by Rank and regParam')
plt.xlabel('Rank')
plt.ylabel('Regularization Parameter (regParam)')
plt.show()