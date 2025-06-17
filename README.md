# sfguide-getting-started-with-model-serving-in-spcs

Snowflake ML: Chicago Bus Ridership Forecasting
This project demonstrates a complete, end-to-end machine learning workflow for time-series forecasting using the Snowflake ML library. The goal is to predict daily Chicago bus ridership by training, deploying, and managing an XGBoost model entirely within the Snowflake ecosystem.

The notebook showcases how to leverage Snowflake's powerful features for data engineering, model training, and deployment, providing a scalable and efficient MLOps solution.

Source Notebook: Forecasting_ChicagoBus/Snowpark_Forecasting_Bus_FeatureStore.ipynb

Dataset and Resources: GitHub Folder

Key Features Demonstrated
Secure Connection: Establishing a secure connection to Snowflake using the Snowpark Python API.
Lazy DataFrames: Loading data from Snowflake tables into Snowpark DataFrames for transformations.
Snowflake Feature Store: Creating and managing features for consistent use in training and inference.
Distributed Feature Engineering: Performing data preparation and feature engineering at scale within Snowflake.
Distributed Model Training: Training a gradient-boosting model using Snowpark ML's distributed processing capabilities.
Snowflake Model Registry: Storing and versioning the trained model for governance and reproducibility.
In-Database Scoring: Running model predictions directly inside Snowflake warehouses.
Model Deployment: Deploying the model as a service to Snowpark Container Services (SPCS) for real-time inference.
Project Workflow
1. Environment Setup
The environment is initialized by importing the necessary libraries from Snowpark, Snowflake ML, and other utilities. A Snowpark session is established to connect to the Snowflake database.

Python

# Snowpark for Python
from snowflake.snowpark.session import Session
from snowflake.snowpark import functions as F

# Snowpark ML
from snowflake.ml.modeling.pipeline import Pipeline
from snowflake.ml.modeling.preprocessing import StandardScaler, OrdinalEncoder
from snowflake.ml.modeling.xgboost import XGBRegressor
from snowflake.ml.feature_store import FeatureStore, CreationMode, Entity, FeatureView
from snowflake.ml.registry import Registry

# Create a session
session = get_active_session()
2. Data Loading and Preparation
The initial dataset, CTA_Daily_Totals_by_Route.csv, is loaded into a Pandas DataFrame, then written to a Snowflake table. This ensures the data is centralized and accessible for future use without needing manual re-uploads.

Python

# Load from CSV and save to a Snowflake table
df_clean = pd.read_csv('CTA_Daily_Totals_by_Route.csv')
df_clean.columns = df_clean.columns.str.upper()
input_df = session.create_dataframe(df_clean)
input_df.write.mode('overwrite').save_as_table('MODEL_SERVING_DB.FEATURE_STORE_MLDEMO.CHICAGO_BUS_RIDES')
3. Distributed Feature Engineering with the Feature Store
Feature engineering is performed directly in Snowflake.

Create an Entity: An Entity is defined to represent the primary key for feature lookups.
Engineer Bus Features: New features like DAY_OF_WEEK, MONTH, and PREV_DAY_RIDERS are created from the raw data. This logic is then registered as a FeatureView named AggBusData.
Incorporate Weather Data: Weather data is seamlessly integrated from the Snowflake Marketplace. This external dataset is joined with the bus data and registered as another FeatureView (weather), which is set to refresh daily.
Generate Training Data: The Feature Store combines these FeatureViews using a spine DataFrame to create a consistent and reliable training dataset.
Python

# Initialize Feature Store
fs = FeatureStore(
    session=session,
    database="MODEL_SERVING_DB",
    name="FEATURE_STORE_MLDEMO",
    default_warehouse="DEMO_BUILD_WH",
    creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
)

# Create and register a feature view for aggregated bus data
agg_fv = FeatureView(
    name="AggBusData",
    entities=[entity],
    feature_df=total_riders,
    timestamp_col="DATE",
)
fs.register_feature_view(agg_fv, version="1", overwrite=True)

# Generate the final training set from the feature store
training_set = fs.generate_training_set(
    spine_df=spine_df,
    features=[agg_fv, weather_fv]
)
4. Distributed Model Training in a Pipeline
A preprocessing and training pipeline is constructed using snowflake.ml. This pipeline handles scaling numeric features, encoding categorical features, and training an XGBRegressor model. Hyperparameter tuning is performed using GridSearchCV. All of these operations are executed in a distributed manner on a Snowpark-optimized warehouse for enhanced performance.

Python

# Define preprocessing steps for numeric and categorical features
numeric_features = ['DAY_OF_WEEK', 'MONTH', 'PREV_DAY_RIDERS', 'MINIMUM_TEMPERATURE', 'MAXIMUM_TEMPERATURE', 'PRECIPITATION']
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

categorical_cols = ['DAYTYPE']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-99999))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define the full pipeline with preprocessor and model
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', XGBRegressor())])

# Use GridSearchCV for hyperparameter optimization
xg_model = GridSearchCV(...)
xg_model.fit(train)
5. Model Evaluation
The model's performance is evaluated on the test set. The mean_absolute_error is calculated, and the predictions are materialized to a new table in Snowflake for analysis and tracking.

Python

from snowflake.ml.modeling.metrics import mean_absolute_error

testpreds = xg_model.predict(test)
print('MSE:', mean_absolute_error(df=testpreds, y_true_col_names='TOTAL_RIDERS', y_pred_col_names='\"TOTAL_RIDERS_FORECAST\"'))

# Save results to a table
testpreds.write.save_as_table(table_name='MODEL_SERVING_DB.FEATURE_STORE_MLDEMO.CHICAGO_BUS_RIDES_FORECAST', mode='overwrite')
6. Model Registry and Prediction
The trained and tuned model is saved to the Snowflake Model Registry. This versions the model, its parameters, and associated metrics, providing a central repository for model governance.

Python

from snowflake.ml.registry import Registry

# Connect to the registry
reg = Registry(session=session, database_name="MODEL_SERVING_DB", schema_name="FEATURE_STORE_MLDEMO")

# Log the model
model_ref = reg.log_model(
    model_name="Forecasting_Bus_Ridership",
    version_name="v2",
    model=xg_model,
    sample_input_data=train,
)

# Retrieve the model and run predictions
reg_model = reg.get_model("Forecasting_Bus_Ridership").version("v2")
remote_prediction = reg_model.run(test, function_name='predict')
Evaluation metrics, like Mean Absolute Error (MAE), are also logged to the registry for the specific model version.

7. Lineage and Explainability
The notebook demonstrates how to use Snowflake's lineage feature to trace data flow from the source table to the final predictions. It also shows how to generate SHAP-based explanations for model predictions.

Python

# Get SHAP explanations
explanations = reg_model.run(test, function_name="explain")

# Trace data lineage
df = session.lineage.trace("MODEL_SERVING_DB.FEATURE_STORE_MLDEMO.CHICAGO_BUS_RIDES", "TABLE", direction="downstream")
8. Deployment to Snowpark Container Services (SPCS)
Finally, the model is deployed from the registry as an inference service in Snowpark Container Services (SPCS). This makes the model available via an endpoint for real-time, low-latency predictions, without needing to manage complex infrastructure.

SQL

-- Create a compute pool for the service
CREATE COMPUTE POOL mypool
  MIN_NODES = 1
  MAX_NODES = 1
  INSTANCE_FAMILY = GPU_NV_S;
Python

# Deploy the model from the registry to an SPCS service
reg_model.create_service(service_name="ChicagoBusForecastv12",
                  service_compute_pool="mypool",
                  ...
                  ingress_enabled=True)

# Run predictions against the deployed service
spcs_prediction = reg_model.run(test, function_name='predict', service_name="CHICAGOBUSFORECASTV12")
