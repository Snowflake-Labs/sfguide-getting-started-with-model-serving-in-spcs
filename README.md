# sfguide-getting-started-with-model-serving-in-spcs

Snowflake ML: Chicago Bus Ridership Forecasting
This project demonstrates a complete, end-to-end machine learning workflow for time-series forecasting using the Snowflake ML library. The goal is to predict daily Chicago bus ridership by training, deploying, and managing an XGBoost model entirely within the Snowflake ecosystem.

The notebook showcases how to leverage Snowflake's powerful features for data engineering, model training, and deployment, providing a scalable and efficient MLOps solution.

Key Features Demonstrated
Secure Connection: Establishing a secure connection to Snowflake using the Snowpark Python API.
Lazy DataFrames: Loading data from Snowflake tables into Snowpark DataFrames for transformations.
Snowflake Feature Store: Creating and managing features for consistent use in training and inference.
Distributed Feature Engineering: Performing data preparation and feature engineering at scale within Snowflake.
Distributed Model Training: Training a gradient-boosting model using Snowpark ML's distributed processing capabilities.
Snowflake Model Registry: Storing and versioning the trained model for governance and reproducibility.
In-Database Scoring: Running model predictions directly inside Snowflake warehouses.
Model Deployment: Deploying the model as a service to Snowpark Container Services (SPCS) for real-time inference.
