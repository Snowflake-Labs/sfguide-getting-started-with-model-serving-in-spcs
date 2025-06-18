import json
from pprint import pprint
import requests
import snowflake.connector
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
import pandas as pd
import numpy as np

# --- 1. Connection and Authentication ---
# This part of your code is correct. It gets a valid token.
def initiate_snowflake_connection():
    # IMPORTANT: Ensure "named_connection" is correctly defined in your
    # ~/.snowflake/connections.toml file.
    connection_parameters = SnowflakeLoginOptions("named_connection")
    connection_parameters["session_parameters"] = {"PYTHON_CONNECTOR_QUERY_RESULT_FORMAT": "json"}
    connection_parameters["insecure_mode"] = True  # Disable SSL verification
    snowflake_conn = snowflake.connector.connect(**connection_parameters)
    return snowflake_conn

def get_headers(snowflake_conn):
    token = snowflake_conn._rest._token_request('ISSUE')
    headers = {'Authorization': f'Snowflake Token="{token["data"]["sessionToken"]}"'}
    return headers

print("Connecting to Snowflake to get auth token...")
headers = get_headers(initiate_snowflake_connection())
print("Token acquired successfully.")

# --- 2. Endpoint URL ---
# This must be the public endpoint for your service.
URL = 'https://<YOUR_INGRESS_PATH>/predict'

# --- 3. Preparing our data ---
def prepare_data():
    # Ensure 'test' is defined before this point
    # Assuming 'test' is a Snowpark DataFrame object
    test = [[6 ,7,"A",486139,734535,"USW00014819",22.8,34.4,3.6 ,'2019-07-13'],  
[6 ,6,"A",444588,784734,"USW00014819",13.9,21.7,5.1 ,'2019-06-15'],
[4 ,6,"W",780768,765409,"USW00014819",11.1,20.6,0.5 ,'2019-06-13']]

    df = pd.DataFrame(test, columns=['DAY_OF_WEEK' ,'MONTH' ,'DAYTYPE' ,'TOTAL_RIDERS' ,'PREV_DAY_RIDERS' ,'NOAA_WEATHER_STATION_ID' ,'MINIMUM_TEMPERATURE' ,'MAXIMUM_TEMPERATURE' ,'PRECIPITATION' ,'DATE'])
    print(df)

    data = {"data": np.column_stack([range(df.shape[0]), df.values]).tolist()}     
    print(data)
    return data

sample_df = prepare_data()


# --- 4. Sending the Request ---
def send_request(data: dict, headers: dict):
    """Posts the data to the URL and handles the response."""
    print("\nSending request to endpoint...")
    try:
        output = requests.post(URL, json=sample_df, headers=headers)
        
        print("--- Response from Server ---")
        print(f"Status Code: {output.status_code}")
        
        # Try to print JSON if possible, otherwise print raw text
        try:
            response_json = output.json()
            print("Response JSON:")
            pprint(response_json)
        except json.JSONDecodeError:
            print("Raw Response Text:")
            print(output.text)
            
        return output

    except requests.exceptions.RequestException as e:
        print(f"An error occurred during the request: {e}")
        return None

# --- Run the request ---
send_request(data=sample_df, headers=headers)