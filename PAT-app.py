from datetime import timedelta
import argparse
import logging
import sys
import requests
import numpy as np
from pprint import pprint
import json
import pandas as pd

logger = logging.getLogger(__name__)

def main():
    args = _parse_args()
    spcs_access_token = token_exchange_pat(args.pat, endpoint=args.endpoint,
					      role=args.role,
					      snowflake_account_url=args.snowflake_account_url, 
      snowflake_account=args.account)
    spcs_url=f'https://{args.endpoint}{args.endpoint_path}'
    connect_to_spcs(spcs_access_token, spcs_url)

def token_exchange_pat(pat, role, endpoint, snowflake_account_url, snowflake_account) -> str:
    scope_role = f'session:role:{role}' if role is not None else None
    scope = f'{scope_role} {endpoint}' if scope_role is not None else endpoint

    data = {
        'grant_type': 'urn:ietf:params:oauth:grant-type:token-exchange',
        'scope': scope,
        'subject_token': pat,
        'subject_token_type': 'programmatic_access_token'
    }
    logger.info(data)
    url = f'https://{snowflake_account}.snowflakecomputing.com/oauth/token'
    if snowflake_account_url:
        url = f'{snowflake_account_url}/oauth/token'
    response = requests.post(url, data=data)
    logger.info("snowflake token : %s" % response.text)
    logger.info("snowflake token response code : %s" % response.status_code)
    assert 200 == response.status_code, "unable to get snowflake token"
    return response.text


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
       

def connect_to_spcs(token, url):
  # Create a request to the ingress endpoint with authz.
  headers = {'Authorization': f'Snowflake Token="{token}"'}
  logger.info(url)
  request_body = prepare_data()  # Convert the DataFrame to the right format
  print (request_body)
  response = requests.post(f'{url}',data=json.dumps(request_body), headers=headers)     #data=json.dumps(request_body)
  #assert (response.status_code == 200), f"Failed to get response from the service. Status code: {response.status_code}"
  #return response.content
  logger.info("return code %s" % response.status_code)
  logger.info(response.text)
  logger.info(response.content)
  logger.debug(response.content)

def _parse_args():
  logging.basicConfig(stream=sys.stdout, level=logging.INFO)
  cli_parser = argparse.ArgumentParser()
  cli_parser.add_argument('--account', required=True,
              help='The account identifier (for example, "myorganization-myaccount" for '
                '"myorganization-myaccount.snowflakecomputing.com").')
  cli_parser.add_argument('--user', required=True, help='The user name.')
  cli_parser.add_argument('--role',
              help='The role we want to use to create and maintain a session for. If a role is not provided, '
                'use the default role.')
  cli_parser.add_argument('--endpoint', required=True,
              help='The ingress endpoint of the service')
  cli_parser.add_argument('--endpoint-path', default='/',
              help='The url path for the ingress endpoint of the service')
  cli_parser.add_argument('--snowflake_account_url', default=None,
              help='The account url of the account for which we want to log in. Type of '
                'https://myorganization-myaccount.snowflakecomputing.com')
  cli_parser.add_argument('--pat',
                            help='PAT for the user.')
  args = cli_parser.parse_args()
  return args


if __name__ == "__main__":
    main()