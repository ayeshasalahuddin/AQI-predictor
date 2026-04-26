import requests
import os
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("AQICN_TOKEN")
CITY = "karachi"

url = f"https://api.waqi.info/feed/{CITY}/?token={TOKEN}"

response = requests.get(url)
data = response.json()

if data["status"] == "ok":
    aqi = data["data"]["aqi"]
    city_name = data["data"]["city"]["name"]
    print(f"City: {city_name}")
    print(f"Current AQI: {aqi}")
    print(f"Raw data keys: {list(data['data'].keys())}")
else:
    print("Something went wrong:", data)