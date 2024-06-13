# heartbeat.py
import requests
import time

url = "https://iatfad.onrender.com"  # Replace with your app's URL

while True:
    try:
        requests.get(url)
        print("Pinged the service to keep it alive.")
    except Exception as e:
        print(f"Failed to ping the service: {e}")
    time.sleep(600)  # Ping every 10 minutes
