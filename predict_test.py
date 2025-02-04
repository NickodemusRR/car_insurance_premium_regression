# Test prediction of a data in the deployed app
import requests

url = "http://localhost:9696/predict"

customer_id = "abc-789"
customer = {
    "driver_age": 56.0,
    "driver_experience": 7.0,
    "previous_accidents": 0.0,
    "annual_mileage_(x1000_km)": 22.0,
    "car_manufacturing_year": 2009.0,
    "car_age": 16.0,
}

response = requests.post(url, json=customer).json()
print(response)
