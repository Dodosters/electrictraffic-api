import requests
import json

# Base URL for the API
BASE_URL = "http://localhost:8000"

def test_get_business_tariffs():
    """Test fetching all business tariffs"""
    response = requests.get(f"{BASE_URL}/business-tariffs")
    if response.status_code == 200:
        data = response.json()
        print("Business Tariffs Test: SUCCESS")
        print(f"Number of tariffs: {len(data['data'])}")
    else:
        print(f"Business Tariffs Test: FAILED with status code {response.status_code}")
        print(response.text)

def test_get_business_tariffs_by_region():
    """Test fetching business tariffs by region"""
    region = "Ростов-на-Дону"
    response = requests.get(f"{BASE_URL}/business-tariffs/{region}")
    if response.status_code == 200:
        data = response.json()
        print(f"Business Tariffs for {region} Test: SUCCESS")
        print(f"Number of tariffs: {len(data['data'])}")
    else:
        print(f"Business Tariffs for {region} Test: FAILED with status code {response.status_code}")
        print(response.text)

def test_calculate_business_cost():
    """Test calculating business electricity cost"""
    payload = {
        "region": "Ростов-на-Дону",
        "tariffType": "Одноставочный",
        "consumption": 1000,
        "period": "Месяц"
    }
    response = requests.post(f"{BASE_URL}/calculate/business", json=payload)
    if response.status_code == 200:
        data = response.json()
        print("Calculate Business Cost Test: SUCCESS")
        print(f"Calculated cost: {data['data']['cost']} {data['data']['currency']}")
    else:
        print(f"Calculate Business Cost Test: FAILED with status code {response.status_code}")
        print(response.text)

def test_calculate_personal_cost():
    """Test calculating personal electricity cost"""
    payload = {
        "region": "Ростов-на-Дону",
        "tariffType": "Одноставочный",
        "consumption": 500,
        "period": "Месяц"
    }
    response = requests.post(f"{BASE_URL}/calculate/personal", json=payload)
    if response.status_code == 200:
        data = response.json()
        print("Calculate Personal Cost Test: SUCCESS")
        print(f"Calculated cost: {data['data']['cost']} {data['data']['currency']}")
    else:
        print(f"Calculate Personal Cost Test: FAILED with status code {response.status_code}")
        print(response.text)

def test_get_analytics():
    """Test fetching analytics data"""
    response = requests.get(f"{BASE_URL}/analytics")
    if response.status_code == 200:
        data = response.json()
        print("Analytics Data Test: SUCCESS")
        print(f"Regional comparisons: {len(data['data']['regionalComparison'])}")
        print(f"Yearly trends: {len(data['data']['yearlyTrends'])}")
    else:
        print(f"Analytics Data Test: FAILED with status code {response.status_code}")
        print(response.text)

def test_get_faqs():
    """Test fetching FAQs"""
    response = requests.get(f"{BASE_URL}/faqs")
    if response.status_code == 200:
        data = response.json()
        print("FAQs Test: SUCCESS")
        print(f"Number of FAQs: {len(data['data'])}")
    else:
        print(f"FAQs Test: FAILED with status code {response.status_code}")
        print(response.text)

def test_get_news():
    """Test fetching news articles"""
    response = requests.get(f"{BASE_URL}/news")
    if response.status_code == 200:
        data = response.json()
        print("News Test: SUCCESS")
        print(f"Number of news articles: {len(data['data'])}")
    else:
        print(f"News Test: FAILED with status code {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    print("Running API tests...\n")
    
    try:
        # Run all tests
        test_get_business_tariffs()
        print()
        
        test_get_business_tariffs_by_region()
        print()
        
        test_calculate_business_cost()
        print()
        
        test_calculate_personal_cost()
        print()
        
        test_get_analytics()
        print()
        
        test_get_faqs()
        print()
        
        test_get_news()
        
        print("\nAll tests completed.")
    except requests.exceptions.ConnectionError:
        print("\nFAILED: Could not connect to the API. Make sure the server is running.")
