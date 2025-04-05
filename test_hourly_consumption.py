import requests
import os

def test_hourly_consumption_file_upload():
    """Test the hourly consumption file upload endpoint"""
    base_url = "http://localhost:8000"
    url = f"{base_url}/process-hourly-consumption"
    
    # Path to the sample CSV file
    csv_path = os.path.join(os.path.dirname(__file__), "sample_hourly_consumption.csv")
    
    if not os.path.exists(csv_path):
        print(f"Error: Sample CSV file not found at {csv_path}")
        return
    
    try:
        # Open the CSV file for upload
        with open(csv_path, 'rb') as file:
            # Create form data with the file and region parameter
            files = {'file': file}
            data = {'region': 'Ростов-на-Дону'}
            
            # Send the POST request
            response = requests.post(url, files=files, data=data)
            
            # Check if the request was successful
            if response.status_code == 200:
                result = response.json()
                
                if result['success']:
                    hourly_data = result['data']['hourlyData']
                    print(f"Hourly Consumption Test: SUCCESS")
                    print(f"Number of days processed: {len(hourly_data)}")
                    
                    # Display sample data for the first day
                    first_day = hourly_data[0]
                    print(f"\nSample data for {first_day['date']}:")
                    print(f"Daily total: {first_day['dailyTotal']:.2f} руб.")
                    
                    # Show consumption and cost for a few hours
                    print("\nHourly breakdown (sample):")
                    hours_to_show = sorted(list(first_day['hoursCost'].keys()))[:5]
                    
                    for hour in hours_to_show:
                        hour_data = first_day['hoursCost'][hour]
                        print(f"Hour {hour}: Consumption: {hour_data['consumption']:.2f} kWh, "
                              f"Rate: {hour_data['rate']} руб/kWh ({hour_data['zone']}), "
                              f"Cost: {hour_data['cost']:.2f} руб.")
                    
                    # Show zone tariffs
                    zone_tariffs = result['data']['zoneTariffs']
                    print("\nZone tariffs:")
                    print(f"Peak: {zone_tariffs['peak']} руб/kWh")
                    print(f"Semi-peak: {zone_tariffs['semiPeak']} руб/kWh")
                    print(f"Night: {zone_tariffs['night']} руб/kWh")
                else:
                    print(f"Hourly Consumption Test: FAILED")
                    print(f"Error: {result['error']}")
            else:
                print(f"Hourly Consumption Test: FAILED with status code {response.status_code}")
                print(response.text)
    
    except Exception as e:
        print(f"Error during hourly consumption test: {str(e)}")

if __name__ == "__main__":
    print("Testing hourly consumption file upload endpoint...\n")
    try:
        test_hourly_consumption_file_upload()
    except requests.exceptions.ConnectionError:
        print("\nFAILED: Could not connect to the API. Make sure the server is running.")
