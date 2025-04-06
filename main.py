
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
import io
import csv
import json
import asyncio
import pandas as pd
import json
from pathlib import Path


# Import mock data
from mock_data import (
    business_tariffs,
    personal_tariffs,
    providers,
    analytics_data,
    faq_data,
    news_data
)

from prediction_api import *

app = FastAPI(
    title="ETariff API",
    description="API для доступа к данным о тарифах на электроэнергию в Ростовской области",
    version="1.0.0"
)

coefficients_file = Path("coefficients.json")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper function to simulate API delay
async def delay(ms: int):
    await asyncio.sleep(ms / 1000)  # Convert milliseconds to seconds

# Define Pydantic models for request/response validation
class CalculateBusinessRequest(BaseModel):
    region: str
    consumption: float
    power_tarif: str
    day_consumption: float
    night_consumption: float
    cost_of_energy: float
    real_volume: list
    real_average_power: float
    real_average_power_broadcast: float

class ProcessHourlyConsumptionRequest(BaseModel):
    csvData: str
    region: Optional[str] = "Ростов-на-Дону"

class ProcessExcelRequest(BaseModel):
    region: Optional[str] = "Ростов-на-Дону"

class PowerTariff(BaseModel):
    VN: float
    SN1: float
    SN2: float
    NN: float

class CoefficientRange(BaseModel):
    price_mean: float
    another_service: float
    sales_for_control: float
    sales: float
    power_tarif: PowerTariff

class CategoryCoefficients(BaseModel):
    before_670: CoefficientRange
    from_670_to_10: CoefficientRange
    from_10: CoefficientRange

# Модели для второй категории
class DayCost(BaseModel):
    price_mean: float
    another_service: float
    sales_for_control: float
    sales: float
    power_tarif: PowerTariff

class NightCost(BaseModel):
    price_mean: float
    another_service: float
    sales_for_control: float
    sales: float
    power_tarif: PowerTariff

class SecondCategoryBefore670(BaseModel):
    day_cost: DayCost
    night_cost: NightCost
    compensation: float

class SecondCategoryRange(BaseModel):
    price_mean: float
    another_service: float
    sales_for_control: float
    sales: float
    power_tarif: PowerTariff

class SecondCategoryCoefficients(BaseModel):
    before_670: SecondCategoryBefore670
    from_670_to_10: SecondCategoryRange
    from_10: SecondCategoryRange

# Модели для третьей категории
class ThirdCategoryRange(BaseModel):
    cost_for_power: float
    another_service: float
    cost_for_onestuff: float
    sales: float
    power_tarif: PowerTariff

class ThirdCategoryCoefficients(BaseModel):
    before_670: ThirdCategoryRange
    from_670_to_10: ThirdCategoryRange
    from_10: ThirdCategoryRange

# Модели для четвертой категории
class FourthCategoryRange(BaseModel):
    cost_for_power_broadcast: float
    cost_for_power: float
    another_service: float
    cost_for_onestuff: float
    sales: float
    power_tarif: PowerTariff

class FourthCategoryCoefficients(BaseModel):
    before_670: FourthCategoryRange
    from_670_to_10: FourthCategoryRange
    from_10: FourthCategoryRange

class AllCoefficients(BaseModel):
    first_category_cost: CategoryCoefficients
    second_category_cost: SecondCategoryCoefficients
    third_category_cost: ThirdCategoryCoefficients
    four_category_cost: FourthCategoryCoefficients

def save_coefficients(coeffs: dict):
    with open(coefficients_file, "w") as f: 
        json.dump(coeffs, f, indent=4, ensure_ascii=False)

def load_coefficients() -> dict:
    with open(coefficients_file, "r") as f:  
        data = json.load(f)
    return data

@app.get("/coefficients")
async def get_all_coefficients():
    return load_coefficients()

# Эндпоинты для первой категории
@app.get("/coefficients/first_category")
async def get_first_category():
    all_coeffs = load_coefficients()
    return all_coeffs["first_category_cost"]

@app.put("/coefficients/first_category")
async def update_first_category(new_coefficients: CategoryCoefficients):
    all_coeffs = load_coefficients()
    all_coeffs["first_category_cost"] = new_coefficients.dict()
    save_coefficients(all_coeffs)
    return {"message": "Коэффициенты первой категории обновлены"}

@app.get("/coefficients/first_category/{range_name}")
async def get_first_category_range(range_name: str):
    coeffs = load_coefficients()
    if range_name not in coeffs["first_category_cost"]:
        raise HTTPException(status_code=404, detail=f"Диапазон {range_name} не найден")
    return coeffs["first_category_cost"][range_name]

# Эндпоинты для второй категории
@app.get("/coefficients/second_category")
async def get_second_category():
    all_coeffs = load_coefficients()
    return all_coeffs["second_category_cost"]

@app.put("/coefficients/second_category")
async def update_second_category(new_coefficients: SecondCategoryCoefficients):
    all_coeffs = load_coefficients()
    all_coeffs["second_category_cost"] = new_coefficients.dict()
    save_coefficients(all_coeffs)
    return {"message": "Коэффициенты второй категории обновлены"}

@app.get("/coefficients/second_category/{range_name}")
async def get_second_category_range(range_name: str):
    coeffs = load_coefficients()
    if range_name not in coeffs["second_category_cost"]:
        raise HTTPException(status_code=404, detail=f"Диапазон {range_name} не найден")
    return coeffs["second_category_cost"][range_name]

# Эндпоинты для третьей категории
@app.get("/coefficients/third_category")
async def get_third_category():
    all_coeffs = load_coefficients()
    return all_coeffs["third_category_cost"]

@app.put("/coefficients/third_category")
async def update_third_category(new_coefficients: ThirdCategoryCoefficients):
    all_coeffs = load_coefficients()
    all_coeffs["third_category_cost"] = new_coefficients.dict()
    save_coefficients(all_coeffs)
    return {"message": "Коэффициенты третьей категории обновлены"}

@app.get("/coefficients/third_category/{range_name}")
async def get_third_category_range(range_name: str):
    coeffs = load_coefficients()
    if range_name not in coeffs["third_category_cost"]:
        raise HTTPException(status_code=404, detail=f"Диапазон {range_name} не найден")
    return coeffs["third_category_cost"][range_name]

# Эндпоинты для четвертой категории
@app.get("/coefficients/four_category")
async def get_fourth_category():
    all_coeffs = load_coefficients()
    return all_coeffs["four_category_cost"]

@app.put("/coefficients/four_category")
async def update_fourth_category(new_coefficients: FourthCategoryCoefficients):
    all_coeffs = load_coefficients()
    all_coeffs["four_category_cost"] = new_coefficients.dict()
    save_coefficients(all_coeffs)
    return {"message": "Коэффициенты четвертой категории обновлены"}

@app.get("/coefficients/four_category/{range_name}")
async def get_fourth_category_range(range_name: str):
    coeffs = load_coefficients()
    if range_name not in coeffs["four_category_cost"]:
        raise HTTPException(status_code=404, detail=f"Диапазон {range_name} не найден")
    return coeffs["four_category_cost"][range_name]

def analyse_excel(file_content):
    
    df = pd.read_excel(file_content)

    # Преобразуем первый столбец в дату (некорректные значения станут NaT)
    df[df.columns[0]] = pd.to_datetime(df.iloc[:, 0], errors='coerce')

    # Удаляем строки, где в первом столбце нет даты
    clean_df = df.dropna(subset=[df.columns[0]]).reset_index(drop=True)

    # Удаляем строки, где дата "1970-01-01" (возникает при некорректном преобразовании)
    clean_df = clean_df[clean_df[df.columns[0]].dt.year > 2000]

    clean_df = clean_df.copy()

    # Убедимся, что правильно указаны индексы столбцов
    date_col = clean_df.columns[0]  # Столбец с датами
    hour_col = clean_df.columns[2]  # Столбец с номером часа
    kwh_col = clean_df.columns[4]   # Столбец с потреблением

    # Создаем структурированные данные
    clean_df['date'] = pd.to_datetime(clean_df[date_col]).dt.date
    clean_df['hour'] = clean_df[hour_col].astype(int)
    # clean_df['kwh'] = .fillna(0)
    # print('eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee')
    clean_df['kwh'] = pd.to_numeric(clean_df[kwh_col], errors='coerce').interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
    

    # Проверяем наличие столбца 'date'
    if 'date' not in clean_df.columns:
        raise KeyError("Столбец 'date' не найден в DataFrame")


            # Создаем каркас для всех часов
    all_hours = pd.DataFrame({'hour': range(24)})

    # Группируем и заполняем пропуски
    hourly_series = (
        clean_df.groupby('date', group_keys=False)
        .apply(lambda grp: (
            pd.merge(all_hours, grp[['hour', 'kwh']], on='hour', how='left')
            .assign(date=grp.name) # Добавляем дату обратно
        ))
        .fillna({'kwh': 0})
        .groupby('date')['kwh']
        .apply(list)
    )
    
    # Преобразуем Series в DataFrame
    result_df = pd.DataFrame({'hourly_kwh': hourly_series})

    return result_df

# Excel processing function
def excel_to_json(file_content: bytes):
    """
    Обрабатывает Excel-файл, преобразует данные в почасовой формат и возвращает JSON.

    Args:
        file_content (bytes): Содержимое файла (bytes).

    Returns:
        str: JSON-строка с данными о почасовом потреблении.
              Возвращает None, если произошла ошибка.
    """
    try:
        
        result = analyse_excel(file_content)
        # all_hours = pd.DataFrame({'hour': range(24)})
        # result = (
        #     clean_df.groupby('date', group_keys=False)
        #     .apply(lambda grp: (
        #         pd.merge(all_hours, grp[['hour', 'kwh']], on='hour', how='left')
        #         .assign(date=grp.name)  # Добавляем дату обратно
        #     ))
        #     .fillna({'kwh': 0})
        #     .groupby('date')['kwh']
        #     .apply(list)
        #     .to_dict()
        # )
        # Конвертируем в JSON

        print(result.head())

        result_dict = {str(idx): [round(float(v), 2) for v in row] 
                     for idx, row in result['hourly_kwh'].items()}
        
        json_output = json.dumps(
            result_dict,
            indent=2,
            ensure_ascii=False
        )

        print('wwwwwwwwwwwwwwwwwww')
        return json_output

    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Файл не найден.")
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Произошла ошибка: {str(e)}")

# API routes
@app.get("/business-tariffs")
async def get_business_tariffs():
    await delay(500)
    return {"success": True, "data": business_tariffs}

@app.get("/business-tariffs/{region}")
async def get_business_tariffs_by_region(region: str):
    await delay(500)
    tariffs = [tariff for tariff in business_tariffs if tariff["region"] == region]
    return {"success": True, "data": tariffs}

@app.get("/personal-tariffs")
async def get_personal_tariffs():
    await delay(500)
    return {"success": True, "data": personal_tariffs}

@app.get("/personal-tariffs/{region}")
async def get_personal_tariffs_by_region(region: str):
    await delay(500)
    tariffs = [tariff for tariff in personal_tariffs if tariff["region"] == region]
    return {"success": True, "data": tariffs}

@app.get("/providers")
async def get_providers():
    await delay(500)
    return {"success": True, "data": providers}

@app.get("/providers/{id}")
async def get_provider_by_id(id: int):
    await delay(500)
    provider = next((p for p in providers if p["id"] == id), None)
    if provider:
        return {"success": True, "data": provider}
    return {"success": False, "error": "Provider not found"}

@app.get("/analytics")
async def get_analytics_data():
    await delay(700)
    return {"success": True, "data": analytics_data}

@app.post("/calculate/business")
async def calculate_business_electricity_cost(params: CalculateBusinessRequest):
    await delay(600)
    region = params.region
    consumption_real = params.consumption
    power_tarif = params.power_tarif
    day_consumption = params.day_consumption
    night_consumption = params.night_consumption
    cost_of_energy = params.cost_of_energy
    real_volume = params.real_volume
    real_average_power = params.real_average_power
    real_average_power_broadcast = params.real_average_power_broadcast

    coefficients = load_coefficients()

    consumption = "before_670" if consumption_real < 670 else "from_670_to_10" if consumption_real < 1000 else "from_10"


    def category_one(region, consumption, coefficients):
        region_tariff = next((t for t in business_tariffs if t["region"] == region), None)
        if not region_tariff:
            return {"success": False, "error": "Region not found"}
        category = "first_category_cost" #первая категория 
        result_coeff = coefficients[category][consumption]["price_mean"]+coefficients[category][consumption]["another_service"]+coefficients[category][consumption]["sales_for_control"]+coefficients[category][consumption]["sales"]+coefficients[category][consumption]["power_tarif"][power_tarif]
        cost = consumption_real*result_coeff
        return cost
    def category_two(region, coefficients, night_consumption, day_consumption):
        region_tariff = next((t for t in business_tariffs if t["region"] == region), None)
        if not region_tariff:
            return {"success": False, "error": "Region not found"}
        category = "second_category_cost"
        
        # Проверяем, есть ли структура day_cost/night_cost в данном диапазоне потребления
        if "day_cost" in coefficients[category][consumption] and "night_cost" in coefficients[category][consumption]:
            # Для диапазона before_670
            night_coeff = coefficients[category][consumption]["night_cost"]["price_mean"]+coefficients[category][consumption]["night_cost"]["another_service"]+coefficients[category][consumption]["night_cost"]["sales_for_control"]+coefficients[category][consumption]["night_cost"]["sales"]+coefficients[category][consumption]["night_cost"]["power_tarif"][power_tarif]
            day_coeff = coefficients[category][consumption]["day_cost"]["price_mean"]+coefficients[category][consumption]["day_cost"]["another_service"]+coefficients[category][consumption]["day_cost"]["sales_for_control"]+coefficients[category][consumption]["day_cost"]["sales"]+coefficients[category][consumption]["day_cost"]["power_tarif"][power_tarif]
            lost = (night_consumption+day_consumption) * coefficients[category][consumption]["compensation"]
            lost_for_day = lost*day_consumption/(night_consumption+day_consumption) if (night_consumption+day_consumption) > 0 else 0
            lost_for_night = lost*night_consumption/(night_consumption+day_consumption) if (night_consumption+day_consumption) > 0 else 0
            common_lost = lost_for_day*day_coeff+lost_for_night*night_coeff
            broadcast = (day_consumption+night_consumption)*(1-coefficients[category][consumption]["compensation"])
            cost = broadcast+common_lost+night_coeff*night_consumption+day_coeff*day_consumption
        else:
            # Для диапазонов from_670_to_10 и from_10
            coeff = coefficients[category][consumption]["price_mean"]+coefficients[category][consumption]["another_service"]+coefficients[category][consumption]["sales_for_control"]+coefficients[category][consumption]["sales"]+coefficients[category][consumption]["power_tarif"][power_tarif]
            cost = coeff * (day_consumption + night_consumption)
            
        return cost 
    def category_third(region, cost_of_energy, coefficients, real_average_power, real_volume):
        region_tariff = next((t for t in business_tariffs if t["region"] == region), None)
        if not region_tariff:
            return {"success": False, "error": "Region not found"}
        category = "third_category_cost"
        result = 0
        for i in range(len(real_volume)):
            result=real_volume[i]*(cost_of_energy+coefficients[category][consumption]["another_service"]+cost_of_energy+coefficients[category][consumption]["sales"]+cost_of_energy+coefficients[category][consumption]["cost_for_onestuff"])
        return (result+real_average_power*coefficients[category][consumption]["cost_for_power"])
    
    def category_four(region, cost_of_energy, coefficients, real_average_power, real_average_power_broadcast, real_volume):
        region_tariff = next((t for t in business_tariffs if t["region"] == region), None)
        if not region_tariff:
            return {"success": False, "error": "Region not found"}
        category = "four_category_cost"
        result = 0
        for i in range(len(real_volume)):
            result=real_volume[i]*(cost_of_energy+coefficients[category][consumption]["another_service"]+cost_of_energy+coefficients[category][consumption]["sales"]+cost_of_energy+coefficients[category][consumption]["cost_for_onestuff"])
        return (result+real_average_power*coefficients[category][consumption]["cost_for_power"]+real_average_power_broadcast*coefficients[category][consumption]["cost_for_power_broadcast"])
        
    return {
        "success": True,
        "data": {
            "region": region,
            "consumption": consumption,
            "cost1": round(category_one(region, consumption, coefficients) * 100) / 100,
            "cost2": round(category_two(region, coefficients, night_consumption, day_consumption) * 100) / 100,
            "cost3": round(category_third(region, cost_of_energy, coefficients, real_average_power, real_volume) * 100) / 100,
            "cost4": round(category_four(region, cost_of_energy, coefficients, real_average_power, real_average_power_broadcast, real_volume) * 100) / 100,
            "currency": "руб."
        }
    }


@app.post("/process-hourly-consumption")
async def process_hourly_consumption(file: UploadFile = File(...), region: str = Form("Ростов-на-Дону")):

    await delay(800)
    
    try:
        # Read the uploaded CSV file
        contents = await file.read()
        csv_data = contents.decode("utf-8")
        
        # Parse CSV data
        rows = csv_data.split("\n")
        
        if len(rows) < 2:
            return {"success": False, "error": "Файл не содержит данных"}
        
        # Determine delimiter (';' or ',')
        delimiter = ";"
        if "," in rows[0] and ";" not in rows[0]:
            delimiter = ","
        
        # Extract headers
        headers = rows[0].split(delimiter)
        
        # Create a map to store data by date
        days_map = {}
        
        # Process data rows
        for i in range(1, len(rows)):
            if not rows[i].strip():
                continue  # Skip empty lines
            
            values = rows[i].split(delimiter)
            if len(values) < 3:
                continue  # Skip rows with insufficient columns
            
            # Extract date and hour
            date_str = values[0].strip()
            hour_str = values[2].strip()
            
            # Convert date to a more readable format
            # Format M/D/YYYY to DD.MM.YYYY
            if "/" in date_str and date_str.count("/") == 2:
                month, day, year = date_str.split("/")
                date_str = f"{day}.{month}.{year}"
            
            # Convert hour to integer
            try:
                hour = int(hour_str)
                if hour < 0 or hour > 23:
                    continue
                
                # Normalize hour to 1-24 range (instead of 0-23)
                normalized_hour = 24 if hour == 0 else hour + 1
                
                # Determine consumption column (assuming index 9)
                consumption_column_index = 9
                
                consumption = 0
                if len(values) > consumption_column_index:
                    try:
                        # Remove quotes and replace comma with dot for number conversion
                        consumption_str = values[consumption_column_index].replace('"', '').replace(',', '.')
                        consumption = float(consumption_str) if consumption_str else 0
                    except Exception as e:
                        print(f"Error converting value to number: {values[consumption_column_index]}", e)
                
                # Add data to the dictionary, grouping by date
                if date_str not in days_map:
                    days_map[date_str] = {"date": date_str, "hours": {}}
                
                days_map[date_str]["hours"][normalized_hour] = consumption
            except ValueError:
                continue  # Skip non-numeric hour values
        
        # Convert dictionary to list for further processing
        hourly_data = list(days_map.values())
        
        if not hourly_data:
            return {
                "success": False,
                "error": "Не удалось извлечь данные из файла. Пожалуйста, проверьте формат файла или укажите какой столбец содержит потребление."
            }
        
        # Get tariff data for the region
        region_tariff = next((t for t in business_tariffs if t["region"] == region), None)
        
        if not region_tariff:
            return {"success": False, "error": "Регион не найден"}
        
        # Get three-zone tariff for hourly calculation
        three_zone_tariff = next((t for t in region_tariff["tariffTypes"] if t["name"] == "Трехставочный"), None)
        
        if not three_zone_tariff:
            return {"success": False, "error": "Трехставочный тариф не найден для указанного региона"}
        
        # Define day zones and tariffs
        hourly_rates = {}
        
        # Peak hours (assume 8-11 and 16-21)
        peak_hours = [8, 9, 10, 11, 16, 17, 18, 19, 20, 21]
        
        # Night hours (23-7)
        night_hours = [23, 0, 1, 2, 3, 4, 5, 6, 7]
        
        # Semi-peak hours - all others
        semi_peak_hours = [h for h in range(1, 25) if h not in peak_hours and h not in night_hours]
        
        # Fill hourly rates
        for hour in range(1, 25):
            if hour in peak_hours:
                hourly_rates[hour] = {
                    "rate": next((r["rate"] for r in three_zone_tariff["rates"] if r["name"] == "Пик"), 0),
                    "zone": "Пик"
                }
            elif hour in night_hours:
                hourly_rates[hour] = {
                    "rate": next((r["rate"] for r in three_zone_tariff["rates"] if r["name"] == "Ночь"), 0),
                    "zone": "Ночь"
                }
            else:
                hourly_rates[hour] = {
                    "rate": next((r["rate"] for r in three_zone_tariff["rates"] if r["name"] == "Полупик"), 0),
                    "zone": "Полупик"
                }
        
        # Calculate cost for each day and hour
        processed_data = []
        for day in hourly_data:
            hours_cost = {}
            daily_total = 0
            
            for hour_str, consumption in day["hours"].items():
                hour = int(hour_str)
                hour_rate = hourly_rates.get(hour, {
                    "rate": next((r["rate"] for r in three_zone_tariff["rates"] if r["name"] == "Полупик"), 0),
                    "zone": "Полупик"
                })
                
                cost = consumption * hour_rate["rate"]
                
                hours_cost[hour_str] = {
                    "consumption": consumption,
                    "rate": hour_rate["rate"],
                    "zone": hour_rate["zone"],
                    "cost": cost
                }
                
                daily_total += cost
            
            processed_data.append({
                **day,
                "hoursCost": hours_cost,
                "dailyTotal": daily_total
            })
        
        return {
            "success": True,
            "data": {
                "hourlyData": processed_data,
                "hourlyRates": hourly_rates,
                "zoneTariffs": {
                    "peak": next((r["rate"] for r in three_zone_tariff["rates"] if r["name"] == "Пик"), 0),
                    "semiPeak": next((r["rate"] for r in three_zone_tariff["rates"] if r["name"] == "Полупик"), 0),
                    "night": next((r["rate"] for r in three_zone_tariff["rates"] if r["name"] == "Ночь"), 0)
                }
            }
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": f"Ошибка обработки данных: {str(e)}"
        }

@app.post("/process-hourly-consumption-string")
async def process_hourly_consumption_string(params: ProcessHourlyConsumptionRequest):
    await delay(800)
    
    try:
        # Parse CSV data
        csv_data = params.csvData
        rows = csv_data.split("\n")
        
        if len(rows) < 2:
            return {"success": False, "error": "Файл не содержит данных"}
        
        # Determine delimiter (';' or ',')
        delimiter = ";"
        if "," in rows[0] and ";" not in rows[0]:
            delimiter = ","
        
        # Extract headers
        headers = rows[0].split(delimiter)
        
        # Create a map to store data by date
        days_map = {}
        
        # Process data rows
        for i in range(1, len(rows)):
            if not rows[i].strip():
                continue  # Skip empty lines
            
            values = rows[i].split(delimiter)
            if len(values) < 3:
                continue  # Skip rows with insufficient columns
            
            # Extract date and hour
            date_str = values[0].strip()
            hour_str = values[2].strip()
            
            # Convert date to a more readable format
            # Format M/D/YYYY to DD.MM.YYYY
            if "/" in date_str and date_str.count("/") == 2:
                month, day, year = date_str.split("/")
                date_str = f"{day}.{month}.{year}"
            
            # Convert hour to integer
            try:
                hour = int(hour_str)
                if hour < 0 or hour > 23:
                    continue
                
                # Normalize hour to 1-24 range (instead of 0-23)
                normalized_hour = 24 if hour == 0 else hour + 1
                
                # Determine consumption column (assuming index 9)
                consumption_column_index = 9
                
                consumption = 0
                if len(values) > consumption_column_index:
                    try:
                        # Remove quotes and replace comma with dot for number conversion
                        consumption_str = values[consumption_column_index].replace('"', '').replace(',', '.')
                        consumption = float(consumption_str) if consumption_str else 0
                    except Exception as e:
                        print(f"Error converting value to number: {values[consumption_column_index]}", e)
                
                # Add data to the dictionary, grouping by date
                if date_str not in days_map:
                    days_map[date_str] = {"date": date_str, "hours": {}}
                
                days_map[date_str]["hours"][normalized_hour] = consumption
            except ValueError:
                continue  # Skip non-numeric hour values
        
        # Convert dictionary to list for further processing
        hourly_data = list(days_map.values())
        
        if not hourly_data:
            return {
                "success": False,
                "error": "Не удалось извлечь данные из файла. Пожалуйста, проверьте формат файла или укажите какой столбец содержит потребление."
            }
        
        # Get tariff data for the region
        region = params.region
        region_tariff = next((t for t in business_tariffs if t["region"] == region), None)
        
        if not region_tariff:
            return {"success": False, "error": "Регион не найден"}
        
        # Get three-zone tariff for hourly calculation
        three_zone_tariff = next((t for t in region_tariff["tariffTypes"] if t["name"] == "Трехставочный"), None)
        
        if not three_zone_tariff:
            return {"success": False, "error": "Трехставочный тариф не найден для указанного региона"}
        
        # Define day zones and tariffs
        hourly_rates = {}
        
        # Peak hours (assume 8-11 and 16-21)
        peak_hours = [8, 9, 10, 11, 16, 17, 18, 19, 20, 21]
        
        # Night hours (23-7)
        night_hours = [23, 0, 1, 2, 3, 4, 5, 6, 7]
        
        # Semi-peak hours - all others
        semi_peak_hours = [h for h in range(1, 25) if h not in peak_hours and h not in night_hours]
        
        # Fill hourly rates
        for hour in range(1, 25):
            if hour in peak_hours:
                hourly_rates[hour] = {
                    "rate": next((r["rate"] for r in three_zone_tariff["rates"] if r["name"] == "Пик"), 0),
                    "zone": "Пик"
                }
            elif hour in night_hours:
                hourly_rates[hour] = {
                    "rate": next((r["rate"] for r in three_zone_tariff["rates"] if r["name"] == "Ночь"), 0),
                    "zone": "Ночь"
                }
            else:
                hourly_rates[hour] = {
                    "rate": next((r["rate"] for r in three_zone_tariff["rates"] if r["name"] == "Полупик"), 0),
                    "zone": "Полупик"
                }
        
        # Calculate cost for each day and hour
        processed_data = []
        for day in hourly_data:
            hours_cost = {}
            daily_total = 0
            
            for hour_str, consumption in day["hours"].items():
                hour = int(hour_str)
                hour_rate = hourly_rates.get(hour, {
                    "rate": next((r["rate"] for r in three_zone_tariff["rates"] if r["name"] == "Полупик"), 0),
                    "zone": "Полупик"
                })
                
                cost = consumption * hour_rate["rate"]
                
                hours_cost[hour_str] = {
                    "consumption": consumption,
                    "rate": hour_rate["rate"],
                    "zone": hour_rate["zone"],
                    "cost": cost
                }
                
                daily_total += cost
            
            processed_data.append({
                **day,
                "hoursCost": hours_cost,
                "dailyTotal": daily_total
            })
        
        return {
            "success": True,
            "data": {
                "hourlyData": processed_data,
                "hourlyRates": hourly_rates,
                "zoneTariffs": {
                    "peak": next((r["rate"] for r in three_zone_tariff["rates"] if r["name"] == "Пик"), 0),
                    "semiPeak": next((r["rate"] for r in three_zone_tariff["rates"] if r["name"] == "Полупик"), 0),
                    "night": next((r["rate"] for r in three_zone_tariff["rates"] if r["name"] == "Ночь"), 0)
                }
            }
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": f"Ошибка обработки данных: {str(e)}"
        }

@app.post("/process-excel")
async def process_excel(file: UploadFile = File(...)):
    """
    API endpoint для обработки Excel-файла, отправленного в теле запроса.

    Returns:
        JSON: JSON-строка с данными о почасовом потреблении, либо сообщение об ошибке.
    """
    await delay(800)

    if not file:
        raise HTTPException(status_code=400, detail="Нет файла в запросе")

    try:
        contents = await file.read()
        json_data = excel_to_json(contents)
        
        return JSONResponse(content=json.loads(json_data), media_type="application/json") #Convert json_data (string) to dict (JSON)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке файла: {str(e)}")


@app.get("/faqs")
async def get_faqs():
    await delay(400)
    return {"success": True, "data": faq_data}

@app.get("/news")
async def get_news():
    await delay(400)
    return {"success": True, "data": news_data}

@app.get("/news/{id}")
async def get_news_by_id(id: int):
    await delay(300)
    news = next((n for n in news_data if n["id"] == id), None)
    if news:
        return {"success": True, "data": news}
    return {"success": False, "error": "News article not found"}

# Предсказание
@app.post("/forecast/", response_model=ForecastResponse)
async def upload_train_forecast(months_ahead: int, file: UploadFile = File(...)):
    """Upload data, train model, and generate forecast in one request"""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    if months_ahead < 1 or months_ahead > 24:
        raise HTTPException(status_code=400, detail="months_ahead must be between 1 and 24")
    
    try:
        # Read CSV content
        contents = await file.read()
        buffer = io.StringIO(contents.decode('utf-8'))
        df = pd.read_csv(buffer)
        
        # Check required columns
        required_cols = ['year', 'month', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise HTTPException(
                status_code=400,
                detail=f"CSV must contain the following columns: {', '.join(required_cols)}"
            )
        
        # Check if data is valid for forecasting
        if len(df) < 3:
            raise HTTPException(
                status_code=400, 
                detail="Data must contain at least 3 data points for forecasting"
            )
        
        # Prepare the data
        df_prepared = load_and_prepare_data(df)
        
        # Train the model
        models, feature_names = train_ensemble_model(df_prepared)
        
        # Generate forecast
        forecast_results, future_df = forecast_recursive(
            models, 
            feature_names, 
            df_prepared, 
            months_ahead
        )
        
        # Prepare response
        response = {
            "forecast": forecast_results
        }
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")





def is_outlier(value: float, series: pd.Series, factor: float = 1.5) -> bool:
    """
    Определяет, является ли значение выбросом на основе метода межквартильного размаха (IQR)
    
    Args:
        value: Проверяемое значение
        series: Серия данных для вычисления статистики
        factor: Множитель для диапазона IQR (обычно 1.5)
        
    Returns:
        True, если значение является выбросом
    """
    # Удаляем NaN значения
    series = series.dropna()
    
    # Находим квартили
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    
    # Вычисляем межквартильный размах
    iqr = q3 - q1
    
    # Определяем границы выбросов
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    # Проверяем, выходит ли значение за границы
    return value < lower_bound or value > upper_bound

def transform_to_hourly_chart_format(df: pd.DataFrame) -> Dict[str, List[Dict]]:
    """
    Преобразует DataFrame в формат почасового графика с определением выбросов
    
    Args:
        df: Исходный DataFrame с почасовыми данными
    
    Returns:
        Данные в формате для графика с пометками выбросов
    """
    # Определяем столбец с датами и часовые столбцы
    # Предполагаем, что первый столбец - это дата или идентификатор
    series = []
    
    # Предполагаем, что df имеет индекс с датами и столбец 'hourly_kwh' со списками значений
    for date, row in df.iterrows():
        date_label = str(date)
        hourly_data = row['hourly_kwh']
        
        # Создаем временный DataFrame для определения выбросов
        temp_df = pd.DataFrame({'hour': range(24), 'value': hourly_data})
        
        points = []
        for hour, value in enumerate(hourly_data):
            # Определяем, является ли точка выбросом
            is_outlier_point = is_outlier(value, pd.Series(hourly_data))
            
            points.append({
                "hour": hour,
                "value": float(value),  # Преобразуем в float для корректной сериализации
                "isOutlier": bool(is_outlier_point)
            })
        
        series.append({
            "date": date_label,
            "points": points
        })
    
    return {"series": series}

def transform_to_daily_chart_format(df: pd.DataFrame) -> Dict[str, List[Dict]]:
    """
    Преобразует DataFrame в формат для графика по дням
    с суммированием почасовых данных в рамках дня и определением выбросов
    
    Args:
        df: Исходный DataFrame с почасовыми данными
    
    Returns:
        Данные в формате для графика по дням с пометками выбросов
    """
    series = []
    
    # Создаем новый массив для хранения суммарных значений по дням
    daily_values = []
    daily_labels = []
    
    # Для каждой даты в DataFrame суммируем все часовые значения
    for date, row in df.iterrows():
        date_label = str(date)
        hourly_data = row['hourly_kwh']
        
        # Суммируем все почасовые значения для данного дня
        daily_sum = sum(hourly_data)
        
        # Сохраняем результат
        daily_values.append(daily_sum)
        daily_labels.append(date_label)
    
    # Преобразуем в Series для определения выбросов
    daily_series = pd.Series(daily_values)
    
    # Добавляем точки для каждого дня в формате графика
    points = []
    for i, (day_label, value) in enumerate(zip(daily_labels, daily_values)):
        # Определяем, является ли значение выбросом
        is_outlier_point = is_outlier(value, daily_series)
        
        points.append({
            "day": i,  # Индекс дня или можно использовать день месяца
            "date": day_label,  # Сохраняем метку даты
            "value": float(value),  # Преобразуем в float для корректной сериализации
            "isOutlier": bool(is_outlier_point)
        })
    
    # Группируем все точки в один набор данных
    series.append({
        "points": points
    })
    
    return {"series": series}

def transform_to_weekday_profile_format(df: pd.DataFrame) -> Dict[str, List[Dict]]:
    """
    Преобразует DataFrame в формат профилей нагрузки по дням недели
    с агрегацией почасовых данных и определением выбросов
    
    Args:
        df: Исходный DataFrame с почасовыми данными
    
    Returns:
        Данные в формате для графика профилей нагрузки по дням недели с пометками выбросов
    """
    # Названия дней недели для отображения
    weekday_names = ['Понедельник', 'Вторник', 'Среда', 'Четверг', 'Пятница', 'Суббота', 'Воскресенье']
    
    # Словарь для хранения данных по каждому дню недели
    weekday_data = {day: [[] for _ in range(24)] for day in range(7)}
    
    # Обрабатываем каждую дату в DataFrame
    for date, row in df.iterrows():
        # Преобразуем индекс в datetime, если он еще не является таковым
        if not isinstance(date, pd.Timestamp):
            try:
                date = pd.to_datetime(date)
            except:
                # Если не удалось преобразовать, пропускаем эту запись
                continue
        
        # Получаем день недели (0 - понедельник, 6 - воскресенье)
        weekday = date.weekday()
        hourly_data = row['hourly_kwh']
        
        # Добавляем почасовые данные в соответствующие списки дня недели
        for hour, value in enumerate(hourly_data):
            weekday_data[weekday][hour].append(float(value))
    
    # Создаем серию результатов
    series = []
    
    # Обрабатываем каждый день недели
    for weekday in range(7):        
        # Обрабатываем каждый час в дне недели
        day_value = 0
        for hour in range(24):
            hour_values = weekday_data[weekday][hour]
            
            # Если есть данные для этого часа в этот день недели
            if hour_values:
                # Вычисляем среднее значение
                avg_value = sum(hour_values) / len(hour_values)
                
                # Определяем, является ли среднее значение выбросом                
                day_value += avg_value
        
        # Добавляем профиль дня недели в результаты
        series.append({
            "date": weekday_names[weekday],  # Используем название дня недели вместо даты
            "day_value": day_value
        })
    
    return {"series": series}


@app.post("/chart-data")
async def get_chart_data(file: UploadFile = File(...), view_type: Optional[str] = "hourly"):
    """
    Маршрут для получения данных графика из загруженного Excel файла

    Args:
        file: Загруженный Excel файл
        view_type: Тип представления данных. Возможные значения: "hourly" (почасовое), "daily" (по дням)

    Returns:
        Данные в формате JSON для построения графика с пометками выбросов
    """
    try:
        file_content = await file.read()
        
        # Используем функцию analyse_excel для получения DataFrame
        df = analyse_excel(file_content)

        # В зависимости от типа представления, возвращаем разные данные
        if view_type.lower() == "daily":
            # Данные агрегированы по дням
            chart_data = transform_to_daily_chart_format(df)
        else:
            # Почасовые данные (по умолчанию)
            chart_data = transform_to_hourly_chart_format(df)
            
        
        return chart_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process Excel file: {str(e)}")

@app.post("/weekday-profile")
async def get_weekday_profile(file: UploadFile = File(...)):
    """
    Маршрут для получения профилей нагрузки по дням недели из загруженного Excel файла
    
    Args:
        file: Загруженный Excel файл
    
    Returns:
        Данные в формате JSON для построения графика профилей нагрузки 
        по дням недели с пометками выбросов
    """
    try:
        file_content = await file.read()
        
        # Используем функцию analyse_excel для получения DataFrame
        df = analyse_excel(file_content)
        
        # Преобразуем данные для профилей нагрузки по дням недели
        chart_data = transform_to_weekday_profile_format(df)
            
        return chart_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process Excel file: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
