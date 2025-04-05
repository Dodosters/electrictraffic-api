import pandas as pd
import numpy as np
import io
from typing import List, Dict, Any, Optional
import json
from datetime import datetime, timedelta

from main import analyse_excel

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
    print('eeeeeeeeeeeeeeeeeeeeeeeeeee')
    date_column = df.columns[0]
    hour_columns = df.columns[1:]
    
    # Подготавливаем структуру для хранения результатов
    series = []
    
    # Для каждой строки в DataFrame (для каждой даты)
    for idx, row in df.iterrows():
        date_label = str(row[date_column])
        points = []
        
        # Для каждого часового столбца
        for hour_col in hour_columns:
            # Извлекаем час из имени столбца (если столбец называется "Hour 1", "1:00" и т.д.)
            hour_num = hour_col
            if isinstance(hour_col, str):
                # Пытаемся извлечь числовое значение из названия столбца
                import re
                match = re.search(r'(\d+)', hour_col)
                if match:
                    hour_num = int(match.group(1))
                elif hour_col.isdigit():
                    hour_num = int(hour_col)
            
            value = row[hour_col]
            # Проверяем, является ли это числовым значением
            if pd.notna(value) and (isinstance(value, (int, float))):
                # Определяем, является ли эта точка выбросом
                is_outlier_point = is_outlier(value, df[hour_col])
                
                points.append({
                    "hour": hour_num,
                    "value": float(value),  # Преобразуем в float для корректной сериализации в JSON
                    "isOutlier": bool(is_outlier_point)
                })
        
        # Сортируем точки по часам
        points.sort(key=lambda x: x["hour"])
        
        series.append({
            "date": date_label,
            "points": points
        })
    
    return {"series": series}

def transform_to_daily_chart_format(df: pd.DataFrame) -> Dict[str, List[Dict]]:
    """
    Преобразует DataFrame в формат для графика по дням
    с агрегацией почасовых данных и определением выбросов
    
    Args:
        df: Исходный DataFrame с почасовыми данными
    
    Returns:
        Данные в формате для графика по дням с пометками выбросов
    """
    # Определяем столбец с датами и часовые столбцы
    date_column = df.columns[0]
    hour_columns = df.columns[1:]
    
    # Создаем новый DataFrame для агрегированных данных по дням
    daily_data = []
    
    # Предполагаем, что даты могут быть в разных форматах
    try:
        # Пытаемся преобразовать значения в даты, если они еще не являются datetime
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df[date_column] = pd.to_datetime(df[date_column])
        
        # Извлекаем только дату (без времени)
        days = df[date_column].dt.date.unique()
    except:
        # Если не удалось преобразовать в datetime, используем значения как есть
        days = df[date_column].unique()
    
    # Для каждого дня извлекаем все строки
    for day in days:
        day_rows = df[df[date_column].astype(str).str.contains(str(day))]
        
        # Для каждого часа вычисляем агрегированные значения по часам
        day_points = []
        
        # Для каждого часового столбца
        for hour_col in hour_columns:
            # Извлекаем час из имени столбца
            hour_num = hour_col
            if isinstance(hour_col, str):
                import re
                match = re.search(r'(\d+)', hour_col)
                if match:
                    hour_num = int(match.group(1))
                elif hour_col.isdigit():
                    hour_num = int(hour_col)
            
            # Получаем все значения для этого часа в этот день
            hour_values = day_rows[hour_col].dropna()
            
            if not hour_values.empty:
                # Вычисляем среднее значение
                avg_value = float(hour_values.mean())
                
                # Определяем, является ли среднее значение выбросом
                is_outlier_point = is_outlier(avg_value, df[hour_col])
                
                day_points.append({
                    "hour": hour_num, 
                    "value": avg_value,
                    "isOutlier": bool(is_outlier_point)
                })
        
        # Сортируем точки по часам
        day_points.sort(key=lambda x: x["hour"])
        
        daily_data.append({
            "date": str(day),
            "points": day_points
        })
    
    return {"series": daily_data}