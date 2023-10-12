import json
import urllib
import yfinance as yf
import plotly.graph_objs as go
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
import datetime
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t, f
from sklearn.linear_model import LinearRegression
import pandas as pd
from django.http import HttpResponse
from django.shortcuts import render
import requests
from xml.etree import ElementTree
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import plotly.offline as opy
from statsmodels.tsa.arima_model import ARIMA
import datetime
import matplotlib.pyplot as plt
import numpy as np
from django.shortcuts import render
from scipy.stats import t, f
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from scipy.optimize import minimize

import pandas as pd
from django.http import HttpResponse
from sklearn.metrics import mean_squared_error

# Устанавливаем высокое разрешение
plt.gcf().set_dpi(600)

df = pd.DataFrame()


def get_current_price():
    data = yf.download(tickers='ALI=F', period='7d')
    latest_price = data['Close'].iloc[-1]
    return (latest_price)


def get_preview_image(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    meta = soup.find('meta', property='og:image')
    return meta['content'] if meta else None


def get_rss_feed():
    url = "https://news.rambler.ru/rss/politics/"
    response = requests.get(url)
    if response.status_code == 200:
        tree = ElementTree.fromstring(response.content)

        items = tree.findall('channel/item')
        i = 0
        news = []

        for item in items:
            if i != 8:
                news_item = {
                    'title': item.find('title').text,
                    'description': item.find('description').text,
                    'link': item.find('link').text,
                    'pubDate': item.find('pubDate').text,
                    'image': get_preview_image(item.find('link').text),
                }
                news.append(news_item)
                i = i + 1
        return news
    else:
        return None


def index(request):
    try:
        if request.method == 'POST':
            url = f"http://192.168.18.159/data?mes=Курс алюминия {get_current_price()}$ &p=2 &n=3"
            news = get_rss_feed()
            current_price = get_current_price();
            if 'file' in request.FILES:
                file = request.FILES['file']
                alfa = float(request.POST['alfa'])
                if file.name.endswith('.csv'):
                    df = pd.read_csv(file)

                    # Проверка на буквы в файле и удаление строк с ними кроме 1 строки
                    data = yf.download(tickers='ALI=F', period='3650d', interval='1mo')
                    # Подготовка данных для графика
                    data.reset_index(inplace=True)
                    dates = data['Date'].dt.strftime('%Y-%m-%d')
                    closes = data['Close']
                    future_df_data = pd.DataFrame({'Date': dates, 'Close': closes})
                    # Подготовка данных для графика
                    charts_data = future_df_data.to_json(orient='records')

                    df = df.rename(
                        columns={'Дата': 'Date', 'Значение': 'Value'})

                    df = df.dropna(subset=['Date', 'Value'])
                    for index, row in df.iterrows():
                        date_str = row['Date']
                        value_str = row['Value']

                        # Проверка соответствия формату даты
                        try:
                            datetime.datetime.strptime(date_str, '%Y-%m-%d')
                        except ValueError:
                            df = df.drop(index)
                            continue

                        # Проверка соответствия формату числа
                        try:
                            float(value_str)
                        except ValueError:
                            return render(request, 'index.html',
                                          {'price': current_price, 'error': 'Файл содержит лишние символы'})

                        # Проверка на увеличение значения на 50%
                        if index > 0:
                            prev_value = float(df.iloc[index - 1]['Value'])
                            if value_str > 1.5 * prev_value:
                                df = df.drop(index)
                                print('Удалена строка с датой ' + date_str + ' из-за увеличения значения на 50%')
                                continue

                    data = df.copy().to_json(orient='records')

                    # Создание и обучение модели ARIMA
                    model = ARIMA(df['Value'], order=(10, 0, 10))
                    model_fit = model.fit()

                    # Получение прогноза на следующие 10 периодов
                    forecast = model_fit.forecast(steps=10)

                    # Получение даты следующих 10 периодов
                    forecast_dates = pd.date_range(start=df['Date'].iloc[-1], periods=10, freq='D')

                    # получение датафрейма с прогнозом
                    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Value': forecast})

                    # конвертация датафрейма в json
                    forecast_json = forecast_df.to_json(orient='records')

                    # Примеры 1-3
                    average_level = df['Value'].mean()

                    df['Growth'] = df['Value'].diff()
                    growth = df['Growth'].mean()

                    df['Growth_1'] = df['Value'].pct_change() + 1
                    growth_1 = df['Growth_1'].iloc[1:].prod() ** (1 / (len(df) - 1))
                    growth_srednegod = round(growth_1 * 100, 2)

                    # Пример 4
                    df['Month'] = pd.to_datetime(df['Date']).dt.month
                    df_first_half = df[df['Month'].isin([1, 2, 3, 4, 5, 6])]
                    df_second_half = df[df['Month'].isin([7, 8, 9, 10, 11, 12])]

                    y1 = df_first_half['Value'].mean()
                    y2 = df_second_half['Value'].mean()

                    # Пример 5
                    sigma1 = df_first_half['Value'].var(ddof=1)
                    sigma2 = df_second_half['Value'].var(ddof=1)

                    F = sigma1 / sigma2

                    n1 = len(df_first_half)
                    n2 = len(df_second_half)
                    v1 = n1 - 1
                    v2 = n2 - 1
                    F_tabl = f.ppf(1 - alfa, v1, v2)

                    SE = np.sqrt(sigma1 / n1 + sigma2 / n2)
                    t_stat = abs(y1 - y2) / SE

                    df_t = n1 + n2 - 2
                    t_tabl = t.ppf(1 - alfa / 2, df_t)
                    # проверка гипотезы
                    if t_stat < t_tabl:
                        gipoteza = "отвергается"
                    else:
                        gipoteza = "принимается"

                    # Пример 6
                    def moving_average(data, window_size, center=False):
                        if center:
                            window_size += 1
                        averages = []
                        for i in range(len(data) - window_size + 1):
                            window = data.iloc[i:i + window_size]["Value"].tolist()
                            average = sum(window) / len(window)
                            if center:
                                position = i + (window_size - 1) // 2
                            else:
                                position = i + window_size - 1
                            averages.append((data.iloc[position]["Date"], average))
                        return pd.DataFrame(averages, columns=["Date", "Average"])

                    # Скользящая средняя из трех уровней
                    ma3 = moving_average(df, 3)
                    # Скользящая средняя из четырёх уровней без центрирования
                    ma4_no_center = moving_average(df, 4)
                    # Скользящая средняя из четырёх уровней с центрированием
                    ma4_center = moving_average(df, 4, center=True)

                    plt.figure(figsize=(12, 6))
                    # уменьшение размера маркеров
                    plt.rcParams['lines.markersize'] = 0.5
                    plt.plot(df["Date"], df["Value"], marker='o', label='Исходные данные', linestyle='-')
                    # отображение по оси x только даты, с шагом 12
                    plt.xticks(df["Date"][::12])

                    # соеденить ma3, ma4_no_center, ma4_center в один датафрейм
                    ma3_copy = ma3.rename(columns={"Average": "MA3"})
                    ma4_no_center_copy = ma4_no_center.rename(columns={"Average": "MA4_no_center"})
                    ma4_center_copy = ma4_center.rename(columns={"Average": "MA4_center"})
                    ma3_copy = ma3_copy.set_index("Date")
                    ma4_no_center_copy = ma4_no_center_copy.set_index("Date")
                    ma4_center_copy = ma4_center_copy.set_index("Date")
                    ma3_copy = ma3_copy.join(ma4_no_center_copy)
                    ma3_copy = ma3_copy.join(ma4_center_copy)

                    ma3_data = ma3.to_json(orient='records')
                    ma4_no_center_data = ma4_no_center.to_json(orient='records')
                    ma4_center_data = ma4_center.to_json(orient='records')

                    # Пример 7
                    df["Date_encoded"] = range(1, len(df) + 1)

                    # Разделение данных на обучающую и тестовую выборки
                    train_size = int(len(df) * 0.8)
                    train_set, test_set = df[:train_size], df[train_size:]

                    # Обучение модели линейной регрессии
                    regressor = LinearRegression()
                    regressor.fit(train_set[["Date_encoded"]], train_set["Value"])

                    # Прогноз на тестовой выборке
                    test_set["Predictions"] = regressor.predict(test_set[["Date_encoded"]])

                    # Прогноз на будущий период
                    future_dates = pd.date_range(start=df["Date"].iloc[-1], periods=12, freq="M")
                    future_values = regressor.predict(pd.DataFrame({"Date_encoded": range(len(df) + 1, len(df) + 13)}))
                    future_df = pd.DataFrame({"Date": future_dates, "Value": future_values})

                    test_set_data = test_set[["Date", "Predictions"]].copy().to_json(orient='records')
                    future_df_data = future_df[["Date", "Value"]].copy().to_json(orient='records')

                    df['Value'] = df['Value'].astype(float)

                    # создание датасета для построения графика с полями date и value из df
                    df_copy = df[['Date', 'Value']].copy()

                    chart_data = df_copy.to_json(orient='records')

                    # 1. Абсолютный прирост (цепной)
                    df['chain_abs_growth'] = df['Value'].diff()

                    # 2. Абсолютный прирост (базисный)
                    df['base_abs_growth'] = df['Value'] - df['Value'].iloc[0]

                    # 3. Темп роста (цепной)
                    df['chain_growth_rate'] = df['Value'] / df['Value'].shift(1)

                    # 4. Темп роста (базисный)
                    df['base_growth_rate'] = df['Value'] / df['Value'].iloc[0]

                    # 5. Темп прироста (цепной)
                    df['chain_growth_tempo'] = df['chain_abs_growth'] / df['Value'].shift(1)

                    # 6. Темп прироста (базисный)
                    df['base_growth_tempo'] = df['base_growth_rate'] - df['base_growth_rate'].iloc[0]

                    # 7. Абсолютное значение 1% прироста
                    df['abs_value_of_1_percent_growth'] = df['Value'].shift(1) * 0.01

                    # 8. Относительное ускорение темпов роста
                    df['relative_growth_acceleration'] = df['chain_growth_rate'].diff()

                    # 9. Коэффициент опережения
                    df['lead_ratio'] = df['chain_growth_rate'] / df['chain_growth_rate'].shift(1)

                    # Исключаем первую строку, так как не можем вычислить прирост для первого значения
                    # df = df[1:]

                    # Функция для создания линейного тренда и предсказания
                    # Получение коэффициентов линейного тренда
                    linear_coeffs = np.polyfit(range(len(df['Value'])), df['Value'], 1)

                    # Создание линейного тренда
                    linear_trend = np.poly1d(linear_coeffs)

                    # Предсказание на основе линейного тренда
                    linear_predictions = linear_trend(range(len(df['Value'])))

                    # Определение функции экспоненциального тренда
                    def exponential_func(params, x):
                        a, b = params
                        return a * np.exp(b * x)

                    # Определение функции потерь для минимизации
                    def loss_func(params, x, y):
                        y_pred = exponential_func(params, x)
                        return np.mean((y - y_pred) ** 2)

                    # Автоподбор начальных параметров
                    def auto_initial_params(x, y):
                        # Начальные параметры
                        a0 = np.mean(y)
                        b0 = np.log(np.mean(y) / np.min(y)) / (x[-1] - x[0])
                        return [a0, b0]

                    # Получение начальных параметров
                    initial_params = auto_initial_params(range(len(df['Value'])), df['Value'])

                    # Минимизация функции потерь для подбора оптимальных параметров
                    result = minimize(loss_func, initial_params, args=(range(len(df['Value'])), df['Value']))

                    # Получение оптимальных параметров экспоненциального тренда
                    exp_coeffs = result.x

                    # Создание экспоненциального тренда
                    exp_trend = exponential_func(exp_coeffs, range(len(df['Value'])))

                    # Предсказание на основе экспоненциального тренда
                    exp_predictions = exponential_func(exp_coeffs, range(len(df['Value'])))

                    # Получение коэффициентов гиперболического тренда
                    hyper_coeffs = np.polyfit(range(len(df['Value'])), 1 / df['Value'], 1)

                    # Создание гиперболического тренда
                    hyper_trend = np.poly1d(hyper_coeffs)

                    # Предсказание на основе гиперболического тренда
                    hyper_predictions = 1 / hyper_trend(range(len(df['Value'])))

                    # Создаем тренды для каждого типа модели
                    df['linear_trend'] = linear_predictions
                    df['exponential_trend'] = exp_predictions
                    df['hyperbolic_trend'] = hyper_predictions

                    hyperbolic_trend = df[['Date', 'hyperbolic_trend']].copy()
                    hyperbolic_trend = hyperbolic_trend.to_json(orient='records')
                    exponential_trend = df[['Date', 'exponential_trend']].copy()
                    exponential_trend = exponential_trend.to_json(orient='records')
                    linear_trend = df[['Date', 'linear_trend']].copy()
                    linear_trend = linear_trend.to_json(orient='records')
                    linear_trend_data = df[['Date', 'Value', 'linear_trend']].copy()
                    hyperbolic_trend_data = df[['Date', 'Value', 'hyperbolic_trend']].copy()
                    exponential_trend_data = df[['Date', 'Value', 'exponential_trend']].copy()

                    # Вычисляем ошибку для линейного тренда
                    mse_linear = mean_squared_error(df['Value'], df['linear_trend'])

                    # Вычисляем ошибку для экспоненциального тренда
                    mse_exponential = mean_squared_error(df['Value'], df['exponential_trend'])

                    # Вычисляем ошибку для гиперболического тренда
                    mse_hyperbolic = mean_squared_error(df['Value'], df['hyperbolic_trend'])

                    # выбор наилучшей по меньшей ошибке модели
                    if mse_linear < mse_exponential and mse_linear < mse_hyperbolic:
                        best_model = 'linear'
                    elif mse_exponential < mse_linear and mse_exponential < mse_hyperbolic:
                        best_model = 'exponential'
                    else:
                        best_model = 'hyperbolic'
                    # Отрисовываем исходные данные
                    plt.figure(figsize=(12, 8))
                    plt.plot(df['Value'], label='Original data')

                    df.drop(['Growth', 'Growth_1', 'Month', 'Date_encoded'], axis=1, inplace=True)

                    df = df.rename(
                        columns={'Date': 'Дата', 'Value': 'Значение', 'chain_abs_growth': 'Абсолютный прирост (цепной)',
                                 'base_abs_growth': 'Абсолютный прирост (базисный)',
                                 'chain_growth_rate': 'Темп роста (цепной)',
                                 'base_growth_rate': 'Темп роста (базисный)',
                                 'chain_growth_tempo': 'Темп прироста (цепной)',
                                 'base_growth_tempo': 'Темп прироста (базисный)',
                                 'abs_value_of_1_percent_growth': 'Абсолютное значение 1% прироста',
                                 'relative_growth_acceleration': 'Относительное ускорение темпов роста',
                                 'lead_ratio': 'Коэффициент опережения', 'linear_trend': 'Линейный тренд',
                                 'exponential_trend': 'Экспоненциальный тренд',
                                 'hyperbolic_trend': 'Гиперболический тренд'})

                    request.session['df'] = df.to_json()

                    df = df.to_html(
                        classes='table table-striped table-hover table-bordered table-sm table-responsive text-center')

                    ma3_copy = ma3_copy.to_html(
                        classes='table table-striped table-hover table-bordered table-sm table-responsive text-center')

                    hyperbolic_trend_data = hyperbolic_trend_data.rename(
                        columns={'Date': 'Дата', 'Value': 'Значение', 'hyperbolic_trend': 'Гиперболический тренд'})
                    hyperbolic_trend_data = hyperbolic_trend_data.to_html(
                        classes='table table-striped table-hover table-bordered table-sm table-responsive text-center')

                    exponential_trend_data = exponential_trend_data.rename(
                        columns={'Date': 'Дата', 'Value': 'Значение', 'exponential_trend': 'Экспоненциальный тренд'})
                    exponential_trend_data = exponential_trend_data.to_html(
                        classes='table table-striped table-hover table-bordered table-sm table-responsive text-center')

                    linear_trend_data = linear_trend_data.rename(
                        columns={'Date': 'Дата', 'Value': 'Значение', 'linear_trend': 'Линейный тренд'})
                    linear_trend_data = linear_trend_data.to_html(
                        classes='table table-striped table-hover table-bordered table-sm table-responsive text-center')

                    return render(request, 'success.html',
                                  {'average_level': round(average_level, 2), 'growth': round(growth, 2),
                                   'growth_srednegod': round(growth_srednegod, 2),
                                   'y1': round(y1, 2), 'y2': round(y2, 2),
                                   'sigma1': round(sigma1, 2), 'sigma2': round(sigma2, 2),
                                   'F': round(F, 2), 'F_tabl': round(F_tabl, 2), 't_stat': round(t_stat, 2),
                                   't_tabl': round(t_tabl, 2),
                                   'gipoteza': gipoteza,
                                   'df': df,
                                   'chart_data': chart_data,
                                   'ma3_data': ma3_data,
                                   'ma4_no_center_data': ma4_no_center_data,
                                   'ma4_center_data': ma4_center_data,
                                   'data': data,
                                   'hyperbolic_trend': hyperbolic_trend,
                                   'exponential_trend': exponential_trend,
                                   'linear_trend': linear_trend,
                                   'test_set_data': test_set_data,
                                   'future_df_data': future_df_data,
                                   'ma3': ma3_copy,
                                   'linear_trend_data': linear_trend_data,
                                   'hyperbolic_trend_data': hyperbolic_trend_data,
                                   'exponential_trend_data': exponential_trend_data,
                                   'mse_linear': mse_linear,
                                   'mse_exponential': mse_exponential,
                                   'mse_hyperbolic': mse_hyperbolic,
                                   'best_model': best_model,
                                   'forecast_json': forecast_json,
                                   'news': news,
                                   'alfa': alfa,
                                   'price': current_price,
                                   'url': url,
                                   'charts_data': charts_data,

                                   })



                else:
                    current_price = get_current_price();
                    return render(request, 'index.html', {'price': current_price, 'error': 'Неверный формат файла'})
            else:
                current_price = get_current_price();
                return render(request, 'index.html', {'price': current_price, 'error': 'Файл не выбран'})

        else:
            current_price = get_current_price();
            return render(request, 'index.html', {'price': current_price, })


    except:
        current_price = get_current_price();
        return render(request, 'index.html', { 'price': current_price,'error': 'Что-то пошло не так...'})


def download_df(request):
    # Извлекаем DataFrame из сессии
    df = pd.read_json(request.session['df'])

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="data.csv"'
    df.to_csv(path_or_buf=response, sep=',', float_format='%.2f', index=False, encoding='utf-8')

    return response


def test(request):
    return render(request, 'test.html')


def chart(request):
    return render(request, 'chart.html')
