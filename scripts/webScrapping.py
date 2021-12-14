# Major Cities Weather
import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt

URL = 'https://mausam.imd.gov.in/'

page = requests.get(URL)
soup = BeautifulSoup(page.content, 'html.parser')
# print(soup.prettify())

today_weather = soup.find(id="today")
# print(today_weather.prettify())

# finding names of Cities listed
names = [nm.get_text() for nm in today_weather.find_all("h3")]
print(names)
# ['Mumbai ', 'Bengaluru ', 'Chennai ', 'Hyderabad ', 'Kolkata ', 'Ahmedabad ', 'Pune ', 'Delhi ']

today_forecast = today_weather.find_all(class_="capital")

temp_now_tags = [temp.get_text() for temp in today_weather.select(".capital .now .val")]
print(temp_now_tags)

wind_tag = [w.get_text() for w in today_weather.select(".capital .wind")]
print(wind_tag)

min_max = [mn.get_text() for mn in today_weather.select(".capital .minmax .max ")]
print(min_max)

# Create DataFrame of weather Data

forecast_pd = pd.DataFrame({
    "Cities": names,
    "Temp": temp_now_tags,
    "Wind": wind_tag,
    "min_max": min_max
})

# Replace degree symbol from data (Example 32° to 32)
forecast_pd['Temp'] = forecast_pd['Temp'].str.replace("°", '')
# print(forecast_pd['Temp'].describe())
# Convert Temp column to numeric
forecast_pd['Temp'] = pd.to_numeric(forecast_pd['Temp'], errors='ignore')

# Highest and lowest temperature
high_temp = forecast_pd['Temp'].max()
low_temp = forecast_pd['Temp'].min()

# print(forecast_pd[forecast_pd['Temp'] == high_temp]['Cities'])
# print(forecast_pd)

fig, ax = plt.subplots(figsize=(10,6))

# Add Data to plot
ax.plot(forecast_pd["Cities"], forecast_pd['Temp'], color='b', marker='o')

# set x_ticklabels orientation
ax.xaxis.set_ticks(forecast_pd["Cities"])  # Added to handle fixed locator waring
ax.set_xticklabels(forecast_pd["Cities"], rotation=45)

# set label
ax.set_xlabel("Cities")
ax.set_ylabel("Temperature(in Celsius)")

# Set Title
ax.set_title("Major Cities Today's Weather", color='gray')

# annotate Highest Temperature
ax.annotate("Highest Temperature", xy=(forecast_pd[forecast_pd['Temp'] == high_temp]['Cities'], high_temp),
            xytext=(forecast_pd[forecast_pd['Temp'] == high_temp]['Cities'], high_temp + 0.2))

# annotate Lowest Temperature
ax.annotate("Lowest Temperature", xy=(forecast_pd[forecast_pd['Temp'] == low_temp]['Cities'], low_temp),
            xytext=(forecast_pd[forecast_pd['Temp'] == low_temp]['Cities'], low_temp+2),
            arrowprops={"arrowstyle": "->", "color": "gray"})
# Show Plot
plt.show()
