import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title = "Bike Sharing Prediction",
    page_icon = ":bike:",
)

model = joblib.load("model_regresi_bikesharing.joblib")

st.title(":bike: Bike Sharing Prediction")
st.markdown("Aplikasi machine learning regresi untuk memprediksi jumlah penyewaan sepeda menggunakan algoritma random forest regression")

with st.container(border=True):
	season = st.pills("Season", ["Spring","Summer","Fall","Winter"], default="Spring")
	year = st.pills("Year", ["2011","2012"], default="2011")
	month = st.pills("Month", ["Jan","Feb","Mar","Apr","May","June","july","Aug","Sep","Oct","Nov","Dec"], default="Jan")
	holiday = st.pills("Holiday", ["No", "Yes"], default="No")
	weekday = st.pills("Day", ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"], default="Sunday")
	workday = st.pills("Workingday", ["No","Yes"], default="No")
	weather = st.pills("Weather", ["Clear","Cloudy","Rain"], default="Clear")
	temp = st.slider("Temperature (*C)", -8.0, 39.0, 15.0)
	humidity = st.slider("Humidity", 0.0, 100.0, 50.0)
	wind = st.slider("Windspeed", 0.0, 67.0, 30.0)

	st.write("")
	st.write("")
	if st.button("Start Prediction", type = "primary"):
		season_map = {"Spring": 1, "Summer": 2, "Fall": 3, "Winter": 4}
		year_map = {"2011": 0, "2012": 1}
		month_map = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "June": 6, "july": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
		holiday_map = {"No": 0, "Yes": 1}
		weekday_map = {"Sunday": 0, "Monday": 1, "Tuesday": 2, "Wednesday": 3, "Thursday": 4, "Friday": 5, "Saturday": 6}
		workday_map = {"No": 0, "Yes": 1}
		weather_map = {"Clear": 1, "Cloudy": 2, "Rain": 3}

		season_val = season_map[season]
		yr_val = year_map[year]
		mnth_val = month_map[month]
		holiday_val = holiday_map[holiday]
		weekday_val = weekday_map[weekday]
		workingday_val = workday_map[workday]
		weathersit_val = weather_map[weather]
		temp_val = (temp - (-8)) / (39 - (-8))
		hum_val = humidity / 100.0
		windspeed_val = wind / 67.0
		new_data = pd.DataFrame([[season_val, yr_val, mnth_val, holiday_val, weekday_val, workingday_val, weathersit_val, temp_val, hum_val, windspeed_val]], columns = ['season','year','month', 'holiday', 'weekday', 'workingday', 'weather', 'temperature', 'humidity', 'windspeed'])
		prediction = model.predict(new_data)[0]
		st.success(f'Model memprediksi jumlah sepeda disewa adalah **{prediction:.4f}**')

st.divider()

st.caption("Dibuat dengan :heart: oleh M. Irfan Nopriansyah")

