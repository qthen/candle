from data.Kepler import KeplerPeriodSpacing

kepler = KeplerPeriodSpacing()
data = kepler.get_data(max_number_of_stars = 50)
print(data)