import tushare as ts

# Fill in the stock code, data start date and end date
df = ts.get_k_data('000338', ktype='D', start='2010-06-15', end='2020-06-15')

# Save data into csv file
data_path = "./SH000338.csv"
df.to_csv(data_path)
