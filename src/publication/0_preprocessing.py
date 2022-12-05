# Standard Libraries
import pandas as pd

########################################################################################################################
print(" \nData PreProcessing \n")
########################################################################################################################

# Data selection
meter_data = ["electricity"]
weather_cols = ["airTemperature", "windSpeed", "precipDepth1HR"]
columns_considered = meter_data + weather_cols

# Path & URL definitions
path_data_save = r"C:\Users\20190285\Documents\GitHub\hierarchicallearning\io\input/"
url_root = 'https://media.githubusercontent.com/media/buds-lab/building-data-genome-project-2/master/data/'
path_meters = "meters/cleaned/"
url_path_weather = "weather/"
url_path_meta = "metadata/"
meter_file = url_root + path_meters + "electricity_cleaned.csv"


print("Data Integration \n")
meter_files = [path_data_save + meter + "_cleaned.csv" for meter in meter_data]
# Weather
weather = pd.read_csv(url_root + url_path_weather + "weather.csv", usecols=(["timestamp", "site_id"]+weather_cols))
weather["timestamp"] = pd.to_datetime(weather["timestamp"], format="%Y-%m-%d %H:%M:%S")
weather.set_index("timestamp")
# Meter
meter = pd.read_csv(meter_file, index_col="timestamp")
meter.index = pd.to_datetime(meter.index, format='%Y-%m-%d %H:%M:%S')
meter = meter.loc[:, meter.isna().mean() < .6]

# Memory clearing
meta = pd.read_csv(url_root + url_path_meta + "metadata.csv", usecols=["building_id", "site_id"]+meter_data,)
meta = meta.loc[meta["electricity"] == "Yes"]
building_id_list = meter.columns.tolist()
building_site_unique = set([site.split("_")[0] for site in building_id_list])
blg_id_per_site_dict = dict()
for key in building_site_unique:
    blg_id_per_site_dict[key] = [building_id for building_id in building_id_list if building_id.split("_")[0] == key]


weather = weather.loc[weather["site_id"].isin(building_site_unique)].copy()
del meta


# Data fusion
for site_id in blg_id_per_site_dict:
    df = pd.DataFrame(columns=["timestamp", "building_id"])
    blg_ids = blg_id_per_site_dict[site_id]

    # Format information
    df_site = meter.loc[:, blg_ids].dropna(how="all")
    df_weather = weather.loc[weather["site_id"] == site_id][weather_cols]
    df_site[weather_cols] = df_weather[weather_cols]

    # Saving data per site
    df_site.to_csv(path_data_save + site_id +'_preproc.csv', header=True, index=True)
