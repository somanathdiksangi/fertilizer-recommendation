import pandas as pd

# Mapping categorical data to numbers
soil_color_map = {'Black': 1, 'Red': 2, 'Medium Brown': 3, 'Dark Brown': 4, 'Light Brown': 5, 'Reddish Brown': 6}
district_map = {'Kolhapur': 1, 'Solapur': 2, 'Satara': 3, 'Sangli': 4, 'Pune': 5}
crop_mapping = {'Sugarcane': 1, 'Jowar': 2, 'Cotton': 3, 'Rice': 4, 'Wheat': 5}

def preprocess_input(data):
    df = pd.DataFrame([data])
    df['District_Name'] = df['District_Name'].map(district_map)
    df['Soil_color'] = df['Soil_color'].map(soil_color_map)
    df['Crop'] = df['Crop'].map(crop_mapping)
    return df
