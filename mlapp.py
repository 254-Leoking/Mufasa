import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import os

# Define the path to the uploaded ZIP file
zip_file_path = 'C:/Users/user/Downloads/stack-overflow-developer-survey-2020.zip'
extract_dir = '/mnt/data/stack_overflow_survey'

# Extract the ZIP file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

# Load the CSV file into a pandas DataFrame
csv_file_path = os.path.join(extract_dir, 'survey_results_public.csv')
df = pd.read_csv(csv_file_path)

df = df[["Country", "EdLevel", "YearsCodePro", "Employment", "ConvertedComp"]]
df = df.rename({"ConvertedComp": "Salary"}, axis=1)

df = df[df["Salary"].notnull()]

df = df.dropna()

df = df[df["Employment"] == "Employed full-time"]
df = df.drop("Employment", axis=1)
df.info()

df['Country'].value_counts()
print(df['Country'].value_counts())


def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Other'
    return categorical_map


country_map = shorten_categories(df.Country.value_counts(), 400)
df['Country'] = df['Country'].map(country_map)


fig, ax = plt.subplots(1, 1, figsize=(12, 7))
df.boxplot('Salary', 'Country', ax=ax)
plt.suptitle('Salary (US$) v Country')
plt.title('')
plt.ylabel('Salary')
plt.xticks(rotation=90)
plt.show()


df = df[df['Salary'] <= 250000]
df = df[df['Salary'] >= 10000]
df = df[df['Salary'] != 'Other']


fig, ax = plt.subplots(1, 1, figsize=(12, 7))
df.boxplot('Salary', 'Country', ax=ax)
plt.suptitle('Salary (US$) v Country')
plt.title('')
plt.ylabel('Salary')
plt.xticks(rotation=90)
plt.show()


df["YearsCodePro"].unique()


def clean_experience(x):
    if x == 'More than 50 years':
        return 50
    if x == 'Less than 1 year':
        return 0.5
    return float(x)


df['YearsCodePro'] = df['YearsCodePro'].apply(clean_experience)

print(df['EdLevel'].unique())


def clean_education(x):
    if "Bachelor’s degree" in x or "Bachelor's degree" in x:
        return "Bachelor's degree"
    if "Master’s degree" in x or "Bachelor's degree" in x:
        return "Master's degree"
    if "Professional degree" in x or "Other doctoral degree" in x:
        return "Post grad"
    return "Less than a Bachelors"


df['EdLevel'] = df['EdLevel'].apply(clean_education)


print(df['EdLevel'].unique())

from sklearn.preprocessing import LabelEncoder
le_education = LabelEncoder()
df['EdLevel'] = le_education.fit_transform(df['EdLevel'])

print(df['EdLevel'].unique())

le_country = LabelEncoder()
df['Country'] = le_country.fit_transform(df['Country'])
print(df['Country'].unique())

x = df.drop('Salary', axis=1)
y = df['Salary']


from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(x, y.values)

LinearRegression()

y_pred = linear_reg.predict(x)

from sklearn.metrics import mean_squared_error
import numpy as np
error = np.sqrt(mean_squared_error(y, y_pred))
print(error)


from sklearn.tree import DecisionTreeRegressor
dec_tree_reg = DecisionTreeRegressor(random_state=0)
dec_tree_reg.fit(x, y.values)

y_pred = dec_tree_reg.predict(x)

error = np.sqrt(mean_squared_error(y, y_pred))
print("${0:,.2f}".format(error))

from sklearn.ensemble import RandomForestRegressor
random_forest = RandomForestRegressor(random_state=0)
random_forest.fit(x, y.values)
y_pred = random_forest.predict(x)
error = np.sqrt(mean_squared_error(y, y_pred))
print("${0:,.2f}".format(error))


from sklearn.model_selection import GridSearchCV

max_depth = [None, 2, 4, 6, 8, 10, 12]
parameters = {'max_depth': max_depth}

regressor = DecisionTreeRegressor(random_state=0)
gs = GridSearchCV(regressor, parameters, scoring='neg_mean_squared_error')
gs.fit(x, y.values)

regressor = gs.best_estimator_
y_pred = regressor.predict(x)
error = np.sqrt(mean_squared_error(y, y_pred))
print("${0:,.2f}".format(error))


print(x)


import numpy as np


x = np.array([["United States", "Master's degree", 15]])


x[:, 0] = le_country.transform(x[:, 0])
x[:, 1] = le_education.transform(x[:, 1])

x = x.astype(float)

print(x)

y_pred = regressor.predict(x)
print(y_pred)


import pickle

data = {"model": regressor, "le_country": le_country, "le_education": le_education}
with open('saved_steps.pkl', 'wb') as file:
    pickle.dump(data, file)


with open('saved_steps.pkl', 'rb') as file:
    data = pickle.load(file)

regressor_loaded = data['model']
le_country = data['le_country']
le_education = data['le_education']

y_pred = regressor_loaded.predict(x)
print(y_pred)



























