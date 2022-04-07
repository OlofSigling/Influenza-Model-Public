import pandas as pd
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import category_encoders as ce
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from feature_importance import FeatureImportance

influenza_file_path = 'Influenza_NY.csv'
NYC_data = pd.read_csv(influenza_file_path)

dropped_columns = ['Year', 'Season', 'Week Ending Date', 'County_Served_hospital', 'Service_hospital']
dropped_data = NYC_data.dropna().drop(columns=dropped_columns)

X = dropped_data#.drop(columns=['Disease'])
y = dropped_data.Disease

trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.8)

cat_columns = dropped_data.select_dtypes(include='object').drop(columns='County').columns

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

encoder= ce.BinaryEncoder(cols='County',return_df=True)
hicardinal_transformer = Pipeline(steps=[('binary', encoder)])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, cat_columns),
        ('cat2', hicardinal_transformer, 'County')
    ])

NYC_model = RandomForestClassifier(n_estimators=100, random_state=0)
NYC_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', NYC_model)])

NYC_pipeline.fit(trainX, trainy)



# nyc_model = DecisionTreeRegressor()
# nyc_model.fit(trainX, trainy)
# nyc_predictions = nyc_model.predict(testX)
# print(mean_absolute_error(testy, nyc_predictions))



# Empty columns: ['Beds_adult_facility_care', 'Beds_hospital', 'County_Served_hospital',
#        'Service_hospital', 'Discharges_Other_Hospital_intervention',
#        'Discharges_Respiratory_system_interventions',
#        'Total_Charge_Other_Hospital_intervention',
#        'Total_Charge_Respiratory_system_interventions']

# Beds_adult_facility_care                          1920
# Beds_hospital                                      960
# County_Served_hospital                             960
# Service_hospital                                   960
# Discharges_Other_Hospital_intervention           13650
# Discharges_Respiratory_system_interventions      13650
# Total_Charge_Other_Hospital_intervention         13650
# Total_Charge_Respiratory_system_interventions    13650

