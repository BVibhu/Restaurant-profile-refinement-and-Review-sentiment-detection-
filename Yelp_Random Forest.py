#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 21:54:31 2022

@author: alicia
"""
import pandas as pd
import numpy as np

business_attributes = pd.read_csv("yelp_business_attributes.csv",engine="python")
business_hours = pd.read_csv("yelp_business_hours.csv",engine="python")
business = pd.read_csv("yelp_business.csv",engine="python")
checkin = pd.read_csv("yelp_checkin.csv",engine="python")
review = pd.read_csv("yelp_review.csv",engine="python")
tip = pd.read_csv("yelp_tip.csv",engine="python")
user = pd.read_csv("yelp_user.csv",engine="python")


# Get the number of stars (our target variable) of each business
df_rating = business[['business_id', 'stars', 'categories']]
# Combine the above with the attributes
df1 = pd.merge(df_rating, business_hours, on = 'business_id')

restaurant=df1[df1['categories'].str.contains("restaurant|food")==True]
restaurant_2=pd.merge(restaurant,business_attributes,on='business_id')
#df1=df1.drop(labels='categories',axis=1)
#df1=df1.drop(labels='Unnamed: 0',axis=1)

#define values for predictor and response variables
restaurant_2=restaurant_2.replace(to_replace={'Na':0,'True':1,'False':0,'beer_and_wine':1,
                                              'full_bar':1,'no':0,'none':0,'outdoor':1,'yes':1})
#X = restaurant_2[['HasTV','WheelchairAccessible','BikeParking','GoodForKids']]
#y = restaurant_2['stars']

#no dummy data to transform
#X_dummy = pd.get_dummies(X, columns=['weekdays','data_channel'])

#transform into categorical data
#Encode target labels with value between 0 and n_classes-1.
star_df=restaurant_2['stars']
star_label = []
for star in star_df: #assign the percentile category to the new column
    if star <= 1:
        star_label.append('Very Poor')
    elif star > 1 and star <= 2:
        star_label.append('Poor')
    elif star > 2 and star <= 3:
        star_label.append('Average')
    elif star > 3 and star <= 4:
        star_label.append('Good')
    elif star > 4:
        star_label.append('Very Good')

#combine the new column to the df
restaurant_3 = pd.concat([restaurant_2, pd.DataFrame(star_label, columns=['star_label'])], axis=1)
restaurant_3['stars']=restaurant_3['stars'].astype('category') #make sure identified as categorical

#1. Parking related
X1 = restaurant_3[['BusinessParking_garage','BusinessParking_street',
                 'BusinessParking_lot','BusinessParking_valet','BusinessParking_validated','BikeParking']]

#2. Meal attributes
X2 = restaurant_3[['GoodForMeal_breakfast','GoodForMeal_brunch','GoodForMeal_lunch','GoodForMeal_dinner','GoodForMeal_dessert','GoodForMeal_latenight']]

#3. Ambiance
X3 = restaurant_3[['Ambience_romantic','Ambience_intimate','Ambience_classy','Ambience_hipster',
                   'Ambience_divey','Ambience_touristy','Ambience_trendy','Ambience_upscale','Ambience_casual']]

#4. Dietary Restrictions
X4 = restaurant_3[['DietaryRestrictions_kosher','DietaryRestrictions_gluten-free','DietaryRestrictions_vegan']]

#5. Other attrbutes
X5 = restaurant_3[['DogsAllowed','WiFi','RestaurantsTakeOut','DriveThru','OutdoorSeating','BusinessAcceptsCreditCards','ByAppointmentOnly','RestaurantsTableService',
                   'Smoking','NoiseLevel','RestaurantsDelivery','HasTV','WheelchairAccessible','GoodForKids','Alcohol']]

y = restaurant_3['star_label']


#from sklearn.preprocessing import LabelEncoder 
#Encode target labels with value between 0 and n_classes-1.
#labelEn = LabelEncoder()
#encoded_labels = labelEn.fit_transform(y.values)


#RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier #random forest doing featyre selection
rf1=RandomForestClassifier(random_state=(5)) 
#1. Parking related
model1=rf1.fit(X1,y)
model1.feature_importances_
pd.DataFrame(list(zip(X1.columns,model1.feature_importances_)), columns = ['predictor','feature importance']).sort_values(by='feature importance',ascending=False)

#2. Meal attributes
rf2=RandomForestClassifier(random_state=(5)) 
model2=rf2.fit(X2,y)
model2.feature_importances_
pd.DataFrame(list(zip(X2.columns,model2.feature_importances_)), columns = ['predictor','feature importance']).sort_values(by='feature importance',ascending=False)

#3. Ambiance
#all in 0, should avoid this analysis
rf3=RandomForestClassifier(random_state=(5)) 
model3=rf3.fit(X3,y)
model3.feature_importances_
pd.DataFrame(list(zip(X3.columns,model3.feature_importances_)), columns = ['predictor','feature importance']).sort_values(by='feature importance',ascending=False)

#4. Dietary Restrictions
#all in 0, should avoid this analysis
rf4=RandomForestClassifier(random_state=(5)) 
model4=rf4.fit(X3,y)
model4.feature_importances_
pd.DataFrame(list(zip(X4.columns,model4.feature_importances_)), columns = ['predictor','feature importance']).sort_values(by='feature importance',ascending=False)

#5. Other attrbutes
rf5=RandomForestClassifier(random_state=(5)) 
model5=rf4.fit(X5,y)
model5.feature_importances_
pd.DataFrame(list(zip(X5.columns,model5.feature_importances_)), columns = ['predictor','feature importance']).sort_values(by='feature importance',ascending=False)


