import requests
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import preprocessing
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 
%matplotlib inline 
sns.set(color_codes=True)

#Gathering stations for which data has to be fetched from api

mumbai_stations=['us-consulate','bandra','nmmc-airoli','vile-parle-west','colaba','kurla','worli','chhatrapati-shivaji-intl.-airport-t2','vasai-west']
delhi_stations=['','jawaharlal-nehru-stadium','mother-dairy-plant--parparganj','shaheed-sukhdev-college-of-business-studies--rohini','iti-shahdra--jhilmil-industrial-area','pooth-khurd--bawana','mundka','sonia-vihar-water-treatment-plant-djb','lodhi-road','siri-fort','pgdav-college--sriniwaspuri','crri-mathura-road','dite-okhla','dr.-karni-singh-shooting-range','major-dhyan-chand-national-stadium','punjabi-bagh','dwarka','igi-airport-terminal-3','national-institute-of-malaria-research--sector-8--dwarka','igi-airport','ito','alipur','narela','iti-jahangirpuri','burari-crossing','satyawati-college','delhi-institute-of-tool-engineering--wazirpur','north-campus','anand-vihar']
noida_stations=['sector-125','sector-116','sector-62','sector-1']
lucknow_stations=['talkatora','central-school']
bangalore_stations=['city-railway-station','saneguravahalli','peenya','btm','bwssb']
Kolkata_station=['us-consulate','victoria','rabindra-bharati-university','jadavpur','fort-william','bidhannagar','ballygunge']
patna_stations=['muradpur','rajbansi-nagar','samanpura','govt.-high-school-shikarpur','igsc-planetarium-complex']
Indore_stations=['chhoti-gwaltoli'] 
gzb_stations=['sanjay-nagar','vasundhara','indirapuram','loni']
meerut_stations=['pallavpuram-phase-2','ganga-nagar','jai-bhim-nagar']
jaipur_stations=['shastri-nagar','police-commissionerate','adarsh-nagar','vk-industrial-area-jaipur']
jodhpur_stations=['jodhpur','collectorate-jodhpur']
howrah_stations=['padmapukur','ghusuri','belur-math']
fbd_stations=['sector16a-faridabad','sector-30']
gur_stations=['nise-gwal-pahari','sector-51','teri-gram']
beng_stations=['jayanagar-5th-block','hebbal','silk-board','jayanagar-5th-block','hombegowda-nagar','bapuji-nagar','bwssb-kadabesanahalli']
pune_stations=['bhumkar-chowk','nigdi','bhosari','alandi','lohegaon','hadapsar','manjri','shivajinagar','katraj','karve-road-pune']
bul_stations=['yamunapuram']

#Certain api's have different endpoints for e.g delhi and pune

delhi_api='delhi'
mumbai_api='india/mumbai'
noida_api='india/noida'
lucknow_api='india/lucknow'
bangalore_api='india/bangalore'
Kolkata_api='india/kolkata'
patna_api='india/patna'
indore_api='india/indore'
gzb_api='india/ghaziabad'
meerut_api='india/meerut'
jaipur_api='india/jaipur'
jodhpur_api='india/jodhpur'
howrah_api='india/howrah'
fbd_api='india/faridabad'
gur_api='india/gurugram'
pune_api='pune'
beng_api='india/bengaluru'
bul_api='india/bulandshahr'


#for generating dataframe from json objects fetched from api

def generate_df(state_api,stations,City):
    df1=pd.DataFrame()
    for i in range(len(stations)):
        r=requests.get(f"https://api.waqi.info/feed/{state_api}/{stations[i]}/?token=b1962fea1af8060d8982ec2ad356fe3c330b0e2a")
    
        data=r.json()
        x=data['data']
       
        df=pd.DataFrame(x['iaqi'])
        df=df.rename( index={'v': '0'})
        df['city']=City
        df1=pd.concat([df1,df],ignore_index=True)
        
    return df1

mumbai_df=generate_df(mumbai_api,mumbai_stations,'mumbai')
delhi_df=generate_df(delhi_api,delhi_stations,'delhi')
noida_df=generate_df(noida_api,noida_stations,'noida')
lucknow_df=generate_df(lucknow_api,lucknow_stations,'lucknow')
bangalore_df=generate_df(bangalore_api,bangalore_stations,'bangalore')
kolkata_df=generate_df(Kolkata_api,Kolkata_station,'kolkata')
patna_df=generate_df(patna_api,patna_stations,'patna')
indore_df=generate_df(indore_api,Indore_stations,'indore')
gzb_df=generate_df(gzb_api,gzb_stations,'ghaziabad')
meerut_df=generate_df(meerut_api,meerut_stations,'meerut')
jaipur_df=generate_df(jaipur_api,jaipur_stations,'jaipur')
jodhpur_df=generate_df(jodhpur_api,jodhpur_stations,'jodhpur')
howrah_df=generate_df(howrah_api,howrah_stations,'kolkata')
fbd_df=generate_df(fbd_api,fbd_stations,'faridabad')
gur_df=generate_df(gur_api,gur_stations,'gurugram')
pun_df=generate_df(pune_api,pune_stations,'pune')
beng_df=generate_df(beng_api,beng_stations,'bangalore')
bul_df=generate_df(bul_api,bul_stations,'bulandsheher')

#Dataframes will be merged
combined_df=pd.concat([combined_df,mumbai_df,delhi_df,noida_df,lucknow_df,bangalore_df,kolkata_df,patna_df,indore_df,gzb_df,meerut_df,jaipur_df,jodhpur_df,howrah_df,fbd_df,gur_df,pun_df,beng_df,bul_df],ignore_index=True)


#dropping irrelevant columns
combined_df.drop(['dew','t','w','wg','wd','r','p','h'], axis=1, inplace=True)
combined_df.head(20)

#handling null values
combined_df.isnull().sum()

#Replacing null values with mean
combined_df['so2']=combined_df['so2'].fillna(combined_df['so2'].mean())
combined_df['no2']=combined_df['no2'].fillna(combined_df['no2'].mean())
combined_df['pm25']=combined_df['pm25'].fillna(combined_df['pm25'].mean())
combined_df['pm10']=combined_df['pm10'].fillna(combined_df['pm10'].mean())
combined_df['co']=combined_df['co'].fillna(combined_df['co'].mean())
combined_df['o3']=combined_df['o3'].fillna(combined_df['o3'].mean())

#checking for no more null values
combined_df.isnull().sum()

#handling duplicate rows
duplicate_rows = combined_df[combined_df.duplicated()]
print(duplicate_rows)

combined_df = combined_df.drop_duplicates()
combined_df.head(5)

#visualizing outliers
sns.boxplot(x=combined_df['no2'])
sns.boxplot(x=combined_df['so2'])
sns.boxplot(x=combined_df['pm10'])

#removing outliers

q1=combined_df.quantile(0.25)
q3=combined_df.quantile(0.75)
IQR = q3-q1
combined_df = combined_df[~((combined_df < (q1-1.5 * IQR)) |(combined_df > (q3 + 1.5 * IQR))).any(axis=1)]

#composition of examples state wise
combined_df.city.value_counts().nlargest(40).plot(kind='bar', figsize=(10,5))

#finding corelation between gases
plt.figure(figsize=(20,10))
rel= combined_df.corr()
sns.heatmap(rel,cmap='BrBG',annot=True)
print(rel)

#so2 vs states
fig, ax = plt.subplots(figsize=(20,10))
ax.scatter(combined_df['so2'], combined_df['city'])
ax.set_xlabel('so2')
ax.set_ylabel('city')
plt.show()

#no2 vs states

fig, ax = plt.subplots(figsize=(20,10))
ax.scatter(combined_df['no2'], combined_df['city'])
ax.set_xlabel('no2')
ax.set_ylabel('state')
plt.show()

#label encoding for city columns
labelencoder = LabelEncoder()
combined_df['city'] = labelencoder.fit_transform(combined_df['city'])
combined_df

#Defining x and y feature set
X = combined_df[['pm25','co','no2','o3','pm10','so2']].values
X[0:5]
y = combined_df['city'].values
y [0:5]

#normalizing data

X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X[0:5]

#creatuing train and test datasets

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


k = 3
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
neigh

yhat = neigh.predict(X_test)
yhat[0:5]

print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))
