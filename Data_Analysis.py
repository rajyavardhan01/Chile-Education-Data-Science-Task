
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
get_ipython().magic('matplotlib inline')
import matplotlib as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)

#Reading the dataset in a dataframe
df = pd.read_csv("D:\\Assignment\\edu_chile_survey_output.csv")

df.head(10)


# In[2]:

df.describe()


# In[3]:

# The count of math test score/language test score/area/nivel/major entered for each search and raw tuition estimate for each 
# search is quite less. So these fields will not have significant impact on the analysis and hence, we can ignore them 
# and drop them later. 
# Mean(female) = 0.577 implies there are almost 58% of the selected respondants are females.


# In[4]:

# Exploratory Data Analysis

df.apply(lambda x: sum(x.isnull()),axis=0)


# In[5]:

# We can also notice that the count of searches is decreasing starting from search1.
# Now, count the number of null values in each search clicked button column.

test_search1_isnull = df['search1_clicked_button'].isnull()
test_search2_isnull = df['search2_clicked_button'].isnull()
test_search3_isnull = df['search3_clicked_button'].isnull()
test_search4_isnull = df['search4_clicked_button'].isnull()
test_search5_isnull = df['search5_clicked_button'].isnull()
test_search6_isnull = df['search6_clicked_button'].isnull()
test_search7_isnull = df['search7_clicked_button'].isnull()
test_search8_isnull = df['search8_clicked_button'].isnull()
test_search9_isnull = df['search9_clicked_button'].isnull()
test_search10_isnull = df['search10_clicked_button'].isnull()

print (sum(test_search1_isnull))
print (sum(test_search2_isnull))
print (sum(test_search3_isnull))
print (sum(test_search4_isnull))
print (sum(test_search5_isnull))
print (sum(test_search6_isnull))
print (sum(test_search7_isnull))
print (sum(test_search8_isnull))
print (sum(test_search9_isnull))
print (sum(test_search10_isnull))


# In[6]:

# The count of null values in each search button clicked is quite high and gradually increasing with each search.
# This means that there is a general trend that not all selected students are interested in searching 10 times.
# So we can have "search1_clicked_button" as the binary classifier beacuse it has the minimum number of null values.


# In[7]:

df['rbd'].value_counts()


# In[8]:

# We can see that there is not a particular secondary school ID from where significant number of selected students were selected.
# Therefore, we can ignore and drop it later.


# In[9]:

# Fill the null values in "search1_clicked_button" as "no"


# In[10]:

df['search1_clicked_button'] = df['search1_clicked_button'].fillna('no')
df['search1_clicked_button']


# In[11]:

sns.countplot(x="search1_clicked_button", data=df, palette="Greens_d")


# In[12]:

sns.countplot(x="search1_clicked_button", hue="female", data=df, palette="Greens_d")


# In[13]:

# The number of females is greater than the number of males irrespective of the search button clicked.
# Therefore, gender doesn't play a significant role in the search click as of now.


# In[14]:

sns.violinplot(x="search1_clicked_button", y="PSU_2013", hue="female", data=df, split=True);


# In[15]:

# The median 2013 language and math score for both the gender is the same range.


# In[16]:

# Distribution analyis of the 2013 language scores
df['PSU_leng_2013'].hist(bins=50)


# In[17]:

# There are no outliers noticed in the histogram


# In[18]:

# we look at box plots to understand the distribution
df.boxplot(column='PSU_leng_2013')


# In[19]:

# No outlier noticed again


# In[20]:

df.boxplot(column='PSU_leng_2013', by = 'search1_clicked_button')


# In[21]:

# No extreme values noticed


# In[22]:

# Analyzing Math score using histogram and boxplot

df['PSU_mate_2013'].hist(bins=50)


# In[23]:

df.boxplot(column='PSU_mate_2013')


# In[24]:

df.boxplot(column='PSU_mate_2013', by = 'search1_clicked_button')


# In[25]:

# Similarly, analyzing 2013 total score

df['PSU_2013'].hist(bins=50)


# In[26]:

import matplotlib.pyplot as plt1

sns.FacetGrid(df, hue="search1_clicked_button", size = 10).map(plt1.scatter, "SIMCEMath10", "SIMCELang10").add_legend()


# In[27]:

# PLotting of grade 10 math and language score is pretty much similar.
# Most of the scores in both the tests lie in the same range.


# In[28]:

# Categorical Data Analysis


# In[29]:

df['mom_educ_simce'].value_counts()


# In[30]:

df.mom_educ_simce.value_counts().plot(kind='bar')


# In[31]:

temp1 = df['mom_educ_simce'].value_counts(ascending=True)
temp2 = df.pivot_table(values='search1_clicked_button',index=['mom_educ_simce'],aggfunc=lambda x: x.map({'yes':1,'no':0}).mean())
print ('\nFrequency Table for mom education:') 
print (temp1)

print ('\n\nProbability of clicking button as per each type of moms education:') 
print (temp2)


# In[32]:

# We can see most of the moms of selected students are high school graduates.
# But there is no significant relation between the type of moms' education and student clicking the search button because the 
# percentage of students clicking the search as per the moms' education lie almost in the same range from 27% - 32%


# In[33]:

# Similar analysis of dads' education

temp3 = df['dad_educ_simce'].value_counts(ascending=True)
temp4 = df.pivot_table(values='search1_clicked_button',index=['dad_educ_simce'],aggfunc=lambda x: x.map({'yes':1,'no':0}).mean())
print ('\nFrequency Table for dad education:') 
print (temp3)

print ('\n\nProbability of clicking button as per each type of dads education:') 
print (temp4)


# In[34]:

# Again, the inference is same as the mom education's inference.


# In[35]:

# Analyzing school type

temp3 = pd.crosstab(df['schl_type'], df['search1_clicked_button'])
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)


# In[36]:

# We can see that the portion of students clicking the search button is almost the same in all types of schools.


# In[37]:

# Analyzing secondary school poverty rating

temp3 = pd.crosstab(df['rbdRating'], df['search1_clicked_button'])
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)


# In[38]:

# We can see that the portion of students clicking the search button is almost the same in all types of school ratings.


# In[39]:

# Now, building predictive model using Random Forest. As we can see that there are multiple predictors, both numerical and 
# categorical, that can interact with the classifier (search1_clicked_button). It is fast and easy to tune.


# In[40]:

# Replacing null values in the numerical predictors with their mean values in order to get the model running on the data.

df["PSU_leng_2013"].fillna(df.PSU_leng_2013.mean(), inplace=True)
df["PSU_mate_2013"].fillna(df.PSU_mate_2013.mean(), inplace=True)
df["PSU_2013"].fillna(df.PSU_2013.mean(), inplace=True)
df["SIMCEMath10"].fillna(df.SIMCEMath10.mean(), inplace=True)
df["SIMCELang10"].fillna(df.SIMCELang10.mean(), inplace=True)

# To get the model running, I am replacing all other column values with a dummy value because ultimately they will be dropped.
df["q6_math_orig"].fillna(df.q6_math_orig.mean(), inplace=True)
df["q6_lang_orig"].fillna(df.q6_lang_orig.mean(), inplace=True)
df["search1_psu_math"].fillna(df.search1_psu_math.mean(), inplace=True)
df["search2_psu_math"].fillna(df.search2_psu_math.mean(), inplace=True)
df["search3_psu_math"].fillna(df.search3_psu_math.mean(), inplace=True)
df["search3_psu_lang"].fillna(df.search3_psu_lang.mean(), inplace=True)
df["search4_psu_math"].fillna(df.search4_psu_math.mean(), inplace=True)
df["search4_psu_lang"].fillna(df.search1_psu_math.mean(), inplace=True)
df["search5_psu_math"].fillna(df.search2_psu_math.mean(), inplace=True)
df["search5_psu_lang"].fillna(df.search5_psu_lang.mean(), inplace=True)
df["search6_psu_math"].fillna(df.search6_psu_math.mean(), inplace=True)
df["search6_psu_lang"].fillna(df.search6_psu_lang.mean(), inplace=True)
df["search7_psu_math"].fillna(df.search7_psu_math.mean(), inplace=True)
df["search7_psu_lang"].fillna(df.search7_psu_math.mean(), inplace=True)
df["search8_psu_math"].fillna(df.search8_psu_math.mean(), inplace=True)
df["search8_psu_lang"].fillna(df.search8_psu_lang.mean(), inplace=True)
df["search9_psu_math"].fillna(df.search9_psu_math.mean(), inplace=True)
df["search9_psu_lang"].fillna(df.search9_psu_math.mean(), inplace=True)
df["search10_psu_math"].fillna(df.search10_psu_math.mean(), inplace=True)
df["search10_psu_lang"].fillna(df.search10_psu_lang.mean(), inplace=True)
df["rbd"].fillna(df.rbd.mean(), inplace=True)
df.describe()


# In[41]:

# We can now see that count is 38004 in all the fields except female.

df.female.value_counts().plot(kind='bar')


# In[42]:

# There are 38004 entried in total so replacing just 1 value with 0 or 1 will not make much of a difference. So, replacing the 
# null value with 0

df['female'] = df['female'].fillna(0)


# In[43]:

# Replacing with values in search1_clicked_button (yes = 1 , no = 0) to make it a binary classifier.

df['search1_clicked_button'] = df.search1_clicked_button.map(dict(yes=1, no=0))
df['search1_clicked_button']


# In[44]:

# descriptive statistics for categorical data
def describe_categorical(df):
    from IPython.display import display, HTML
    display(HTML(df[df.columns[df.dtypes == "object"]].describe().to_html()))
    
describe_categorical(df)


# In[45]:

# There are mulitple categorical fields that are not predictors and hence, need to be dropped.
# The following categorical predictors are identified during exploratory data analysis
# "mom_educ_simce", "dad_educ_simce", "schl_type", "rbdRating"

df.drop(df.columns[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97]], axis=1, inplace=True)
df


# In[46]:

# In order to run the model, we need to get rid of all the NaN

categorical_variables=['mom_educ_simce','dad_educ_simce','schl_type','rbdRating']

for variable in categorical_variables:
# fill missing data with the word "Missing"
    df[variable].fillna("Missing", inplace=True)
# creating array of dummies, update df to include them and drop the main variable
    dummies = pd.get_dummies(df[variable],prefix=variable)
    df = pd.concat([df, dummies], axis = 1)


# In[47]:

# "df" is currently our train data

# Running the Random Forest Classifier on the train data

from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

# Set the seed
np.random.seed(12)

# Initialize label encoder
label_encoder = preprocessing.LabelEncoder()

# Convert the string variables to numeric
df["mom_educ_simce"] = label_encoder.fit_transform(df["mom_educ_simce"])
df["dad_educ_simce"] = label_encoder.fit_transform(df["dad_educ_simce"])
df["schl_type"] = label_encoder.fit_transform(df["mom_educ_simce"])
df["rbdRating"] = label_encoder.fit_transform(df["rbdRating"])

# Initialize the model
rf_model = RandomForestClassifier(n_estimators=1000, # Number of trees
                                  max_features=2,    # Num features considered
                                  oob_score=True)    # Use OOB scoring*

features = ["female","PSU_leng_2013","PSU_mate_2013","PSU_2013","SIMCEMath10","SIMCELang10", "mom_educ_simce", "dad_educ_simce","schl_type","rbdRating"]

# Train the model
rf_model.fit(X=df[features],
             y=df["search1_clicked_button"])


# In[48]:

# Lets check the OOB accuracy

print("OOB accuracy: ")
print(rf_model.oob_score_)


# In[49]:

# The accuracy is approx. 66% which is not bad for a data that is a mixture of numeric and categorical predictors and
# contains large number of missing values.


# In[50]:

# Let's check the feature importance for our random forest model:

for feature, imp in zip(features, rf_model.feature_importances_):
    print(feature, imp)


# In[51]:

# We can notice that the numerical predictors, mostly the PSU 2013 scores and the grade 10 scores are more important than 
# other categorical predictors.
# This means that eduaction history plays greater role than rest of the other factors.


# In[52]:

# Read and prepare test data

df_test1 = pd.read_csv("D:\\Assignment\\edu_chile_survey_output.csv")    # Read the data

df_test1["PSU_leng_2013"].fillna(df_test1.PSU_leng_2013.mean(), inplace=True)
df_test1["PSU_mate_2013"].fillna(df_test1.PSU_mate_2013.mean(), inplace=True)
df_test1["PSU_2013"].fillna(df_test1.PSU_2013.mean(), inplace=True)
df_test1["SIMCEMath10"].fillna(df_test1.SIMCEMath10.mean(), inplace=True)
df_test1["SIMCELang10"].fillna(df_test1.SIMCELang10.mean(), inplace=True)

# to get the model running, I am replacing all other column values with a dummy value
df_test1["q6_math_orig"].fillna(df_test1.q6_math_orig.mean(), inplace=True)
df_test1["q6_lang_orig"].fillna(df_test1.q6_lang_orig.mean(), inplace=True)
df_test1["search1_psu_math"].fillna(df_test1.search1_psu_math.mean(), inplace=True)
df_test1["search2_psu_math"].fillna(df_test1.search2_psu_math.mean(), inplace=True)
df_test1["search3_psu_math"].fillna(df_test1.search3_psu_math.mean(), inplace=True)
df_test1["search3_psu_lang"].fillna(df_test1.search3_psu_lang.mean(), inplace=True)
df_test1["search4_psu_math"].fillna(df_test1.search4_psu_math.mean(), inplace=True)
df_test1["search4_psu_lang"].fillna(df_test1.search1_psu_math.mean(), inplace=True)
df_test1["search5_psu_math"].fillna(df_test1.search2_psu_math.mean(), inplace=True)
df_test1["search5_psu_lang"].fillna(df_test1.search5_psu_lang.mean(), inplace=True)
df_test1["search6_psu_math"].fillna(df_test1.search6_psu_math.mean(), inplace=True)
df_test1["search6_psu_lang"].fillna(df_test1.search6_psu_lang.mean(), inplace=True)
df_test1["search7_psu_math"].fillna(df_test1.search7_psu_math.mean(), inplace=True)
df_test1["search7_psu_lang"].fillna(df_test1.search7_psu_math.mean(), inplace=True)
df_test1["search8_psu_math"].fillna(df_test1.search8_psu_math.mean(), inplace=True)
df_test1["search8_psu_lang"].fillna(df_test1.search8_psu_lang.mean(), inplace=True)
df_test1["search9_psu_math"].fillna(df_test1.search9_psu_math.mean(), inplace=True)
df_test1["search9_psu_lang"].fillna(df_test1.search9_psu_math.mean(), inplace=True)
df_test1["search10_psu_math"].fillna(df_test1.search10_psu_math.mean(), inplace=True)
df_test1["search10_psu_lang"].fillna(df_test1.search10_psu_lang.mean(), inplace=True)
df_test1["rbd"].fillna(df_test1.rbd.mean(), inplace=True)

# since most of them are 1, to try to make data balanced we will replace the 1 empty cell as 0
df_test1['female'] = df_test1['female'].fillna(0)

# so many null valules , replace it with no first and then yes = 1 , no = 0
df_test1['search1_clicked_button'] = df_test1['search1_clicked_button'].fillna('no')
df_test1['search1_clicked_button'] = df_test1.search1_clicked_button.map(dict(yes=1, no=0))

df_test1.drop(df_test1.columns[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96]], axis=1, inplace=True)

# there are so many NaN, need to get rid of them

categorical_variables1=['mom_educ_simce','dad_educ_simce','schl_type','rbdRating']

for variable in categorical_variables1:
# fill missing data with the word "Missing"
    df_test1[variable].fillna("Missing", inplace=True)
# creating array of dummies, update df_test1 to include them and drop the main variable
    dummies1 = pd.get_dummies(df_test1[variable],prefix=variable)
    df_test11 = pd.concat([df_test1, dummies1], axis = 1)
    
# Convert some variables to numeric
df_test1["mom_educ_simce"] = label_encoder.fit_transform(df_test1["mom_educ_simce"])
df_test1["dad_educ_simce"] = label_encoder.fit_transform(df_test1["dad_educ_simce"])
df_test1["schl_type"] = label_encoder.fit_transform(df_test1["mom_educ_simce"])
df_test1["rbdRating"] = label_encoder.fit_transform(df_test1["rbdRating"])

# Make test set predictions
test_preds = rf_model.predict(X= df_test1[features])


# In[54]:

# We can see the predicted values in an array

test_preds


# In[55]:

# Analysing the performance of the model using Confusion Matrix

from sklearn.metrics import confusion_matrix
confusion_matrix(df['search1_clicked_button'], test_preds)


# In[56]:

# Total number of True Negatives and True positives = 26139+11463 = 37602
# Total number of False Negatives and False positives = 74+328 = 402
# Total number of observations = 38004

# Accuracy = Total number of True Negatives and True positives/Total number of observations

accuracy = 37602/38004
accuracy


# In[57]:

# The accuracy in the testing data has jumped up to almost 99% which is incredible given the type of dataset.


# In[ ]:



