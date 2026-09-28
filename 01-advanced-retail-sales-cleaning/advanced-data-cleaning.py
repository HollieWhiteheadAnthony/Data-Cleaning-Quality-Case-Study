"""
Advanced Data Cleaning
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


database = pd.read_csv('retail_store_sales.csv')
df = pd.DataFrame(database)

#data inspection - taking note for next steps
print(df.head())
print()
print(df.tail())
print(df.describe())
print()
print(df.info())
print()
print(df.isnull().mean() * 100) #prefer this method, as it shows what percent of my data is null
print()
print("Number of duplicates found:\n",df.duplicated().sum())
print()


# I wanted to check multiple columns for unique values, so I thought it would be more efficient to use a for loop
for col in ['Payment Method', 'Location', 'Discount Applied', 'Category', 'Item']: # naming the columns I want to loop through
    print(f"{col}:") # prints column name
    print("Unique values for:", df[col].unique())# prints values for each column
    print()
    print("Value Counts for:", df[col].value_counts()) # prints how many times the value appears
    print()


# Changing incorrect data types
df['Transaction Date'] = pd.to_datetime(df['Transaction Date'],errors='coerce')
print("Data Type for Transaction Date:",df['Transaction Date'].dtype)
print()
print("Number of nulls in Transaction Date Column are now:\n",df['Transaction Date'].isnull().sum())
print()


"""
Investigating the high null values in Discount Applied Column
Wanted to be sure it was random and not tied to anything specific
made notes as to my steps for future reference.
"""
# 1. Grouping Discount Applied by Payment Method & Location
# 2. Calculating the nulls to percentage per group for easier reading
# 3. Resetting the index, because when we group by it changes the index and naming the column Null%
# 4. Printing the results in descending order so  we can see highest to lowest null amounts
null_check_discount = df.groupby(['Payment Method', 'Location'])['Discount Applied']\
                .apply(lambda x: x.isnull().mean()*100) \
                .reset_index(name='Null %')
print(null_check_discount.sort_values(by=['Null %'], ascending=False))


# Null percentage found to be uniform within my groups, chose to change to 'False'
df['Discount Applied'] = df['Discount Applied'].fillna(False)
print()
print("Number of nulls now:\n",df['Discount Applied'].isnull().sum())
print()
print("Number of unique values for Discount Applied column now:\n",df['Discount Applied'].unique())
print(df['Discount Applied'].value_counts())
print()

"""
Investigating a high amount of null values for item column grouping 
individually by location and payment method
"""
null_check_item = df.groupby(['Category'])['Item']\
                    .apply(lambda x: x.isnull().mean()*100) \
                    .reset_index(name='Null %')
print(null_check_item.sort_values(by=['Null %'], ascending=False))
print()

null_check_item = df.groupby(['Location'])['Item']\
                    .apply(lambda x: x.isnull().mean()*100) \
                    .reset_index(name='Null %')
print(null_check_item.sort_values(by=['Null %'], ascending=False))
print()

null_check_item = df.groupby(['Payment Method'])['Item']\
                    .apply(lambda x: x.isnull().mean()*100) \
                    .reset_index(name='Null %')
print(null_check_item.sort_values(by=['Null %'], ascending=False))
print()


"""
Nulls appear to be random for Item column, will now fill them.
KNN was not a good choice here where its categorical type data.
I assumed the concept was the same for the other models we learned
and chose to use the decision tree model instead and enlisted guidance
from the web/AI. I chose to do this more for practice/experience.
"""

from sklearn.tree import DecisionTreeClassifier  # importing decision tree
# splitting my data
train_df = df[df['Item'].notnull()]  # what model will learn from
test_df = df[df['Item'].isnull()]    # what model needs to predict

x_train = train_df[['Price Per Unit', 'Quantity', 'Total Spent']] #values to use to make predictions from
y_train = train_df['Item']          # values I want predicted


model = DecisionTreeClassifier() # initializing my model/creating my tool to learn the pattern
model.fit(x_train, y_train)      # model is learning

# preventing errors (no missing values-skip prediction) / (missing values exist-predict)
# checking to see if there is something to predict
if not test_df.empty:
    x_test = test_df[['Price Per Unit', 'Quantity', 'Total Spent']] # preparing test data
    df.loc[df['Item'].isnull(), 'Item'] = model.predict(x_test)     # predict the missing values
else:                                                               # no missing values....
    print("No missing Item values found.")
print()
print(df['Item'].isnull().sum()) # validating
print(df['Item'].value_counts())   # helping to detect bias results
print(df['Total Spent'].isnull().sum())  # ensuring nothing happened to other columns
print()


 # Imputing for nulls in price per unit and quantity columns
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5) # creating the imputer, when missing value, looks at 5 similar rows to estimate its value
df[['Price Per Unit', 'Quantity']] = imputer.fit_transform(df[['Price Per Unit', 'Quantity']])
# above we are saying use these columns to learn from, understand how they relate(fit) and fill in the missing values(transform)
print("Amount of nulls for Price Per Unit now:\n", df['Price Per Unit'].isnull().sum())
print()
print("Amount of nulls for Quantity now:\n", df['Quantity'].isnull().sum())
print()


# Calculating the total spent column null values now that we have imputed the
# values for quantity and price per unit
df['Total Spent'] = df['Price Per Unit'] * df['Quantity']
print("Amount of nulls for Total Spent now:\n", df['Price Per Unit'].isnull().sum())
print()


# Checking for outliers IQR method
def outliers_iqr(df_copy, column):
    data = df_copy[column]
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data > upper_bound) | (data < lower_bound)]
    return outliers

price_outliers = outliers_iqr(df,'Quantity')
print(f"Number of outliers for Price Per Unit are:{len(price_outliers)}")
print("The outliers are:",price_outliers)
print()

quantity_outliers = outliers_iqr(df,'Quantity')
print(f"Number of outliers for Quantity are:{len(quantity_outliers)}")
print("The outliers are:",quantity_outliers)
print()

total_spent_outliers = outliers_iqr(df,'Total Spent')
print(f"Number of outliers for Total Spent are:{len(total_spent_outliers)}")
print("The outliers are:",total_spent_outliers)
print()


# Applying normalization
from sklearn.preprocessing import MinMaxScaler

columns_to_normalize = ['Price Per Unit', 'Quantity', 'Total Spent'] # naming columns I want to normalize
scaler = MinMaxScaler()
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
print("Before normalization:")
print(database[columns_to_normalize].head())
print()
print("After normalization:")
print(df[columns_to_normalize].head())
print()


# creating a mini table to show the changes in our skewness before and after handling nulls
skew_compare = pd.DataFrame({
    'Before cleaning': database[['Price Per Unit', 'Quantity', 'Total Spent']].skew(),
    'After cleaning': df[['Price Per Unit', 'Quantity', 'Total Spent']].skew()})
print(skew_compare)
print()


"""
Creating multiple box plots for checking outliers
with guidance
"""

columns = ['Price Per Unit', 'Quantity', 'Total Spent'] # naming the columns I want box plots for
plt.figure(figsize=(12,4)) # creating a canvas to plot on 12" wide by 4" high

for i, col in enumerate(columns,1): # iterating between the columns to place them in positions 1,2 & 3.
    plt.subplot(1,3,i) # creating the positions - 1 row, 3 columns, place column here.
    plt.boxplot(df[col]) # plotting the data from each column
    plt.title(col) # labeling each column
plt.tight_layout() # spacing, prevents the box plots from overlapping
plt.show()

# Showing the information after cleaning

print(df.info())
print()
print(df.isnull().mean() * 100)
print()
print(df.describe())

df.to_csv('cleaned_retail_store_sales.csv', index=False)


