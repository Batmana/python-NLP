import pandas as pd
# DataFrames 和 Series 是主要的两个数据结构，被广泛用于数据分析。Series是单维索引数组，DataFrame 是具有列级和行级索引单表格数据结构
series_1 = pd.Series([2, 9, 0, 1])
print(series_1.values)
print(series_1.index)

series_1.index= ['a', 'b', 'c', 'd']
print(series_1['d'])

# Creating dataFrame using pandas
class_data = {
    "Names": ['John', 'Ryan', 'Emily'],
    "Standard": [7, 5, 8],
    "Subject": ['Englisg', 'Mathematics', 'Science']
}

class_df = pd.DataFrame(class_data,
                        index=['Student1', 'Student2', 'Student3'],
                        columns=['Names', 'Standard', 'Subject'])
print(class_df)
print(class_df.Names)
# Add new entry to the dataframe
import numpy as np
# Pandas 1.0.0 版本后移除了ix属性
class_df.loc['Student4'] = ['Robin', np.nan, 'History']
print(class_df.T)
print("---------")
print(class_df.sort_values(by='Standard'))

# Adding one more column to the dataframe as Series objecct
col_entry = pd.Series(
    ['A', 'B', 'A+', 'C'],
    index=['Student1', 'Student2', 'Student3', 'Student4']
)
class_df['Grade'] = col_entry
print('-------')
print(class_df)

# Filling the misssing entries in the dataframe, inplace
# 缺失数据补充
class_df.fillna(10, inplace=True)
print('-------')
print(class_df)

# concatenation of 2 dataframes
student_ages = pd.DataFrame(
    data={'Age': [13, 10, 15, 18]},
    index=['Student1', 'Student2', 'Student3', 'Student4']
)
print('------')
print(student_ages)

class_data = pd.concat([class_df, student_ages], axis=1)
print(class_data)

# map函数
class_data['Subject'] = class_data['Subject'].map(lambda x: x + 'sub')
print('--------')
print(class_data['Subject'])

def age_add(x):
    return x + 1
print('------Old Value------')
print(class_data['Age'])
print('-------New Value-----')
print(class_data['Age'].apply(age_add))

class_data['Grade'] = class_data['Grade'].astype('category')
print(class_data.Grade.dtypes)

# Pandas库提供的函数中，合并函数(conccat,merge, append)以及groupby 和 pivot_table函数
