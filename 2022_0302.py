#!/usr/bin/env python
# coding: utf-8

# In[61]:


import os
print(os.path.abspath('.'))


# ## 第1步：导入数据分析库pandas，数据可视化库matplotlib
#  `%matplotlib inline`是Ipython的魔法函数，其作用是使matplotlib绘制的图像嵌入在juptyer notebook的单元格里

# In[62]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## 第2步：导入数据集，查看数据集

# In[63]:


dataset = pd.read_csv('./studentscores.csv')
dataset.head()


# In[64]:


dataset.shape


# In[65]:


dataset.columns


# In[66]:


dataset.info()


# In[67]:


dataset.describe()


# ## 第3步：提取特征
# ### 提取特征：学习时间 提取标签：学习成绩

# In[68]:


feature_columns = ['Hours']
label_column = ['Scores']


# In[69]:


features = dataset[feature_columns]
label = dataset[label_column]


# In[70]:


features.head()


# In[71]:


label.head()


# In[72]:


type(features)


# In[73]:


X = features.values


# In[74]:


Y = label.values


# In[75]:


X


# In[76]:


X.shape


# ## 第四步：建立模型
# ### 拆分数据，四分之三的数据作为训练集，四分之一的数据作为测试集

# In[77]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 1/4, random_state = 0) 


# ### 用训练集的数据进行训练

# In[78]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)


# ### 对测试集进行预测

# In[79]:


Y_pred = regressor.predict(X_test)


# In[80]:


X_test


# In[81]:


Y_pred


# ## 可视化

# In[82]:


# 散点图：红色点表示训练集的点
plt.scatter(X_train , Y_train, color = 'red')
# 线图：蓝色线表示由训练集训练出的线性回归模型
plt.plot(X_train , regressor.predict(X_train), color ='blue')
plt.show()


# In[84]:


# 散点图：红色点表示测试集的点
plt.scatter(X_test , Y_test, color = 'red')
# 线图：蓝色线表示对测试集进行预测的结果
plt.plot(X_test , regressor.predict(X_test), color ='blue')
plt.show()

