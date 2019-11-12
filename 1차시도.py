#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pyspark


# In[2]:


from pyspark import SparkContext


# In[3]:


from pyspark import SparkFiles


# In[4]:


from pyspark.sql import Row


# In[5]:


from pyspark.sql import SQLContext


# In[6]:


from pyspark.ml.feature import VectorAssembler


# In[7]:


from pyspark.ml.regression import LinearRegression


# In[8]:


from pyspark.ml.evaluation import RegressionEvaluator


# In[9]:


from pyspark.ml.feature import OneHotEncoder, StringIndexer


# In[134]:


from pyspark.sql.functions import col


# In[10]:


from pyspark.sql.types import *


# In[102]:


devColumns = [
    StructField("inst_id",IntegerType()),
    StructField("OC",FloatType()),
    #StructField("OC",StringType()),
    StructField("sido",StringType()),
    StructField("sgg",LongType()),
    StructField("openDate",StringType()),
    StructField("bedCound",LongType()),
    StructField("instkind",StringType()),
    
    StructField("revenue1",LongType()),
    StructField("salescost1",LongType()),
    StructField("salary1",LongType()),
    StructField("noi1",LongType()),
    StructField("noe1",LongType()),
    StructField("interest1",LongType()),
    StructField("ctax1",LongType()),
    StructField("profit1",LongType()),
    StructField("liquidAsset1",LongType()),
    StructField("quickAsset1",LongType()),
    StructField("receivableS1",LongType()),
    StructField("inventoryAsset1",LongType()),
    StructField("nonCAsset1",LongType()),
    StructField("tanAsset1",LongType()),
    StructField("OnonCAsset1",LongType()),
    StructField("receivableL1",LongType()),
    StructField("debt1",LongType()),
    StructField("liquidLiabilities1",LongType()),
    StructField("shortLoan1",LongType()),
    StructField("NCLiabilites1",LongType()),
    StructField("longLoan1",LongType()),
    StructField("netAsset1",LongType()),
    StructField("surplus1",LongType()),
    
    StructField("revenue2",LongType()),
    StructField("salescost2",LongType()),
    StructField("salary2",LongType()),
    StructField("noi2",LongType()),
    StructField("noe2",LongType()),
    StructField("interest2",LongType()),
    StructField("ctax2",LongType()),
    StructField("profit2",LongType()),
    StructField("liquidAsset2",LongType()),
    StructField("quickAsset2",LongType()),
    StructField("receivableS2",LongType()),
    StructField("inventoryAsset2",LongType()),
    StructField("nonCAsset2",LongType()),
    StructField("tanAsset2",LongType()),
    StructField("OnonCAsset2",LongType()),
    StructField("receivableL2",LongType()),
    StructField("debt2",LongType()),
    StructField("liquidLiabilities2",LongType()),
    StructField("shortLoan2",LongType()),
    StructField("NCLiabilites2",LongType()),
    StructField("longLoan2",LongType()),
    StructField("netAsset2",LongType()),
    StructField("surplus2",LongType()),
    
    StructField("employee1",LongType()),
    StructField("employee2",LongType()),

    StructField("ownerChange",StringType())
]


#TimeStamp 를 String으로 바꿈


# In[103]:


devSchema = StructType(devColumns)


# In[104]:


sc = SparkContext.getOrCreate();


# In[105]:


sqlContext = SQLContext(sc)


# In[106]:


#df = sqlContext.read.csv("train.csv", quote="", header=True, inferSchema = True)


# In[173]:


df = sqlContext.read.schema(devSchema).csv("train.csv",header=True)
#df = sqlContext.read.csv("train.csv", header=True)

#df = df.where((df.head).isNotNull())

df = df.where(col("revenue1").isNotNull())


# In[174]:


df.printSchema()


# In[175]:


df.take(3)


# In[176]:


df.show()


# In[177]:


#oneHotEncoding
#stringIndexer = StringIndexer(inputCol="OC", outputCol="OCIndex")
#model = stringIndexer.fit(df)
#indexed = model.transform(df)
#encoder = OneHotEncoder(dropLast=False, inputCol="OCIndex", outputCol="OCVec")
#df = encoder.transform(indexed)


stringIndexer = StringIndexer(inputCol="sido", outputCol="sidoIndex")
model = stringIndexer.fit(df)
indexed = model.transform(df)
encoder = OneHotEncoder(dropLast=False, inputCol="sidoIndex", outputCol="sidoVec")
df = encoder.transform(indexed)


stringIndexer = StringIndexer(inputCol="instkind", outputCol="instkindIndex")
model = stringIndexer.fit(df)
indexed = model.transform(df)
encoder = OneHotEncoder(dropLast=False, inputCol="instkindIndex", outputCol="instkindVec")
df = encoder.transform(indexed)


stringIndexer = StringIndexer(inputCol="ownerChange", outputCol="ownerChangeIndex")
model = stringIndexer.fit(df)
indexed = model.transform(df)
encoder = OneHotEncoder(dropLast=False, inputCol="ownerChangeIndex", outputCol="ownerChangeVec")
df = encoder.transform(indexed)


#'OC', 'OCIndex', 'OCVec' 임시로삭제
columns_to_drop = ['OCIndex', 'sidoIndex', 'instkindIndex', 'ownerChangeIndex', 
                    'sido', 'instkind', 'ownerChange', 'openDate', 
                    'sidoVec', 'instkindVec', 'ownerChangeVec', 'inst_id']
df = df.drop(*columns_to_drop)


# In[178]:


#selectDf = df.select("id", "vh_id", "route_id", "now_latitude"
                   # ,"now_longitude", "next_latitude", "next_longitude",
                   # "distance", "next_arrive_time")
selectDf = df


# In[179]:


selectDf.printSchema()


# In[180]:


selectDf.describe().toPandas().transpose()


# In[181]:


#selectDf.show()
print(selectDf.schema.names)


# In[182]:


#now_arrive_time이랑 date는 일단인 빼고봄(오류때문)
vectorAssembler = VectorAssembler(inputCols = selectDf.schema.names,
                                 outputCol = 'features')


# In[183]:


#Skip Null data
v_data = vectorAssembler.transform(selectDf)


# In[184]:


#next_arrive_time이 타겟이므로
v_choice_data =  v_data.select(['features', 'OC'])


# In[185]:


v_choice_data.show()


# In[206]:


(train, test) = v_choice_data.randomSplit([0.8,0.2])


# In[207]:


# maxIter = 10
# regParam = 0.3
# elasticNetParam = 0.8
lr = LinearRegression(featuresCol = 'features', labelCol='OC',
                     maxIter = 10,
                     regParam = 0.01,
                     elasticNetParam = 0.02)

lr_model = lr.fit(train)
print("Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))


# In[208]:


trainingSummary = lr_model.summary
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)


# In[209]:


train.describe().show()


# In[213]:


#예측값 표시(R2가 정확도를 의미함 0.99 --> 99% 정확도)
lr_predictions = lr_model.transform(test)
lr_predictions.select("prediction","OC","features").show(100)

lr_evaluator = RegressionEvaluator(predictionCol="prediction",                  labelCol="OC",metricName="r2")
print("R Squared (R2) on test data = %g" % lr_evaluator.evaluate(lr_predictions))


# In[211]:


#RMSE 결과값 도출
test_result = lr_model.evaluate(test)
print("Root Mean Squared Error (RMSE) on test data = %g" % test_result.rootMeanSquaredError)


# In[212]:


#트레이닝 모델 요약
print("numIterations: %d" % trainingSummary.totalIterations)
print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
trainingSummary.residuals.show()


# In[ ]:




