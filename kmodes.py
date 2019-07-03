import findspark
findspark.init()
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *

conf = pyspark.SparkConf().setAll([('spark.executor.memory', '8g'), ('spark.driver.memory','20g'), ('spark.driver.maxResultSize','20g'), ("spark.ui.showConsoleProgress", "false")])
spark = SparkSession.builder.master('local[12]').config(conf=conf).appName('wo').getOrCreate()

import shutil
import os
import random
import re
import string
import nltk
import string
from tqdm import tqdm
from text_processor.text_processor import TextProcessor
tp = TextProcessor()
from text_processor.stopwords_loader import stopwords_loader
stopwords = stopwords_loader()
import pandas as pd
from datetime import datetime, timedelta
from dateutil.parser import parse
import numpy as np

df_conglomerate=spark.read.json('/root/fachrul/2019/data-structure/aril_result/fb-account-like-conglomerate.json')

X=np.array(df_conglomerate.select('age_range',
 'city',
 'country',
 'district',
 'gender',
 'last_education',
 'personality_archetype',
 'personality_mbti',
 'personality_ocean_dominant',
 'profession',
 'province',
 'relationship',
 'religion').collect())

syms=np.array(df_conglomerate.select('id').collect())

import numpy as np
from kmodes.kmodes import KModes
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd

km = KModes(n_clusters=8, init='Huang', n_init=8,verbose=1)
print('predict')
clusters = km.fit_predict(X)

# Print the cluster centroids
print(km.cluster_centroids_)
kmodes = km.cluster_centroids_
shape = kmodes.shape

result=[]
for i,j in tqdm(zip(km.labels_,syms)):
    result.append({'cluster':i,'id':j})

#save result clustering
dx=pd.DataFrame(result)
dx.to_csv('hasil_clustering_conglomerate.csv', index=False)


#elbow method
cost = []
for num_clusters in list(range(1,8)):
    kmx = KModes(n_clusters=num_clusters, init='Huang', n_init=5, verbose=1)
    clusters = kmx.fit_predict(X)
    cost.append(kmx.cost_)

plt.plot(cost)
plt.show()

