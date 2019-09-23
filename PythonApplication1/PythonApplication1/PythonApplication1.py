import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

aisles = pd.read_csv(r'F:\GroceryPrediction\aisles.csv')
departments = pd.read_csv('F:\GroceryPrediction\departments.csv')
orderprior = pd.read_csv('F:\GroceryPrediction\order_products__prior.csv')
ordertrain=pd.read_csv('F:\GroceryPrediction\order_products__train.csv')
orders = pd.read_csv('F:\GroceryPrediction\orders.csv')
products = pd.read_csv('F:\GroceryPrediction\products.csv')
order_prior = pd.merge(orderprior,orders,on=['order_id','order_id'])
order_prior = order_prior.sort_values(by=['user_id','order_id'])

_mt = pd.merge(orderprior,products, on = ['product_id','product_id'])
_mt = pd.merge(_mt,orders,on=['order_id','order_id'])
mt = pd.merge(_mt,aisles,on=['aisle_id','aisle_id'])
mt.head(10)


cust_prod = pd.crosstab(mt['user_id'], mt['aisle'])


pca = PCA(n_components= 10 )
pca.fit(cust_prod)
pca_samples = pca.transform(cust_prod)

ps = pd.DataFrame(pca_samples)

tocluster = pd.DataFrame(ps[[3,1]])
#print (tocluster.shape)
#print (tocluster.head())

fig = plt.figure(figsize=(8,8))
plt.plot(tocluster[3], tocluster[1], 'o', markersize=2, color='blue', alpha=0.5, label='class1')

plt.xlabel('x_values')
plt.ylabel('y_values')
plt.legend()
plt.show()



clusterer = KMeans(n_clusters=6,random_state=10).fit(tocluster)
centers = clusterer.cluster_centers_
c_preds = clusterer.predict(tocluster)
#print(centers)



fig = plt.figure(figsize=(8,8))
colors = ['orange','blue','purple','green','cyan','black']
colored = [colors[k] for k in c_preds]
#print (colored[0:10])
print(tocluster)
plt.scatter(tocluster[3],tocluster[1],  color = colored)
for ci,c in enumerate(centers):
    plt.plot(c[0], c[1], 'o', markersize=8, color='red', alpha=0.9, label=str(ci))

plt.xlabel('x_values')
plt.ylabel('y_values')
plt.legend()
plt.show()



clust_prod = cust_prod.copy()
clust_prod['cluster'] = c_preds


#print (clust_prod.shape)
f,arr = plt.subplots(3,2,sharex=True,figsize=(15,15))

c1_count = len(clust_prod[clust_prod['cluster']==0])

c0 = clust_prod[clust_prod['cluster']==0].drop('cluster',axis=1).mean()
arr[0,0].bar(range(len(clust_prod.drop('cluster',axis=1).columns)),c0)
c1 = clust_prod[clust_prod['cluster']==1].drop('cluster',axis=1).mean()
arr[0,1].bar(range(len(clust_prod.drop('cluster',axis=1).columns)),c1)
c2 = clust_prod[clust_prod['cluster']==2].drop('cluster',axis=1).mean()
arr[1,0].bar(range(len(clust_prod.drop('cluster',axis=1).columns)),c2)
c3 = clust_prod[clust_prod['cluster']==3].drop('cluster',axis=1).mean()
arr[1,1].bar(range(len(clust_prod.drop('cluster',axis=1).columns)),c3)
c4 = clust_prod[clust_prod['cluster']==4].drop('cluster',axis=1).mean()
arr[2,0].bar(range(len(clust_prod.drop('cluster',axis=1).columns)),c4)
c5 = clust_prod[clust_prod['cluster']==5].drop('cluster',axis=1).mean()
arr[2,1].bar(range(len(clust_prod.drop('cluster',axis=1).columns)),c5)

plt.show()


#c0.sort_values(ascending=False)[0:10]

#c1.sort_values(ascending=False)[0:10]
