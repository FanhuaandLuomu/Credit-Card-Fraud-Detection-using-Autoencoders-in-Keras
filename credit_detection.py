#coding:utf-8
# 信用卡数据异常检测
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

from sklearn.model_selection import train_test_split

LABELS=['Normal','Fraud']

# 加载数据
df=pd.read_csv('data/creditcard.csv')
print(df.shape)

print(df.isnull().values.any())  # False 没有缺失值

# 欺诈数据与正常数据的比例
count_classes=pd.value_counts(df['Class'],sort=True)

count_classes.plot(kind='bar',rot=0)
plt.title('Transaction class distribution')
plt.xticks(range(2),LABELS)
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()

frauds=df[df.Class==1]
normal=df[df.Class==0]

# (492,31) (284315,31)
print(frauds.shape,normal.shape)

# 不同交易中使用的货币数量
print(frauds.Amount.describe())
print(normal.Amount.describe())

# 两行一列 2子图  共享x轴
f,(ax1,ax2)=plt.subplots(2,1,sharex=True)
f.suptitle('Amount per transaction by class')
bins=50

# 柱形图
ax1.hist(frauds.Amount,bins=bins)
ax1.set_title('Fraud')

ax2.hist(normal.Amount,bins=bins)
ax2.set_title('Normal')

plt.xlabel('Amount($)')
plt.ylabel('Number of Transactions')
plt.xlim(0,20000)
# log(y)
plt.yscale('log')
plt.show()

# 不同时间中使用的数量 // 时间特征并不重要
f,(ax1,ax2)=plt.subplots(2,1,sharex=True)
f.suptitle('Time of transaction vs Amount by class')

ax1.scatter(frauds.Time,frauds.Amount)
ax1.set_title('Fraud')

ax2.scatter(normal.Time,normal.Amount)
ax2.set_title('Normal')

plt.xlabel('Time (s)')
plt.ylabel('Amount')

plt.show()

# 建立自编码器
from sklearn.preprocessing import StandardScaler
data=df.drop(['Time'],axis=1)

print(data['Amount'].values.shape)

# (x-mean)/std
data['Amount']=StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))

print(data['Amount'].shape)
# print(data['Amount'])

# 划分0.2为测试集
X_train,X_test=train_test_split(data,test_size=0.2,random_state=42)
X_train=X_train[X_train.Class==0]
X_train=X_train.drop(['Class'],axis=1)

y_test=X_test['Class']
X_test=X_test.drop(['Class'],axis=1)
X_train=X_train.values
X_test=X_test.values
print(X_train.shape,X_test.shape,y_test.shape)

from keras.models import Model,load_model
from keras.layers import Input,Dense
from keras.callbacks import ModelCheckpoint,TensorBoard
from keras import regularizers

# create model
input_dim=X_train.shape[1]
encoding_dim=14

input_layer=Input(shape=(input_dim,))
encoder=Dense(encoding_dim,activation='tanh',\
		activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoder=Dense(int(encoding_dim/2),activation='relu')(encoder)
decoder=Dense(encoding_dim,activation='tanh')(encoder)
decoder=Dense(input_dim,activation='linear')(decoder)

auto_encoder=Model(inputs=input_layer,outputs=decoder)

print(auto_encoder.summary())

nb_epoch=150
batch_size=256

auto_encoder.compile(optimizer='adam',loss='mse',metrics=['accuracy'])

checkpointer=ModelCheckpoint(filepath='model_me.h5',verbose=1,save_best_only=True)

tensorboard=TensorBoard(log_dir='./logs',histogram_freq=0,\
		write_graph=True,write_images=True)

'''
# 训练
hist=auto_encoder.fit(X_train,X_train,epochs=nb_epoch,\
	batch_size=batch_size,shuffle=True,validation_data=(X_test,X_test),\
	verbose=1,callbacks=[checkpointer,tensorboard]).history

plt.plot(hist['loss'])
plt.plot(hist['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper right')
plt.show()
'''

# 导入model
auto_encoder=load_model('model_me.h5')

predictions=auto_encoder.predict(X_test)
mse=np.mean(np.power(X_test-predictions,2),axis=1)

error_df=pd.DataFrame({'reconstruction_error':mse,'true_class':y_test})
print(error_df.describe())

# 无欺诈数据的重构误差
fig=plt.figure()
ax=fig.add_subplot(111)
normal_error_df=error_df[(error_df['true_class']==0) & (error_df['reconstruction_error']<10)]
_=ax.hist(normal_error_df.reconstruction_error.values,bins=10)
plt.show()

# 有欺诈数据的重构误差
fig=plt.figure()
ax=fig.add_subplot(111)
fraud_error_df=error_df[error_df['true_class']==1]
_=ax.hist(fraud_error_df.reconstruction_error.values,bins=10)
plt.show()

from sklearn.metrics import (confusion_matrix,precision_recall_curve,auc,\
	roc_curve,recall_score,classification_report,f1_score,precision_recall_fscore_support)

# ROC曲线
fpr,tpr,thresholds=roc_curve(error_df.true_class,error_df.reconstruction_error)
print(fpr.shape,tpr.shape)
# print(thresholds)

roc_auc=auc(fpr,tpr)  # 面积
print('roc_auc:',roc_auc)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr,tpr,label='AUC=%.4F' %(roc_auc))
plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])

plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# precision VS recall
precision,recall,th=precision_recall_curve(error_df.true_class,error_df.reconstruction_error)
plt.plot(recall,precision,'b',label='Precision-Recall curve')
plt.title('Recall VS Precision')
plt.xlabel('Recall')
plt.ylabel('Precision')

plt.show()

print(precision.shape,recall.shape,th.shape)

# precision by threshold
plt.plot(th,precision[1:],'b',label='Threshold-Precision curve')
plt.title('Precision for different threshold values')
plt.xlabel('threshold')
plt.ylabel('Precision')
plt.show()

# recall by threshold
plt.plot(th,recall[1:],'r',label='Threshold-Recall curve')
plt.title('Recall for different threshold values')
plt.xlabel('threshold')
plt.ylabel('Recall')
plt.show()

'''
我们需要从交易数据本身计算重构误差。
如果误差大于预先定义的阈值，我们就把它标记为欺诈
（因为我们的模型在正常交易上应该有一个低的重构误差）
'''
# TODO
threshold=2.5

groups=error_df.groupby('true_class')
fig,ax=plt.subplots()

for name,group in groups:
	ax.plot(group.index,group.reconstruction_error,marker='o',\
		ms=3.5,linestyle='',label='Fraud' if name==1 else 'Normal')

ax.hlines(threshold,ax.get_xlim()[0],ax.get_xlim()[1],color='r',\
		zorder=100,label='Threshold')
ax.legend()

plt.title('Reconstruction error for different classes')
plt.ylabel('Reconstruction error')
plt.xlabel('Data point index')

plt.show()

# 预测
y_pred=[1 if e>threshold else 0 for e in error_df.reconstruction_error.values]
conf_matrix=confusion_matrix(error_df.true_class,y_pred)

print(conf_matrix)
p_normal=conf_matrix[0][0]*1.0/(conf_matrix[0][0]+conf_matrix[1][0])
r_normal=conf_matrix[0][0]*1.0/(conf_matrix[0][0]+conf_matrix[0][1])

p_fraud=conf_matrix[1][1]*1.0/(conf_matrix[0][1]+conf_matrix[1][1])
r_fraud=conf_matrix[1][1]*1.0/(conf_matrix[1][0]+conf_matrix[1][1])

print('p_normal:%f  r_normal:%f' %(p_normal,r_normal))
print('p_fraud:%f  r_fraud:%f' %(p_fraud,r_fraud))


plt.figure(figsize=(12,12))
sns.heatmap(conf_matrix,xticklabels=LABELS,yticklabels=LABELS,\
							annot=True,fmt='d')
plt.title('Confusion matrix')
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()




