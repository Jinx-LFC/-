from sklearn.datasets import load_iris                    #加载鸢尾花测试集
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split      #分割训练集和测试集的
from sklearn.preprocessing import StandardScaler          #数据标准化的
from sklearn.neighbors import KNeighborsClassifier        #KNN算法 分类对象
from sklearn.metrics import accuracy_score                #模型评估预测的准确率

#1.定义函数，加载鸢尾花数据集，并查看数据集‘
def dm01_load_iris():
    #1.加载数据集
    iris_data=load_iris()
    #2.查看数据集
    #print(f'数据集：{iris_data}')                  #字典形态
    #print(f'数据集的类型是：{type(iris_data)}')
    #3.查看数据集的键
    print(f'数据集的键：{iris_data.keys()}')  #键有：data,target,target_names,feature_names,DESCR
    #4.查看数据集的键对应的值
    #print(f'具体的数据：{iris_data.data[:5]}')              #有150条数据，每条数据有4个特征，只要前五条
    #print(f'具体的标签：{iris_data.target[:5]}')            #有150条数据，每条数据有1个标签，只要前五条

    print(f'具体的数据：{iris_data.data}')
    print(f'具体的标签：{iris_data.target}')
    print(f'标签对应的名称：{iris_data.target_names}')       #标签对应的名称
    print(f'特征对应的名称：{iris_data.feature_names}')      #特征对应的名称
    #print(f'数据集的描述：{iris_data.DESCR}')
    #print(f'数据集的框架：{iris_data.frame}')
    #print(f'数据集的文件名：{iris_data.filename}')
    #print(f'数据集的模型（在哪个包下）：{iris_data.data_module}')

#2.定义函数，绘制数据集的散点图
def dm02_show_iris():
    #1.加载数据集
    iris_data=load_iris()
    #2.把鸢尾花数据集封装成DataFrame对象：
    iris_df=pd.DataFrame(data=iris_data.data,columns=iris_data.feature_names)
    #3.给df对象新增标签列
    iris_df['label']=iris_data.target
    print(iris_df)
    #4.绘制散点图
    sns.lmplot(data=iris_df,x='sepal length (cm)',y='sepal width (cm)',hue='label',fit_reg=True)
    #5.设置标题，显式
    plt.title('iris data')
    plt.tight_layout()
    plt.show()

#3.定义函数，切分训练集和测试集
def dm03_split_train_test():
    #1.加载数据集
    iris_data=load_iris()
    #2.数据的预处理：从250个特征和标签中，按照8：2的比例，切分训练集和测试集
    x_train,x_test,y_train,y_test=train_test_split(iris_data.data,iris_data.target,test_size=0.2,random_state=22) #random_state随机种子，种子一致，每次生成的随机数也是固定的
    #3.打印切割后的结果
    print(f'训练集的特征：{x_train}，个数：{len(x_train)}')   #120条，每条4列（特征）
    print(f'训练集的标签：{y_train}，个数：{len(y_train)}')   #120条，每条1列（标签）
    print(f'测试集的特征：{x_test}，个数：{len(x_test)}')     #30条，每条4列（特征）
    print(f'测试集的标签：{y_test}，个数：{len(y_test)}')     #30条，每条1列（标签）

#4.定义函数，实现完整案例——》加载数据，数据预处理，特征工程，模型训练，模型评估，模型预测
def dm04_iris_evaluate_test():
    #1.加载数据集
    iris_data=load_iris()

    #2.数据的预处理，训练集和测试集的切分
    x_train,x_test,y_train,y_test=train_test_split(iris_data.data,iris_data.target,test_size=0.2,random_state=22)

    #3.特征工程（提取，预处理...）
    #3.1创建标准化对象
    transfer=StandardScaler()
    #3.2对特征列进行标准化
    x_train=transfer.fit_transform(x_train)
    x_test=transfer.transform(x_test)

    #4.模型训练
    estimator=KNeighborsClassifier(n_neighbors=3)
    estimator.fit(x_train,y_train)

    #5.模型预测
    #5.1对切分的数据集进行测试
    y_pre=estimator.predict(x_test)
    print(f'预测结果：{y_pre}')
    #5.2对新的数据进行预测
    my_data=[[6.5,2.5,6.2,1.3]]
    my_data=transfer.transform(my_data)    #测试集数据进行标准化
    y_pre_new=estimator.predict(my_data)
    print(f'预测结果：{y_pre_new}')
    #5.3查看上述数据集，每种分类的预测概率
    y_pre_proba=estimator.predict_proba(my_data)
    print(f'预测概率：{y_pre_proba}')

    #6.模型评估
    #直接评分，基于训练集的特征和训练集的标签
    print(f'准确率：{estimator.score(x_train,y_train)}')
    #基于测试集的标签和预测结果评分
    print(f'准确率：{accuracy_score(y_test,y_pre)}')

if __name__=='__main__':
    #dm01_load_iris()
    #dm02_show_iris()
    #dm03_split_train_test()
    dm04_iris_evaluate_test()
