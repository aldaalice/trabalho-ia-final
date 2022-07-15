import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import matplotlib.pyplot as plt
import warnings

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import metrics
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")

df = pd.read_csv('Iris.csv')

print("dados zerados: ")
print(df.isnull().any())
print("")
print("")
print("tipos de dados: ")
print(df.dtypes)
print("")
print("")

df['PetalWidthCm'].plot.hist()

entradas = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
classes = df['Species'].values

treinar_entradas, test_inputs, treinar_classes, test_classes = train_test_split(entradas, classes, train_size=0.7, random_state=1)

correlacao = DecisionTreeClassifier()
correlacao.fit(treinar_entradas, treinar_classes)
print("acurácia da arvore de decisão: ",correlacao.score(test_inputs, test_classes))
print("")
print("")
sns.pairplot(df, hue= 'Species')

tree.plot_tree(correlacao)

representacao_texto = tree.export_text(correlacao)
print("Arvore de decisão em texto: ")
print(representacao_texto)
print("")
print("")

X =df.drop("Species",axis=1)
Y = df['Species']

treino_x, teste_x,treino_y, teste_y = train_test_split(X,Y,test_size=0.3,random_state=1)

model_knn = KNeighborsClassifier()
model_knn.fit(treino_x,treino_y)
y_pred_model_knn = model_knn.predict(teste_x)
print('A acuracia do modelo KNN eh de ',metrics.accuracy_score(teste_y,y_pred_model_knn))
print("")
print("")

plt.figure(figsize=(7,4))
sns.countplot("Species",data=df)
plt.show()

iris = datasets.load_iris()

X = iris.data[:, [2, 3]]
y = iris.target
iris_df = pd.DataFrame(iris.data[:, [2, 3]], columns=iris.feature_names[2:])

treino_x, teste_x, treino_y, teste_y = train_test_split(X, y, test_size=.3, random_state=0)

sc = StandardScaler()
sc.fit(treino_x)
treino_x_std = sc.transform(treino_x)
teste_x_std = sc.transform(teste_x)

modelo_svm = SVC(kernel='rbf', random_state=0, gamma=.10, C=1.0)
modelo_svm.fit(treino_x_std, treino_y)

# print(' A acuracia do teste do medelo SVM eh de {:.2f} de  1'.format(modelo_svm.score(teste_x_std, teste_y)))
# print("")
# print("")

print(' A acuracia do treino do modelo SVM eh de  {:.2f} de  1'.format(modelo_svm.score(treino_x_std, treino_y)))
print("")
print("")

markers = ('s', 'x', 'o')
colors = ('red', 'blue', 'lightgreen')
cmap = ListedColormap(colors[:len(np.unique(teste_y))])

for idx, cl in enumerate(np.unique(y)):
    plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
               c=cmap(idx), marker=markers[idx], label=cl)
