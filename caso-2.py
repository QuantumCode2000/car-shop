import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# DATOS EN EXCEL
df = pd.read_csv('trabajo.csv')
print(df.head(2))


# VARIABLES DE ENTRADA
entrada = ['EDAD', 'NSE', 'SEXO', 'AUTOMOVI',
           'INSTRUCC']
# VARIABLES DE SALIDA
salida = ['INTCOM']
# DATOS DE ENTRADA
x = df[entrada]
print(x.head(2))
# DATOS DE SALIDA
y = df[salida]
print(y.head(2))

if len(salida) == 1:
    y = np.ravel(y)

# SELECCIONAR DATOS DE train y test PARA ENTRENAMIENTO Y PRUEBA
#from sklearn.model_selection import train_test_split
#xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=0.2)
xtrain = x
ytrain = y
# NORMALIZAR LOS DATOS
#from sklearn.preprocessing import StandardScaler
#escalar = StandardScaler()
#xtrain = escalar.fit_transform(xtrain)
#xtest = escalar.transform(xtest)
# TRAIN
modelo = MLPClassifier(hidden_layer_sizes=(20, 20),
                       activation="logistic", random_state=1, max_iter=10000,
                       solver='lbfgs')
modelo.fit(xtrain, ytrain)
print("precisión de train: ", modelo.score(xtrain, ytrain))
# # TEST
dftest = pd.read_csv('test.csv')
xtest = dftest[entrada]
ytest = dftest[salida]
ypred = modelo.predict(xtest)
print("salida de predicción: ", ypred)
# DATOS DE CONSULTA (query)
dfquery = pd.read_csv('query.csv')
xq = dfquery[entrada]
print("datos de consulta: ")
print(xq)
# xq = escalar.transform(xq)
# NORMALIZAR LOS DATOS DE CONSULTA
# xq = escalar.transform(xq)
# RESULTADO DE LA CONSULTA
print('predicciones:', modelo.predict(xq))
