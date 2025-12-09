  # Prediciendo-el-valor-de-mercado-de-jugadores-de-f-tbol-Predicting-football-players-market-value

En este proyecto usé las bases de datos disponibles en www.kaggle.com/datasets/davidcariboo/player-scores con datos de Transfermarkt.Me propongo entrenar 2 modelos,uno Random Forest y otro XGB, que predigan el valor de mercado de jugadores de futbol de las 5 mayores ligas de Europa(Premier league, Serie A, LaLiga, Bundesliga, Ligue 1) en el periodo 2018-2024 para luego comparar métricas.
  
Dado que el periodo de tiempo elegido abarca varios años decidí que la variable objetivo sea el valor real del jugador durante la última temporada que jugó en su respectiva liga (los precios los actualizo a partir de los datos de inflación de la zona euro, publicados por el Banco Mundial).

Resumen:
      
    -Trabajando los datos:
        -Integré los datos de 5 database en un solo dataframe.
        -No eliminé outliers, empeoraba la performance.
        -Eliminé datos faltantes que podrian romper el modelo.
        -Creé nuevos atributos(efectividad,jugador ofensivo,arquero).
        -Definí variable objetivo(valor real) y variables independientes.
          -Definí variables categoricas y númericas a utilizar.
        -Particioné el data set en train y test.
    -Transformación de la informacino relevante
        -Binarización de variables categoricas y uso de Simple Imputer para faltantes(estrategia constante)
    -Random Forest:
        -Definición del modelo a usar(RF)
        -Búsqueda de hiperparametros con Randomized grid search(100 iteraciones)
        -Gráfico de importancia de atributos
    -XGB:
        -Definición del modelo a usar
        -Búsqueda de hiperparametros con Randomized search(100 iteraciones)
        -Gráfico de importancia de atributos

    -Metricas aproximadas de cada modelo:
               |    RF       | XGB
    -------------------------------------------
    MAE        | ~ 1.400.000 |  ~1.596.506,819|
    ------------------------------------------
    R2         |    ~ 0.82   |   ~0.8740      |
    ------------------------------------------
    R2_ajustado|    ~0.81    |   ~0.8725      |
    ------------------------------------------
    MAE/MEDIA  |     ~32%    |    0.3322      |
    ------------------------------------------

Interpretación de las métricas:





  Proceso completo:

Para empezar, el proyecto esta compuesto por 5 archivos:

    -Según Orden de ejecución:      
      -tratamiento_de_datos.ipynb (1)
      -division_y_preproceso.py (2)
      -funciones_varias.py (3)
      -Modelo_RF (4)
      -Modelo_XGB (5)

En el primer archivo extraigo toda la información necesaria de los 5 datasets que se utilizaron(competitions,clubs,players,game_events,game_lineups) y creo nuevos atributos entre los que esta la variable objetivo, el "valor_real" de mercado de cada jugador. Uso valor real para actualizar por inflación el valor de mercado de cada jugador y tener un valor comparable en cada momento(recordar que el dataset tiene datos del 2018 hasta el 2024). Una vez terminado con eso paso a una breve seccion de estadistica descriptiva del dataset que incluye los siguientes graficos:
      
-------Grafico 1,distribucion del valor real de los jugadores: En este gráfico se puede observar que la mayoria de observaciones se ubican en el lado izquierdo del grafico,marcando.....

-------Gráfico 2,Distribución de posiciones:se puede observar que la posicion con mas jugadores es la de Defensor Central, seguida por.... 

-------Gráfico 3,matriz de correlaciones: este último grafico permite ver las correlaciones entre variables numericas del dataset, se puede ver que la mayor correlacion del valor real es el maximo valor que alcanzó un jugador en su carrera.

Este archivo tambien contiene el calculo de rangos intercuartilicos para la eliminación de outliers, aunque no se lleva a cabo porque luego de varios intentos note que sacar outliers empeoraba la predicción(posiblemente por la poca cantidad de datos que tiene el dataset final, alrededor de 7000 observaciones).Luego se eliminan columnas que no se van a utilizar y se deshechan observaciones con NaN que empeoraban el modelo o imposibilitaban la predicción. Lo último que hace el archivo 1 es descargar el DataFrame final a un archivo csv para poder usarlo en el entrenamiento de los modelos.

El archivo division_y_preproceso contiene una funcion homónima que toma como argumento la dirección(path) del archivo csv descargado por el archivo 1. Esta funcion define la variable objetivo, las columnas categóricas y numericas y su tratamiento(OHE y SimpleImputer) y también particiona el dataset en train y test. La función devuelve los dataframes de X_train,X_test,y_train,y_test y el preprocesamiento de las columnas (para poner directamente en el pipeline).

El tercer archivo, funciones_varias, tiene 3 funciones. La primero funcion ,inf_acum, calcula ,como su nombre lo indica, la inflacion acumulada en un plazo de tiempo en la zona euro(la serie es la provista por el Banco Mundial), esta función se usa para el cálculo del valor real de cada jugador. La segunda función que el archivo contiene es "metricas" que es la encargada de calcular las distintas métricas utilizadas para el análisis de cada modelo, para ello necesita como argumento el modelo fiteado,X_test e y_test. Por último, la tercera función "importancia_atributos" calcula la importancia que el modelo le da a cada atributo a la hora de reducir el error,toma como argumento el modelo fiteado y la cantidad de atributos que se desea observar y devuelve un grafico de importancia de los atributos.

El cuarto y quinto archivo son muy similares por los que describo de forma conjunta. En estos se llama a la funcion división_y_proceso para obtener el output de la función, luego se define el pipeline y el modelo a entrenar. Una vez hecho eso se pasa a la definicion de los espacios de búsqueda de hiperparámetros, a mi entender los espacios de búsqueda que uso son conservadores y evitan el overfitting del modelo.Antes del fit del modelo defino un RandomizedsearchCV donde vuelco el pipeline,los espacios de búsqueda, la cantidad de folds(5) y el score a minimizar. Luego llamo a las funciones "metricas" e "importancia_atributos" y concluyo el proyecto.

  Las librerias que usé fueron Pandas,Seaborn,Matplotlib, Numpy y scikit-learn

Posibles mejoras:

Se podria usar un modelo de stacking con los dos modelos usados y un modelo LASSO como metamodelo,además, se podria usaar la libreria optuna para la búsqueda de hiperparametros óptimos.

Limitaciones: Alta correlación entre variables, el modelo enfrenta un problema que es la alta correlación entre "highest_market_value_in_eur" y "valor_real" esto se debe a que muchas observaciones tienen que su valor real es igual a su "highest_market_value_in_eur" pero me vi imposibilitado de eliminar esta última dado que el dataset de jugadores ya es demasiado pequeño y si yo optase por eliminar todas las observaciones en que coinciden (junto con todas las otras observaciones perdidas por falta de información valiosa) me quedaría demasiada poca información para poder 
