  # Prediciendo-el-valor-de-mercado-de-jugadores-de-f-tbol-Predicting-football-players-market-value
#ADECUAR A LOS NUEVOS CAMBIOS

En este proyecto usé las bases de datos disponibles en www.kaggle.com/datasets/davidcariboo/player-scores con datos de Transfermarkt, me propongo entrenar 2 modelos,uno Random Forest y otro XGB, que predigan el valor de mercado de jugadores de futbol de las 5 mayores ligas de Europa(Premier league, Serie A, LaLiga, Bundesliga, Ligue 1) en el periodo 2018-2024 para luego comparar métricas.
  
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
      
    -Metricas aproximadas del modelo:
               |    RF       | XGB
    ------------------------------------
    MAE        | ~ 1.400.000 |
    -------------------------------------
    RMSE       | ~ 4.000.000 |
    --------------------------------------
    R2         | ~ 0.82      |
    --------------------------------------
    R2_ajustado| ~0.81       |
    --------------------------------------
    MAE/MEDIA  | ~32%        |
    --------------------------------------
  Proceso completo:
  
  Empecé por extraer toda la información necesaria de las bases de datos y unirla en un solo DataFrame.Eliminé algunos datos faltantes que podrían entorpecer el modelo.Luego de correr el modelo observe que el quitar outliers respecto al precio con rango intercuatílico empeoraba la predicción por lo que opté por dejar cualquier outlier que haya en el dataset. Una vez que todos los datasets estuvieron unidos en un solo dataframe creé algunos atributos que mejoraron marginalmente las predicciones y definí variables categoricas y numéricas que pasarán por el preproceso. Luego particioné el dataframe en train y test y definí las variables dependientes y la variable objetivo(valor real del pase). Después establecí el preproceso de la información que iría al pipeline con OneHotEncoder para las variables binarias y un Simple Imputer que llenará valores faltantes con cero.
  
  El modelo predictivo RF que uso es simple, utilizo un randomized grid search con 100 iteraciones para la búsqueda de hiperparametros buscando minimizar el error medio absoluto(MAE) porque es la métrica que uso como referencia en este caso.Uso los hiperpárametros mas comúnes con valores conservadores que buscan evitar el overfitting del modelo. Luego hago el .fit() del modelo, predigo sobre la porción de test e informo MAE,RMSE,R2,R2 ajustado y la relación MAE/MEDIA. 

  
  Las librerias que usé fueron Pandas, Numpy y scikit-learn
