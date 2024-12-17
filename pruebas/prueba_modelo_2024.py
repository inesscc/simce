import pandas as pd

preds = pd.read_excel('data/output/predicciones/predicciones_modelo_final.xlsx')
preds.ruta_imagen_output.iloc[-2]


probs = pd.read_excel('problemas_imagenes.xlsx')
probs.shape
probs.Error.iloc[0]
probs.Pregunta.iloc[1]

tabla99 = pd.read_csv('data/input_procesado/output_tabla_99/casos_99_compilados_6b_estudiantes.csv')


tabla99.shape

probs.Pregunta.str.contains('CE').sum()




