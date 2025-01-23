import pandas as pd
import numpy as np
import os

def generar_activos(n_activos, t_periodos, crecimiento):
    """
    Genera un DataFrame con datos sintéticos de activos.
    
    Parámetros:
    -----------
    n_activos : int
        Número de activos (filas)
    t_periodos : int
        Número de períodos de tiempo (columnas)
    crecimiento : float
        Media del crecimiento para la distribución normal
        
    Retorna:
    --------
    pandas.DataFrame
        DataFrame con los datos sintéticos generados
    """
    
    # Crear nombres de filas y columnas
    index = [f'a_{i}' for i in range(1, n_activos + 1)]
    columns = [f't_{i}' for i in range(t_periodos)]
    
    # Generar datos aleatorios
    data = np.random.normal(
        loc=crecimiento,
        scale=3*crecimiento,
        size=(n_activos, t_periodos)
    )
    
    # Crear DataFrame y redondear a 5 decimales
    df = pd.DataFrame(data, index=index, columns=columns).round(5)
    
    # Crear directorio si no existe
    os.makedirs('data_sintetica', exist_ok=True)
    
    # Guardar DataFrame
    df.to_csv('data_sintetica/activos_rendimientos.csv')
    
    return df

def generar_fondo(m, peso_maximo, csv_path='data_sintetica/activos_rendimientos.csv'):
    """
    Genera un fondo seleccionando m activos aleatorios y asignándoles pesos,
    calculando también su evolución temporal.
    
    Parámetros:
    -----------
    m : int
        Número de activos a seleccionar
    peso_maximo : float
        Peso máximo permitido para cada activo (entre 0 y 1)
    csv_path : str
        Ruta al archivo CSV con los rendimientos de los activos
    
    Retorna:
    --------
    tuple (pandas.DataFrame, pandas.DataFrame, pandas.DataFrame)
        (DataFrame con composición inicial del fondo, 
         DataFrame con la evolución del fondo,
         DataFrame con la evolución de los pesos)
    """
    # Leer el CSV de rendimientos
    df_rendimientos = pd.read_csv(csv_path, index_col=0)
    
    # Validar que m no sea mayor que el número de filas
    n_activos = len(df_rendimientos)
    if m > n_activos:
        raise ValueError(f"m ({m}) no puede ser mayor que el número de activos disponibles ({n_activos})")
    
    if peso_maximo <= 0 or peso_maximo > 1:
        raise ValueError("peso_maximo debe estar entre 0 y 1")
    
    # Seleccionar m filas aleatorias
    activos_seleccionados = df_rendimientos.sample(n=m)
    
    # Generar pesos aleatorios que sumen 1 y no excedan peso_maximo
    while True:
        pesos = np.random.random(m)
        pesos = pesos / pesos.sum()
        if all(peso <= peso_maximo for peso in pesos):
            break
    
    # Añadir los pesos como primera columna
    activos_seleccionados.insert(0, 'A', pesos.round(5))
    
    # Calcular la evolución del fondo
    rendimientos = activos_seleccionados.iloc[:, 1:]  # Excluir columna de pesos
    pesos_actuales = activos_seleccionados['A'].values
    
    # Crear DataFrame para almacenar la evolución de los pesos
    evolucion_pesos = pd.DataFrame(
        index=activos_seleccionados.index,
        columns=rendimientos.columns,
        data=0.0
    )
    evolucion_pesos['t_0'] = pesos_actuales
    
    # Calcular rendimiento del fondo para cada período
    rendimientos_fondo = pd.DataFrame(
        index=['fondo'],
        columns=rendimientos.columns
    )
    
    # Para el primer período
    rendimiento_periodo = np.sum(rendimientos['t_0'] * pesos_actuales)
    rendimientos_fondo['t_0'] = round(rendimiento_periodo, 5)
    
    # Para los siguientes períodos
    for t in range(1, len(rendimientos.columns)):
        col_actual = f't_{t}'
        col_anterior = f't_{t-1}'
        
        # Actualizar los pesos según los rendimientos del período anterior
        valores_actualizados = pesos_actuales * (1 + rendimientos[col_anterior])
        pesos_actuales = valores_actualizados / valores_actualizados.sum()
        evolucion_pesos[col_actual] = pesos_actuales.round(5)
        
        # Calcular el rendimiento del período actual con los pesos actualizados
        rendimiento_periodo = np.sum(rendimientos[col_actual] * pesos_actuales)
        rendimientos_fondo[col_actual] = round(rendimiento_periodo, 5)
    
    # Guardar los resultados
    activos_seleccionados.to_csv('data_sintetica/activos_fondo.csv')
    rendimientos_fondo.to_csv('data_sintetica/fondo_evolucion.csv')
    evolucion_pesos.to_csv('data_sintetica/pesos_evolucion.csv')
    
    return activos_seleccionados, rendimientos_fondo, evolucion_pesos

if __name__ == "__main__":
    # Ejemplo de uso
    df = generar_activos(n_activos=10, t_periodos=10, crecimiento=0.03)
    print("DataFrame generado y guardado en data_sintetica/activos_rendimientos.csv")
    print("\nPrimeras filas del DataFrame:")
    print(df.head())
    
    # Ejemplo de uso de generar_fondo
    try:
        fondo, evolucion, pesos = generar_fondo(m=4, peso_maximo=0.3)
        print("\nFondo generado y guardado en data_sintetica/activos_fondo.csv")
        print("\nComposición inicial del fondo:")
        print(fondo)
        print("\nEvolución del fondo guardada en data_sintetica/fondo_evolucion.csv")
        print(evolucion)
        print("\nEvolución de los pesos guardada en data_sintetica/pesos_evolucion.csv")
        print(pesos)
    except ValueError as e:
        print(f"Error: {e}") 