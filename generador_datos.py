import pandas as pd
import numpy as np
import os
import yaml

def cargar_configuracion(config_path='config.yaml'):
    """Carga la configuración desde el archivo YAML"""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def generar_activos(n_activos, t_periodos, crecimiento):
    """
    Genera un DataFrame con rendimientos sintéticos de activos y sus rendimientos acumulados.
    
    Parámetros:
    -----------
    n_activos : int
        Número de activos a generar
    t_periodos : int
        Número de períodos temporales
    crecimiento : float
        Tendencia de crecimiento medio
    
    Retorna:
    --------
    pandas.DataFrame
        DataFrame con los rendimientos de los activos
    """
    # Crear matriz de rendimientos aleatorios
    rendimientos = np.random.normal(crecimiento, 0.1, size=(n_activos, t_periodos))
    
    # Crear DataFrame
    columnas = [f't_{i+1}' for i in range(t_periodos)]  # Empezamos en t_1
    indices = [f'a_{i+1}' for i in range(n_activos)]
    df_rendimientos = pd.DataFrame(rendimientos, columns=columnas, index=indices)
    
    # Crear DataFrame de rendimientos acumulados
    df_rendimientos_cum = pd.DataFrame(index=df_rendimientos.index, columns=df_rendimientos.columns)
    
    # Calcular rendimientos acumulados
    for t in range(t_periodos):
        col_actual = f't_{t+1}'
        if t == 0:
            # Para el primer período, es simplemente (1 + r1)
            df_rendimientos_cum[col_actual] = 1 + df_rendimientos[col_actual]
        else:
            # Para los siguientes períodos, multiplicamos por (1 + rt)
            col_anterior = f't_{t}'
            df_rendimientos_cum[col_actual] = df_rendimientos_cum[col_anterior] * (1 + df_rendimientos[col_actual])
    
    # Restar 1 para obtener el rendimiento acumulado como porcentaje
    df_rendimientos_cum = df_rendimientos_cum - 1
    
    # Guardar en CSV
    os.makedirs('data_sintetica', exist_ok=True)
    df_rendimientos.to_csv('data_sintetica/activos_rendimientos.csv')
    df_rendimientos_cum.to_csv('data_sintetica/activos_rendimientos_cum.csv')
    
    return df_rendimientos

def generar_fondo(m, peso_maximo):
    """
    Genera un fondo aleatorio con m activos y calcula su evolución.
    
    Parámetros:
    -----------
    m : int
        Número de activos en el fondo
    peso_maximo : float
        Peso máximo permitido para cada activo
    
    Retorna:
    --------
    tuple (pandas.DataFrame, pandas.DataFrame)
        (DataFrame con composición inicial del fondo, 
         DataFrame con la evolución del fondo y los activos)
    """
    # Cargar rendimientos de activos
    df_rendimientos = pd.read_csv('data_sintetica/activos_rendimientos.csv', index_col=0)
    n_periodos = len(df_rendimientos.columns)
    
    # Seleccionar m activos aleatorios
    activos_seleccionados = np.random.choice(df_rendimientos.index, size=m, replace=False)
    
    # Generar pesos aleatorios que sumen 1 y no excedan peso_maximo
    pesos = np.random.uniform(0, peso_maximo, size=m)
    pesos = pesos / pesos.sum()  # Normalizar para que sumen 1
    
    # Crear DataFrame con la composición del fondo (pesos iniciales)
    df_fondo = pd.DataFrame(index=activos_seleccionados, columns=['A_0'])
    df_fondo['A_0'] = pesos
    
    # Crear DataFrame para la evolución de performance y pesos
    columnas_evolucion = ['t_0'] + [f't_{i+1}' for i in range(n_periodos)]
    df_performance = pd.DataFrame(index=list(activos_seleccionados) + ['F'], columns=columnas_evolucion)
    
    # Inicializar t_0
    df_performance.loc[activos_seleccionados, 't_0'] = pesos
    df_performance.loc['F', 't_0'] = 1.0  # El fondo empieza en 1
    # Calcular evolución
    valores_activos = pd.Series(pesos, index=activos_seleccionados)
    
    for t in range(n_periodos):
        col_actual = f't_{t+1}'
        col_anterior = f't_{t}'
        
        # Actualizar valores de los activos
        rendimientos_periodo = df_rendimientos.loc[activos_seleccionados, col_actual]
        valores_activos = valores_activos * (1 + rendimientos_periodo)
        
        # Guardar valores normalizados
        df_performance.loc[activos_seleccionados, col_actual] = valores_activos
        
        # Calcular y guardar valor del fondo como suma de los activos
        df_performance.loc['F', col_actual] = df_performance.loc[activos_seleccionados, col_actual].sum()
    
    # Guardar resultados
    df_fondo.to_csv('data_sintetica/activos_fondo.csv')
    df_performance.to_csv('data_sintetica/performance_peso.csv')
    
    return df_fondo, df_performance

if __name__ == "__main__":
    try:
        # Cargar configuración
        config = cargar_configuracion()
        
        # Generar activos
        print("\nGenerando activos...")
        df_rendimientos = generar_activos(
            n_activos=config['generador']['n_activos'],
            t_periodos=config['generador']['t_periodos'],
            crecimiento=config['generador']['crecimiento']
        )
        print("\nActivos generados y guardados en data_sintetica/activos_rendimientos.csv")
        
        # Generar fondo
        print("\nGenerando fondo...")
        fondo, performance = generar_fondo(
            m=config['generador']['m_fondo'],
            peso_maximo=config['generador']['peso_maximo_fondo']
        )
        
        print("\nComposición inicial del fondo:")
        print(fondo)
        print("\nEvolución del fondo y activos guardada en data_sintetica/performance_peso.csv")
        print(performance)
        
    except Exception as e:
        print(f"Error: {e}") 