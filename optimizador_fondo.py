import pandas as pd
import numpy as np
from pulp import *
import os
import logging
import yaml
from datetime import datetime
import time

def cargar_configuracion(config_path='config.yaml'):
    """Carga la configuración desde el archivo YAML"""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def setup_logger():
    """Configura el logger para el proceso de optimización"""
    # Crear directorio para logs
    os.makedirs('datos_optimizacion', exist_ok=True)
    
    # Crear nombre de archivo de log con timestamp
    log_filename = f'datos_optimizacion/optimizacion_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    # Configurar el logger para escribir tanto en archivo como en consola
    logging.basicConfig(
        level=logging.DEBUG,  # Cambiado a DEBUG para ver todos los detalles
        format='%(message)s',  # Simplificado para mejor lectura de ecuaciones
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def optimizar_composicion(M, P, 
                         rendimientos_cum_path='data_sintetica/activos_rendimientos_cum.csv',
                         performance_path='data_sintetica/performance_peso.csv'):
    """
    Optimiza la composición del fondo usando programación lineal.
    """
    logger = setup_logger()
    tiempo_inicio = time.time()
    
    logger.info(f"Iniciando optimización con M={M} y P={P}")
    
    # Cargar datos
    logger.info("Cargando datos de entrada...")
    df_rendimientos_cum = pd.read_csv(rendimientos_cum_path, index_col=0)
    df_performance = pd.read_csv(performance_path, index_col=0)
    
    # Obtener valor objetivo del fondo en cada período
    valores_objetivo = df_performance.loc['F'].iloc[1:]  # Excluimos t_0
    
    logger.info("\nValores objetivo del fondo por período:")
    for t, valor in valores_objetivo.items():
        logger.info(f"{t}: {valor:.6f}")
    
    # Crear el problema de optimización
    prob = LpProblem("Optimizacion_Fondo", LpMinimize)
    
    # Variables de decisión
    activos = df_rendimientos_cum.index
    x = LpVariable.dicts("peso", activos, 0, P)  # Pesos de los activos
    y = LpVariable.dicts("seleccion", activos, 0, 1, LpBinary)  # Variables binarias para selección
    error = LpVariable.dicts("error", df_rendimientos_cum.columns, 0)  # Error por período
    
    logger.info(f"\nNúmero de activos disponibles: {len(activos)}")
    logger.info(f"Restricciones principales:")
    logger.info(f"- Máximo {M} activos")
    logger.info(f"- Peso máximo por activo: {P}")
    logger.info(f"- Suma de pesos debe ser 1")
    
    # Función objetivo: minimizar la suma de errores
    prob += lpSum(error[t] for t in df_rendimientos_cum.columns)
    
    # Restricciones
    logger.info("\nGenerando restricciones...")
    
    # 1. La suma de los pesos debe ser 1
    prob += lpSum(x[i] for i in activos) == 1
    logger.info("1. Suma de pesos = 1:")
    logger.info(f"   Σ w_i = 1")
    
    # 2. Número máximo de activos
    prob += lpSum(y[i] for i in activos) <= M
    logger.info(f"\n2. Máximo {M} activos:")
    logger.info(f"   Σ y_i ≤ {M}")
    
    # 3. Vincular variables binarias con pesos
    logger.info(f"\n3. Vinculación de pesos con variables binarias:")
    for i in activos:
        prob += x[i] <= y[i] * P
        logger.info(f"   {i}: w_{i} ≤ {P} * y_{i}")
    
    # 4. Para cada período, el valor del fondo debe coincidir con la suma ponderada
    logger.info("\n4. Restricciones de valor del fondo por período:")
    for t in df_rendimientos_cum.columns:
        valor_objetivo = valores_objetivo[t]
        logger.info(f"\nPeríodo {t}:")
        logger.info(f"Valor objetivo: {valor_objetivo:.6f}")
        
        # Construir y mostrar la restricción completa
        terminos = []
        for i in activos:
            coef = 1 + df_rendimientos_cum.loc[i, t]
            terminos.append(f"({coef:.6f} * w_{i})")
        
        ecuacion = " + ".join(terminos)
        logger.info(f"Restricción completa:")
        logger.info(f"{ecuacion} - {valor_objetivo:.6f} ≤ error_{t}")
        logger.info(f"{valor_objetivo:.6f} - ({ecuacion}) ≤ error_{t}")
        
        # Añadir las restricciones al modelo
        suma_ponderada = lpSum(x[i] * (1 + df_rendimientos_cum.loc[i, t]) for i in activos)
        prob += suma_ponderada - valor_objetivo <= error[t]
        prob += valor_objetivo - suma_ponderada <= error[t]
    
    # Resolver el problema
    logger.info("\nResolviendo el problema de optimización...")
    status = prob.solve()
    
    if status != 1:
        logger.error("No se encontró una solución óptima")
        raise ValueError("No se encontró una solución óptima")
    
    # Extraer resultados
    logger.info("Procesando resultados...")
    pesos_optimos = {i: value(x[i]) for i in activos if value(x[i]) > 1e-6}
    
    # Crear DataFrame con la solución
    df_solucion = pd.DataFrame(pesos_optimos.items(), columns=['Activo', 'A_0'])
    df_solucion.set_index('Activo', inplace=True)
    
    # Calcular error final
    error_total = sum(value(error[t]) for t in df_rendimientos_cum.columns)
    rmse = np.sqrt(error_total / len(df_rendimientos_cum.columns))
    
    # Guardar resultados
    logger.info("Guardando resultados...")
    df_solucion.to_csv('datos_optimizacion/fondo_optimizado.csv')
    
    tiempo_total = time.time() - tiempo_inicio
    logger.info(f"Optimización completada en {tiempo_total:.2f} segundos")
    logger.info(f"Error cuadrático medio final: {rmse:.6f}")
    
    return df_solucion, rmse

if __name__ == "__main__":
    try:
        # Cargar configuración
        config = cargar_configuracion()
        
        # Obtener parámetros de la configuración
        M = config['optimizador']['M']
        P = config['optimizador']['P']
        
        # Ejecutar optimización
        solucion, error = optimizar_composicion(M, P)
        
        print("\nComposición óptima del fondo:")
        print(solucion)
        print(f"\nError cuadrático medio: {error:.6f}")
        
        # Verificar restricciones
        suma_pesos = solucion['A_0'].sum()
        n_activos = len(solucion)
        print(f"\nSuma de pesos: {suma_pesos:.6f}")
        print(f"Número de activos: {n_activos}")
        
    except Exception as e:
        print(f"Error: {e}") 