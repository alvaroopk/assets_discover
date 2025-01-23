import pandas as pd
import numpy as np
from scipy.optimize import minimize
import os
import logging
from datetime import datetime
import time

# Configurar logging
def setup_logger():
    """Configura el logger para el proceso de optimización"""
    # Crear directorio para logs si no existe
    os.makedirs('datos_optimizacion/logs', exist_ok=True)
    
    # Configurar el logger
    log_filename = f'datos_optimizacion/logs/optimizacion_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def calcular_evolucion_pesos(pesos_iniciales, df_rendimientos):
    """
    Calcula la evolución de los pesos y rendimientos del fondo.
    """
    n_periodos = len(df_rendimientos.columns)
    pesos = pd.DataFrame(index=df_rendimientos.index, columns=df_rendimientos.columns)
    rendimientos_fondo = pd.Series(index=df_rendimientos.columns)
    
    # Inicializar primer período
    pesos['t_0'] = pesos_iniciales
    rendimientos_fondo['t_0'] = (pesos_iniciales * df_rendimientos['t_0']).sum()
    
    # Calcular pesos y rendimientos para períodos siguientes
    for t in range(1, n_periodos):
        col_actual = f't_{t}'
        col_anterior = f't_{t-1}'
        
        # Actualizar valores según rendimientos
        valores_actualizados = pesos[col_anterior] * (1 + df_rendimientos[col_anterior])
        # Normalizar para que sumen 1
        pesos[col_actual] = valores_actualizados / valores_actualizados.sum()
        # Calcular rendimiento del período
        rendimientos_fondo[col_actual] = (pesos[col_actual] * df_rendimientos[col_actual]).sum()
    
    return pesos, rendimientos_fondo

def funcion_objetivo(pesos_iniciales, df_rendimientos, df_fondo):
    """
    Calcula el error cuadrático medio entre los rendimientos del fondo y el objetivo.
    """
    # Convertir array a Series con índices correctos
    pesos_series = pd.Series(pesos_iniciales, index=df_rendimientos.index)
    
    # Calcular evolución de pesos y rendimientos
    _, rendimientos_calculados = calcular_evolucion_pesos(pesos_series, df_rendimientos)
    
    # Calcular error cuadrático medio
    rendimientos_objetivo = df_fondo.iloc[0]
    error = ((rendimientos_calculados - rendimientos_objetivo)**2).mean()
    
    return error

def restriccion_suma_uno(pesos):
    """La suma de los pesos debe ser 1"""
    return np.sum(pesos) - 1

def restriccion_num_activos(pesos, M):
    """No más de M activos con peso significativo"""
    return M - np.sum(pesos > 1e-4)

def optimizar_composicion(M, P, 
                         rendimientos_path='data_sintetica/activos_rendimientos.csv',
                         fondo_path='data_sintetica/fondo_evolucion.csv'):
    """
    Optimiza la composición inicial del fondo usando optimización no lineal.
    """
    logger = setup_logger()
    tiempo_inicio = time.time()
    
    logger.info(f"Iniciando optimización con M={M} y P={P}")
    
    # Crear directorio para resultados
    os.makedirs('datos_optimizacion', exist_ok=True)
    
    # Cargar datos
    logger.info("Cargando datos de entrada...")
    df_rendimientos = pd.read_csv(rendimientos_path, index_col=0)
    df_fondo = pd.read_csv(fondo_path, index_col=0)
    
    n_activos = len(df_rendimientos)
    logger.info(f"Datos cargados: {n_activos} activos y {len(df_rendimientos.columns)} períodos")
    
    mejor_resultado = None
    mejor_error = float('inf')
    
    # Realizar múltiples intentos con diferentes puntos iniciales
    for intento in range(10):
        logger.info(f"\nIniciando intento {intento + 1}/10")
        
        # Generar pesos iniciales aleatorios que sumen 1
        x0 = np.random.random(n_activos)
        x0 = x0 / x0.sum()
        
        # Definir límites y restricciones
        bounds = [(0, P) for _ in range(n_activos)]
        restricciones = [
            {'type': 'eq', 'fun': restriccion_suma_uno},
            {'type': 'ineq', 'fun': restriccion_num_activos, 'args': (M,)}
        ]
        
        # Optimizar
        resultado = minimize(
            funcion_objetivo,
            x0,
            args=(df_rendimientos, df_fondo),
            method='SLSQP',
            bounds=bounds,
            constraints=restricciones,
            options={'maxiter': 1000}
        )
        
        if resultado.success:
            logger.info(f"Intento {intento + 1} exitoso. Error: {resultado.fun:.6f}")
            if resultado.fun < mejor_error:
                mejor_error = resultado.fun
                mejor_resultado = resultado
                logger.info("¡Nuevo mejor resultado encontrado!")
        else:
            logger.warning(f"Intento {intento + 1} falló: {resultado.message}")
    
    if mejor_resultado is None:
        logger.error("No se encontró una solución factible en ningún intento")
        raise ValueError("No se encontró una solución factible en ningún intento")
    
    # Obtener y procesar pesos optimizados
    logger.info("\nProcesando mejor solución encontrada...")
    pesos_optimos = pd.Series(mejor_resultado.x, index=df_rendimientos.index)
    
    # Filtrar pesos muy pequeños y renormalizar
    pesos_optimos[pesos_optimos < 1e-4] = 0
    pesos_optimos = pesos_optimos / pesos_optimos.sum()
    
    # Verificar restricciones
    n_activos_seleccionados = (pesos_optimos > 1e-4).sum()
    logger.info(f"Número de activos seleccionados: {n_activos_seleccionados}")
    
    if n_activos_seleccionados > M:
        logger.error(f"La solución excede el número máximo de activos: {n_activos_seleccionados} > {M}")
        raise ValueError(f"La solución tiene {n_activos_seleccionados} activos, más que el máximo permitido {M}")
    
    if any(peso > P for peso in pesos_optimos):
        logger.error(f"Algunos pesos exceden el máximo permitido {P}")
        raise ValueError(f"Algunos pesos exceden el máximo permitido {P}")
    
    # Calcular evolución de pesos y rendimientos
    pesos_evolucion, rendimientos_calculados = calcular_evolucion_pesos(
        pesos_optimos, df_rendimientos
    )
    
    # Calcular error final
    rendimientos_objetivo = df_fondo.iloc[0]
    rmse = np.sqrt(((rendimientos_calculados - rendimientos_objetivo)**2).mean())
    
    # Crear DataFrame con pesos iniciales
    df_solucion = pd.DataFrame(pesos_optimos[pesos_optimos > 0], columns=['A'])
    
    # Guardar resultados
    logger.info("\nGuardando resultados...")
    df_solucion.to_csv('datos_optimizacion/fondo_optimizado.csv')
    pesos_evolucion.to_csv('datos_optimizacion/pesos_optimizados_evolucion.csv')
    pd.DataFrame(rendimientos_calculados).T.to_csv('datos_optimizacion/rendimientos_calculados.csv')
    
    tiempo_total = time.time() - tiempo_inicio
    logger.info(f"\nOptimización completada en {tiempo_total:.2f} segundos")
    logger.info(f"Error cuadrático medio final: {rmse:.6f}")
    
    return df_solucion, rmse

if __name__ == "__main__":
    # Ejemplo de uso
    M = 4  # máximo 4 activos
    P = 0.30  # peso máximo 30%
    
    try:
        solucion, error = optimizar_composicion(M, P)
        print("\nComposición óptima del fondo:")
        print(solucion)
        print(f"\nError cuadrático medio: {error:.6f}")
        
        # Verificar que la suma de pesos es 1
        suma_pesos = solucion['A'].sum()
        print(f"\nSuma de pesos: {suma_pesos:.6f}")
        
        # Verificar número de activos
        n_activos = len(solucion)
        print(f"\nNúmero de activos: {n_activos}")
        
    except Exception as e:
        print(f"Error: {e}") 