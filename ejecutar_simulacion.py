import yaml
import os
import logging
from datetime import datetime
import time
from generador_datos import generar_activos, generar_fondo
from optimizador_fondo import optimizar_composicion

def setup_logger():
    """Configura el logger para el proceso"""
    # Configurar el logger para mostrar solo en consola
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def cargar_configuracion(config_path='config.yaml'):
    """Carga la configuración desde el archivo YAML"""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def ejecutar_simulacion(config_path='config.yaml'):
    """
    Ejecuta el proceso completo de generación y optimización.
    """
    # Configurar logger
    logger = setup_logger()
    tiempo_inicio = time.time()
    
    try:
        # Cargar configuración
        logger.info("Cargando configuración...")
        config = cargar_configuracion(config_path)
        
        # Crear directorio necesario
        os.makedirs('datos_optimizacion', exist_ok=True)
        
        # Generar datos sintéticos
        logger.info("Generando datos sintéticos...")
        df_rendimientos = generar_activos(
            n_activos=config['generador']['n_activos'],
            t_periodos=config['generador']['t_periodos'],
            crecimiento=config['generador']['crecimiento']
        )
        
        # Generar fondo sintético
        logger.info("Generando fondo sintético...")
        fondo_original, evolucion, rendimientos = generar_fondo(
            m=config['generador']['m_fondo'],
            peso_maximo=config['generador']['peso_maximo_fondo']
        )
        
        # Ejecutar optimización
        logger.info("Iniciando proceso de optimización...")
        fondo_optimizado, error = optimizar_composicion(
            M=config['optimizador']['M'],
            P=config['optimizador']['P']
        )
        
        tiempo_total = time.time() - tiempo_inicio
        logger.info(f"Simulación completada en {tiempo_total:.2f} segundos")
        logger.info(f"Error cuadrático medio final: {error:.6f}")
        
        return {
            'error': error,
            'tiempo': tiempo_total
        }
        
    except Exception as e:
        logger.error(f"Error en la simulación: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Ejecutar simulación de fondo de inversión')
    parser.add_argument('--config', default='config.yaml', help='Ruta al archivo de configuración')
    
    args = parser.parse_args()
    
    try:
        resultados = ejecutar_simulacion(args.config)
        print(f"\nSimulación completada exitosamente:")
        print(f"Error final: {resultados['error']:.6f}")
        print(f"Tiempo total: {resultados['tiempo']:.2f} segundos")
    except Exception as e:
        print(f"Error en la simulación: {e}") 