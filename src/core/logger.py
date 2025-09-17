"""
CADIX - Sistema de Logging
Sistema de logging profesional con rotación de archivos y diferentes niveles.
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

class CADIXLogger:
    """Sistema de logging profesional para CADIX"""
    
    def __init__(self, 
                 name: str = "CADIX",
                 log_dir: str = "logs",
                 log_level: int = logging.INFO,
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 10,
                 console_output: bool = True):
        
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Configurar logger principal
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        
        # Evitar duplicación de handlers
        if not self.logger.handlers:
            self._setup_handlers(max_file_size, backup_count, console_output)
    
    def _setup_handlers(self, max_file_size: int, backup_count: int, console_output: bool):
        """Configura los handlers de logging"""
        
        # Formatter
        formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Handler para archivo principal con rotación
        main_log_file = self.log_dir / f"{self.name.lower()}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            main_log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        
        # Handler para errores (archivo separado)
        error_log_file = self.log_dir / f"{self.name.lower()}_errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        error_handler.setFormatter(formatter)
        error_handler.setLevel(logging.ERROR)
        
        # Handler para consola
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_formatter = logging.Formatter(
                fmt='%(levelname)-8s | %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            console_handler.setLevel(logging.INFO)
            self.logger.addHandler(console_handler)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)
    
    def debug(self, message: str, **kwargs):
        """Log de debug"""
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log de información"""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log de advertencia"""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log de error"""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log crítico"""
        self.logger.critical(message, **kwargs)
    
    def log_system_start(self, version: str):
        """Log de inicio del sistema"""
        self.info("=" * 60)
        self.info(f"CADIX Sistema Industrial v{version} - INICIADO")
        self.info(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.info("=" * 60)
    
    def log_system_shutdown(self):
        """Log de cierre del sistema"""
        self.info("=" * 60)
        self.info("CADIX Sistema - DETENIDO")
        self.info(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.info("=" * 60)
    
    def log_measurement(self, value_mm: float, direction: str, confidence: float = None):
        """Log específico para mediciones"""
        msg = f"MEDICIÓN: {value_mm:.2f}mm | Dirección: {direction}"
        if confidence:
            msg += f" | Confianza: {confidence:.3f}"
        self.info(msg)
    
    def log_alert(self, value_mm: float, threshold: float):
        """Log específico para alertas"""
        self.warning(f"ALERTA: Medición {value_mm:.2f}mm excede umbral {threshold:.2f}mm")
    
    def log_camera_event(self, event: str, details: str = ""):
        """Log específico para eventos de cámara"""
        msg = f"CÁMARA: {event}"
        if details:
            msg += f" | {details}"
        self.info(msg)
    
    def log_detection_stats(self, fps: float, detections: int, avg_confidence: float):
        """Log de estadísticas de detección"""
        self.debug(f"STATS: FPS={fps:.1f} | Detecciones={detections} | Conf.Prom={avg_confidence:.3f}")

# Logger global para toda la aplicación
_global_logger: Optional[CADIXLogger] = None

def get_logger(name: str = "CADIX") -> CADIXLogger:
    """Obtiene el logger global o crea uno nuevo"""
    global _global_logger
    if _global_logger is None:
        _global_logger = CADIXLogger(name)
    return _global_logger

def setup_logging(log_dir: str = "logs", log_level: int = logging.INFO):
    """Configura el sistema de logging global"""
    global _global_logger
    _global_logger = CADIXLogger("CADIX", log_dir, log_level)
    return _global_logger