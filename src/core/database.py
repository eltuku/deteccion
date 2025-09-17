"""
CADIX - Sistema de Base de Datos
Gestión de base de datos SQLite para almacenar mediciones y configuraciones.
"""

import sqlite3
import threading
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import shutil

from src.core.logger import get_logger

class DatabaseManager:
    """Gestor de base de datos para CADIX"""
    
    def __init__(self, db_path: str = "data/cadix.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        
        self.logger = get_logger()
        self._lock = threading.Lock()
        
        # Inicializar base de datos
        self._init_database()
        self.logger.info(f"Base de datos inicializada: {self.db_path}")
    
    def _init_database(self):
        """Inicializa las tablas de la base de datos"""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            try:
                # Tabla de mediciones continuas
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS measurements (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        value_mm REAL NOT NULL,
                        value_cm REAL NOT NULL,
                        ema_value REAL,
                        confidence REAL,
                        bbox_x1 INTEGER,
                        bbox_y1 INTEGER,
                        bbox_x2 INTEGER,
                        bbox_y2 INTEGER,
                        fps REAL,
                        session_id TEXT
                    )
                ''')
                
                # Tabla de mediciones one-shot
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS oneshot_measurements (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        value_mm REAL NOT NULL,
                        value_cm REAL NOT NULL,
                        direction TEXT NOT NULL,
                        fps REAL,
                        session_id TEXT,
                        operator TEXT,
                        shift TEXT,
                        notes TEXT
                    )
                ''')
                
                # Tabla de alertas
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        alert_type TEXT NOT NULL,
                        value_mm REAL NOT NULL,
                        threshold_mm REAL NOT NULL,
                        message TEXT,
                        acknowledged BOOLEAN DEFAULT FALSE,
                        session_id TEXT
                    )
                ''')
                
                # Tabla de sesiones
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS sessions (
                        id TEXT PRIMARY KEY,
                        start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                        end_time DATETIME,
                        operator TEXT,
                        shift TEXT,
                        total_measurements INTEGER DEFAULT 0,
                        total_oneshots INTEGER DEFAULT 0,
                        total_alerts INTEGER DEFAULT 0,
                        status TEXT DEFAULT 'active'
                    )
                ''')
                
                # Tabla de configuraciones
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS configurations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        config_type TEXT NOT NULL,
                        config_data TEXT NOT NULL,
                        operator TEXT,
                        description TEXT
                    )
                ''')
                
                # Tabla de eventos del sistema
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS system_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        event_type TEXT NOT NULL,
                        event_data TEXT,
                        severity TEXT DEFAULT 'info',
                        session_id TEXT
                    )
                ''')
                
                # Crear índices para mejor performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_measurements_timestamp ON measurements(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_oneshot_timestamp ON oneshot_measurements(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_timestamp ON system_events(timestamp)')
                
                conn.commit()
                
            except Exception as e:
                self.logger.error(f"Error inicializando base de datos: {e}")
                conn.rollback()
                raise
            finally:
                conn.close()
    
    def start_session(self, operator: str = "", shift: str = "") -> str:
        """Inicia una nueva sesión"""
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            try:
                cursor.execute('''
                    INSERT INTO sessions (id, operator, shift)
                    VALUES (?, ?, ?)
                ''', (session_id, operator, shift))
                
                conn.commit()
                self.logger.info(f"Nueva sesión iniciada: {session_id} | Operador: {operator} | Turno: {shift}")
                
            except Exception as e:
                self.logger.error(f"Error iniciando sesión: {e}")
                conn.rollback()
            finally:
                conn.close()
        
        return session_id
    
    def end_session(self, session_id: str):
        """Finaliza una sesión"""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            try:
                # Obtener estadísticas de la sesión
                cursor.execute('SELECT COUNT(*) FROM measurements WHERE session_id = ?', (session_id,))
                total_measurements = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) FROM oneshot_measurements WHERE session_id = ?', (session_id,))
                total_oneshots = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) FROM alerts WHERE session_id = ?', (session_id,))
                total_alerts = cursor.fetchone()[0]
                
                # Actualizar sesión
                cursor.execute('''
                    UPDATE sessions 
                    SET end_time = CURRENT_TIMESTAMP,
                        total_measurements = ?,
                        total_oneshots = ?,
                        total_alerts = ?,
                        status = 'completed'
                    WHERE id = ?
                ''', (total_measurements, total_oneshots, total_alerts, session_id))
                
                conn.commit()
                self.logger.info(f"Sesión finalizada: {session_id} | Mediciones: {total_measurements} | One-shots: {total_oneshots} | Alertas: {total_alerts}")
                
            except Exception as e:
                self.logger.error(f"Error finalizando sesión: {e}")
                conn.rollback()
            finally:
                conn.close()
    
    def save_measurement(self, 
                        value_mm: float, 
                        ema_value: float = None, 
                        confidence: float = None,
                        bbox: Tuple[int, int, int, int] = None,
                        fps: float = None,
                        session_id: str = None):
        """Guarda una medición continua"""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            try:
                bbox_values = bbox if bbox else (None, None, None, None)
                
                cursor.execute('''
                    INSERT INTO measurements 
                    (value_mm, value_cm, ema_value, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2, fps, session_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (value_mm, value_mm/10.0, ema_value, confidence, 
                      bbox_values[0], bbox_values[1], bbox_values[2], bbox_values[3], 
                      fps, session_id))
                
                conn.commit()
                
            except Exception as e:
                self.logger.error(f"Error guardando medición: {e}")
                conn.rollback()
            finally:
                conn.close()
    
    def save_oneshot(self, 
                    value_mm: float, 
                    direction: str,
                    fps: float = None,
                    session_id: str = None,
                    operator: str = "",
                    shift: str = "",
                    notes: str = ""):
        """Guarda una medición one-shot"""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            try:
                cursor.execute('''
                    INSERT INTO oneshot_measurements 
                    (value_mm, value_cm, direction, fps, session_id, operator, shift, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (value_mm, value_mm/10.0, direction, fps, session_id, operator, shift, notes))
                
                conn.commit()
                self.logger.log_measurement(value_mm, direction)
                
            except Exception as e:
                self.logger.error(f"Error guardando one-shot: {e}")
                conn.rollback()
            finally:
                conn.close()
    
    def save_alert(self, 
                   alert_type: str,
                   value_mm: float,
                   threshold_mm: float,
                   message: str = "",
                   session_id: str = None):
        """Guarda una alerta"""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            try:
                cursor.execute('''
                    INSERT INTO alerts 
                    (alert_type, value_mm, threshold_mm, message, session_id)
                    VALUES (?, ?, ?, ?, ?)
                ''', (alert_type, value_mm, threshold_mm, message, session_id))
                
                conn.commit()
                self.logger.log_alert(value_mm, threshold_mm)
                
            except Exception as e:
                self.logger.error(f"Error guardando alerta: {e}")
                conn.rollback()
            finally:
                conn.close()
    
    def get_recent_oneshots(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Obtiene las últimas mediciones one-shot"""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            try:
                cursor.execute('''
                    SELECT timestamp, value_mm, value_cm, direction, operator
                    FROM oneshot_measurements
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (limit,))
                
                rows = cursor.fetchall()
                return [
                    {
                        'timestamp': row[0],
                        'value_mm': row[1],
                        'value_cm': row[2],
                        'direction': row[3],
                        'operator': row[4] or ""
                    }
                    for row in rows
                ]
                
            except Exception as e:
                self.logger.error(f"Error obteniendo one-shots recientes: {e}")
                return []
            finally:
                conn.close()
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Obtiene estadísticas de una sesión"""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            try:
                # Información de la sesión
                cursor.execute('SELECT * FROM sessions WHERE id = ?', (session_id,))
                session_data = cursor.fetchone()
                
                if not session_data:
                    return {}
                
                # Estadísticas de mediciones
                cursor.execute('''
                    SELECT COUNT(*), AVG(value_mm), MIN(value_mm), MAX(value_mm)
                    FROM measurements WHERE session_id = ?
                ''', (session_id,))
                measurement_stats = cursor.fetchone()
                
                # Estadísticas de one-shots
                cursor.execute('''
                    SELECT COUNT(*), AVG(value_mm), direction, COUNT(*) as dir_count
                    FROM oneshot_measurements 
                    WHERE session_id = ? 
                    GROUP BY direction
                ''', (session_id,))
                oneshot_stats = cursor.fetchall()
                
                return {
                    'session_id': session_data[0],
                    'start_time': session_data[1],
                    'end_time': session_data[2],
                    'operator': session_data[3],
                    'shift': session_data[4],
                    'measurements': {
                        'count': measurement_stats[0],
                        'avg_mm': measurement_stats[1],
                        'min_mm': measurement_stats[2],
                        'max_mm': measurement_stats[3]
                    },
                    'oneshots': {
                        'total': len(oneshot_stats),
                        'by_direction': {row[2]: row[3] for row in oneshot_stats}
                    }
                }
                
            except Exception as e:
                self.logger.error(f"Error obteniendo estadísticas de sesión: {e}")
                return {}
            finally:
                conn.close()
    
    def backup_database(self, backup_dir: str = "backups") -> bool:
        """Crea backup de la base de datos"""
        try:
            backup_path = Path(backup_dir)
            backup_path.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = backup_path / f"cadix_backup_{timestamp}.db"
            
            shutil.copy2(self.db_path, backup_file)
            
            self.logger.info(f"Backup creado: {backup_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creando backup: {e}")
            return False
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """Limpia datos antiguos"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            try:
                # Limpiar mediciones antiguas
                cursor.execute('DELETE FROM measurements WHERE timestamp < ?', (cutoff_date,))
                measurements_deleted = cursor.rowcount
                
                # Limpiar alertas antiguas
                cursor.execute('DELETE FROM alerts WHERE timestamp < ?', (cutoff_date,))
                alerts_deleted = cursor.rowcount
                
                conn.commit()
                
                self.logger.info(f"Limpieza completada: {measurements_deleted} mediciones y {alerts_deleted} alertas eliminadas")
                
            except Exception as e:
                self.logger.error(f"Error en limpieza de datos: {e}")
                conn.rollback()
            finally:
                conn.close()
    
    def execute_query(self, query: str, params: tuple = ()) -> List[tuple]:
        """Ejecuta una consulta personalizada (solo SELECT)"""
        if not query.strip().upper().startswith('SELECT'):
            raise ValueError("Solo se permiten consultas SELECT")
        
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            try:
                cursor.execute(query, params)
                return cursor.fetchall()
            except Exception as e:
                self.logger.error(f"Error ejecutando consulta: {e}")
                return []
            finally:
                conn.close()