"""
CADIX - Configuración del Sistema
Configuraciones centralizadas para el sistema de detección industrial.
"""

import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, Any
from pathlib import Path

@dataclass
class CameraConfig:
    """Configuración de cámara"""
    index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    buffer_size: int = 1
    auto_exposure: bool = False
    brightness: float = 0.0
    contrast: float = 1.0

@dataclass
class DetectionConfig:
    """Configuración de detección"""
    model_path: str = "models/best_y8n.onnx"
    input_size: int = 480
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.50
    lock_iou_threshold: float = 0.35
    lock_max_misses: int = 10
    lock_smoothing: float = 0.55

@dataclass
class MeasurementConfig:
    """Configuración de medición"""
    px_to_mm: float = 0.86
    margin_fraction: float = 0.03
    ema_alpha: float = 0.35
    alert_threshold_mm: float = 120.0
    alert_cooldown_s: float = 1.0
    use_ema_for_shots: bool = False

@dataclass
class OneShotConfig:
    """Configuración de One-Shot"""
    gate_x_ratio: float = 0.50
    gate_y_tolerance_fraction: float = 0.22
    refractory_frames: int = 30
    shot_direction: str = "any"  # "any", "L2R", "R2L"
    freeze_after_shot: bool = True
    release_lock_after_shot: bool = True
    
@dataclass
class ImageProcessingConfig:
    """Configuración de procesamiento de imagen"""
    low_light_enhance: bool = True
    clahe_clip_limit: float = 1.5
    clahe_tile_size: tuple = (8, 8)
    denoise_on_dark: bool = False

@dataclass
class UIConfig:
    """Configuración de interfaz"""
    window_width: int = 1400
    window_height: int = 900
    video_width: int = 960
    video_height: int = 540
    theme: str = "light"
    color_theme: str = "blue"

@dataclass
class DatabaseConfig:
    """Configuración de base de datos"""
    db_path: str = "data/cadix.db"
    backup_interval_hours: int = 24
    max_backup_files: int = 30

@dataclass
class SystemConfig:
    """Configuración principal del sistema"""
    camera: CameraConfig
    detection: DetectionConfig
    measurement: MeasurementConfig
    oneshot: OneShotConfig
    image_processing: ImageProcessingConfig
    ui: UIConfig
    database: DatabaseConfig
    
    # Rutas del sistema
    base_dir: str = ""
    data_dir: str = "data"
    logs_dir: str = "logs"
    reports_dir: str = "reports"
    models_dir: str = "models"
    
    # Información del sistema
    version: str = "2.0.0"
    company: str = "CADIX Industries"
    
    def __post_init__(self):
        if not self.base_dir:
            self.base_dir = str(Path.home() / "CADIX")
        
        # Crear directorios necesarios
        for dir_name in [self.data_dir, self.logs_dir, self.reports_dir, self.models_dir]:
            full_path = os.path.join(self.base_dir, dir_name)
            os.makedirs(full_path, exist_ok=True)

class ConfigManager:
    """Gestor de configuraciones"""
    
    def __init__(self, config_file: str = "config/settings.json"):
        self.config_file = config_file
        self.config = self._load_config()
    
    def _create_default_config(self) -> SystemConfig:
        """Crea configuración por defecto"""
        return SystemConfig(
            camera=CameraConfig(),
            detection=DetectionConfig(),
            measurement=MeasurementConfig(),
            oneshot=OneShotConfig(),
            image_processing=ImageProcessingConfig(),
            ui=UIConfig(),
            database=DatabaseConfig()
        )
    
    def _load_config(self) -> SystemConfig:
        """Carga configuración desde archivo"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                return SystemConfig(
                    camera=CameraConfig(**data.get('camera', {})),
                    detection=DetectionConfig(**data.get('detection', {})),
                    measurement=MeasurementConfig(**data.get('measurement', {})),
                    oneshot=OneShotConfig(**data.get('oneshot', {})),
                    image_processing=ImageProcessingConfig(**data.get('image_processing', {})),
                    ui=UIConfig(**data.get('ui', {})),
                    database=DatabaseConfig(**data.get('database', {})),
                    **{k: v for k, v in data.items() if k not in [
                        'camera', 'detection', 'measurement', 'oneshot', 
                        'image_processing', 'ui', 'database'
                    ]}
                )
            else:
                config = self._create_default_config()
                self.save_config(config)
                return config
                
        except Exception as e:
            print(f"Error cargando configuración: {e}")
            return self._create_default_config()
    
    def save_config(self, config: SystemConfig = None) -> bool:
        """Guarda configuración a archivo"""
        try:
            if config is None:
                config = self.config
            
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            
            config_dict = {
                'camera': asdict(config.camera),
                'detection': asdict(config.detection),
                'measurement': asdict(config.measurement),
                'oneshot': asdict(config.oneshot),
                'image_processing': asdict(config.image_processing),
                'ui': asdict(config.ui),
                'database': asdict(config.database),
                'base_dir': config.base_dir,
                'data_dir': config.data_dir,
                'logs_dir': config.logs_dir,
                'reports_dir': config.reports_dir,
                'models_dir': config.models_dir,
                'version': config.version,
                'company': config.company
            }
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=4, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            print(f"Error guardando configuración: {e}")
            return False
    
    def get_config(self) -> SystemConfig:
        """Obtiene configuración actual"""
        return self.config
    
    def update_config(self, **kwargs) -> bool:
        """Actualiza configuración"""
        try:
            for key, value in kwargs.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            return self.save_config()
        except Exception as e:
            print(f"Error actualizando configuración: {e}")
            return False