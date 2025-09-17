import os
import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, messagebox

from src.core.database import DatabaseManager
from src.config.settings import SystemConfig, ConfigManager
from src.core.logger import get_logger


class CADIXMainWindow:
    def __init__(self, config: SystemConfig, config_manager: ConfigManager):
        self.config = config
        self.config_manager = config_manager
        self.logger = get_logger()

        self.root = ctk.CTk()
        self.root.title("CADIX - Sistema Industrial de Detección de Cadenas")
        self.root.geometry(f"{self.config.ui.window_width}x{self.config.ui.window_height}")
        self.root.minsize(800, 600)

        # Base de datos (usar ruta, no el objeto config completo)
        self.db = DatabaseManager(os.path.join(self.config.base_dir, self.config.database.db_path))

        # Frames principales
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(fill="both", expand=True)

        # Panel de configuración
        config_frame = ctk.CTkFrame(self.main_frame)
        config_frame.pack(side="left", fill="y", padx=10, pady=10)

        ctk.CTkLabel(config_frame, text="Configuración", font=("Arial", 16, "bold")).pack(pady=10)

        # Conversión px→mm
        ctk.CTkLabel(config_frame, text="Factor px→mm").pack(pady=2)
        self.conversion_entry = ctk.CTkEntry(config_frame)
        self.conversion_entry.pack(pady=2)
        self.conversion_entry.insert(0, str(self.config.measurement.px_to_mm))

        # Modo one-shot congelar
        self.freeze_var = tk.BooleanVar(master=self.root, value=self.config.oneshot.freeze_after_shot)
        freeze_chk = ctk.CTkCheckBox(
            config_frame,
            text="Medición instantánea (recomendado)",
            variable=self.freeze_var,
            command=lambda: self._toggle_freeze()
        )
        freeze_chk.pack(pady=2)
        
        # Período refractario
        ctk.CTkLabel(config_frame, text="Período refractario (frames)").pack(pady=2)
        self.refractory_entry = ctk.CTkEntry(config_frame)
        self.refractory_entry.pack(pady=2)
        self.refractory_entry.insert(0, str(self.config.oneshot.refractory_frames))

        # Botones principales
        ctk.CTkButton(config_frame, text="Probar Cámara", command=self._test_camera).pack(pady=5)
        ctk.CTkButton(config_frame, text="Guardar Configuración", command=self._save_config).pack(pady=5)
        ctk.CTkButton(config_frame, text="Iniciar Detector", command=self._start_detector).pack(pady=5)
        ctk.CTkButton(config_frame, text="Detener Detector", command=self._stop_detector).pack(pady=5)
        ctk.CTkButton(config_frame, text="Salir", command=self.root.quit).pack(pady=20)

        # Área de video / resultados
        self.video_frame = ctk.CTkFrame(self.main_frame)
        self.video_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.status_label = ctk.CTkLabel(self.video_frame, text="Estado: Inactivo")
        self.status_label.pack(pady=5)

        self.detector = None

    def _save_config(self):
        try:
            self.config.measurement.px_to_mm = float(self.conversion_entry.get())
            self.config.oneshot.freeze_after_shot = bool(self.freeze_var.get())
            self.config.oneshot.refractory_frames = int(self.refractory_entry.get())
            self.config_manager.save_config(self.config)
            self.logger.info("Configuración guardada correctamente")
            return True
        except Exception as e:
            self.logger.error(f"No se pudo guardar configuración: {e}")
            return False

    def _toggle_freeze(self):
        """Actualiza la opción de congelar medición one-shot en la config"""
        try:
            self.config.oneshot.freeze_after_shot = bool(self.freeze_var.get())
            self.config_manager.save_config(self.config)
        except Exception as e:
            self.logger.error(f"No pude actualizar freeze_after_shot: {e}")

    def _start_detector(self):
        if self.detector and self.detector.is_alive():
            messagebox.showwarning("Advertencia", "El detector ya está en ejecución.")
            return
        try:
            # Guardar configuración antes de iniciar
            self._save_config()
            
            # Import dinámico aquí para evitar ciclos y mostrar errores reales de detector.py
            from src.core.detector import CADIXDetector

            self.detector = CADIXDetector(
                config=self.config,
                frame_callback=self._on_new_frame,
                stats_callback=self._on_new_stats,
                status_callback=self._on_status,
                shot_callback=self._on_shot
            )
            self.detector.start()
            self.status_label.configure(text="Estado: Iniciando detector...")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo iniciar el detector: {e}")
            self.logger.error(f"No se pudo iniciar el detector: {e}")
            self.status_label.configure(text="Estado: Error al iniciar")

    def _stop_detector(self):
        if self.detector:
            self.detector.stop()
            self.status_label.configure(text="Estado: Detector detenido")

    def _on_new_frame(self, frame):
        # TODO: mostrar imagen en interfaz (ej: con PIL + ImageTk)
        pass

    def _on_new_stats(self, *args, **kwargs):
        # TODO: actualizar estadísticas en la interfaz
        pass

    def _on_status(self, msg: str):
        self.status_label.configure(text=f"Estado: {msg}")

    def _on_shot(self, value, direction, session_id):
        # Registrar en DB
        try:
            self.db.save_oneshot(value, direction, session_id=session_id)
            self.logger.info(f"Medición registrada: {value} mm ({direction})")
        except Exception as e:
            self.logger.error(f"No se pudo registrar medición: {e}")
    
    def _test_camera(self):
        """Prueba la cámara para verificar que funciona"""
        try:
            import cv2 as cv
            import platform
            
            idx = self.config.camera.index
            
            if platform.system().lower().startswith("win"):
                api = cv.CAP_MSMF
            else:
                api = cv.CAP_ANY
            
            cap = cv.VideoCapture(idx, api)
            
            if not cap.isOpened():
                cap = cv.VideoCapture(idx, cv.CAP_ANY)
            
            if not cap.isOpened():
                # Probar otros índices
                for k in range(1, 4):
                    cap = cv.VideoCapture(k, api)
                    if cap.isOpened():
                        idx = k
                        break
            
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    h, w = frame.shape[:2]
                    messagebox.showinfo("Éxito", f"Cámara funcionando correctamente\nÍndice: {idx}\nResolución: {w}x{h}")
                else:
                    messagebox.showerror("Error", "La cámara no puede capturar frames")
                cap.release()
            else:
                messagebox.showerror("Error", "No se pudo abrir ninguna cámara")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error probando cámara: {e}")
            self.logger.error(f"Error probando cámara: {e}")

    def run(self):
        self.root.mainloop()
