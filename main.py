import sys, os
from pathlib import Path

# --- Ensure 'src' is importable no matter where you run main.py ---
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""
CADIX - Sistema Industrial de Detección de Cadenas v2.0
Punto de entrada principal de la aplicación.
"""

import traceback
from typing import List

from src.core.logger import setup_logging, get_logger
from src.config.settings import ConfigManager
from src.ui.main_window import CADIXMainWindow


def check_dependencies() -> bool:
    """Verifica dependencias mínimas para arrancar."""
    required = {
        "customtkinter": "customtkinter",
        "cv2": "opencv-python",
        "onnxruntime": "onnxruntime",
        "PIL": "Pillow",
        "numpy": "numpy",
    }
    missing: List[str] = []
    for module_name, package_name in required.items():
        try:
            __import__(module_name)
        except ImportError:
            missing.append(package_name)

    if missing:
        print("ERROR: Faltan las siguientes dependencias:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nInstálalas con:")
        print(f"pip install {' '.join(missing)}")
        return False
    return True


def setup_directories():
    """Crea los directorios base en %USERPROFILE%/CADIX."""
    from pathlib import Path as _Path
    base_dir = _Path.home() / "CADIX"
    dirs_to_create = [
        base_dir / "data",
        base_dir / "logs",
        base_dir / "reports",
        base_dir / "models",
        base_dir / "backups",
        base_dir / "config",   # evitar strings sueltos
    ]
    for directory in dirs_to_create:
        d = _Path(directory)
        d.mkdir(parents=True, exist_ok=True)


def main() -> int:
    try:
        print("=" * 60)
        print("CADIX - Sistema Industrial de Detección de Cadenas v2.0")
        print("=" * 60)

        # 1) Dependencias
        print("Verificando dependencias...")
        if not check_dependencies():
            input("\nPresiona Enter para salir...")
            return 1

        # 2) Directorios
        print("Configurando directorios...")
        setup_directories()

        # 3) Logging
        print("Inicializando sistema de logging...")
        logger = setup_logging()
        logger.info("Logging iniciado correctamente")

        # 4) Configuración
        print("Iniciando interfaz gráfica...")
        config_manager = ConfigManager()            # <-- NUEVO
        config = config_manager.get_config()        # <-- NUEVO

        # 5) UI (pasar config y manager)
        app = CADIXMainWindow(config, config_manager)   # <-- PASAMOS ARGUMENTOS
        return app.run() or 0

    except Exception as e:
        print("\nERROR CRÍTICO:", str(e))
        print("\nDetalles del error:")
        print(traceback.format_exc())

        try:
            logger = get_logger()
            logger.critical(f"Error crítico en aplicación: {e}")
            logger.critical(traceback.format_exc())
        except Exception:
            pass

        input("\nPresiona Enter para salir...")
        return 1


if __name__ == "__main__":
    sys.exit(main())
