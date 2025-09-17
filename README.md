# CADIX - Sistema Industrial de Detección de Cadenas v2.0

Sistema profesional de detección y medición de eslabones de cadena utilizando visión artificial con IA.

## 🚀 Características Principales

### Detección con IA
- ✅ Modelo YOLO optimizado para detección de eslabones
- ✅ Tracking robusto con recuperación automática
- ✅ Procesamiento en tiempo real (30+ FPS)
- ✅ Mejora automática de imagen en condiciones de poca luz

### Medición Precisa
- ✅ Medición continua con suavizado EMA
- ✅ Sistema "One-Shot" para mediciones puntuales
- ✅ Calibración automática px→mm
- ✅ Alertas configurables por umbral

### Base de Datos Profesional
- ✅ SQLite para almacenamiento robusto
- ✅ Historial completo de mediciones
- ✅ Gestión de sesiones por operador/turno
- ✅ Backup automático de datos

### Interfaz Industrial
- ✅ Dashboard en tiempo real
- ✅ Gráficos de tendencias
- ✅ Métricas de performance
- ✅ Controles intuitivos

### Reportes y Exportación
- ✅ Exportación a CSV/PDF
- ✅ Reportes por sesión
- ✅ Estadísticas avanzadas
- ✅ Configuración por perfiles

## 📋 Requisitos del Sistema

### Hardware Mínimo
- **CPU**: Intel i5 o AMD equivalent (4+ cores)
- **RAM**: 8GB mínimo, 16GB recomendado
- **Cámara**: USB 2.0+ compatible con OpenCV
- **OS**: Windows 10/11, Linux Ubuntu 18+

### Software
- Python 3.8 - 3.11
- OpenCV 4.5+
- ONNX Runtime
- Espacio libre: 2GB+

## 🔧 Instalación

### 1. Clonar el Repositorio
```bash
git clone https://github.com/tu-usuario/cadix-industrial.git
cd cadix-industrial
```

### 2. Crear Entorno Virtual
```bash
python -m venv cadix_env

# Windows
cadix_env\Scripts\activate

# Linux/Mac
source cadix_env/bin/activate
```

### 3. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 4. Configurar Modelo
- Coloca tu modelo YOLO (.onnx) en la carpeta `models/`
- Actualiza la ruta en `config/settings.json`

### 5. Ejecutar
```bash
python main.py
```

## 📖 Uso del Sistema

### Configuración Inicial

1. **Iniciar la aplicación**
   ```bash
   python main.py
   ```

2. **Configurar cámara**
   - El sistema detecta automáticamente las cámaras disponibles
   - Ajusta resolución en Configuración si es necesario

3. **Calibrar medición**
   - Coloca un objeto de dimensión conocida
   - Ajusta el factor px→mm en configuración rápida

### Operación Normal

1. **Iniciar sesión**
   - Ingresa nombre del operador
   - Selecciona turno de trabajo
   - Presiona "INICIAR"

2. **Monitoreo en tiempo real**
   - Vista de cámara con overlays de detección
   - Métricas actualizadas en tiempo real
   - Gráfico de tendencias

3. **Mediciones One-Shot**
   - Se activan automáticamente cuando el eslabón cruza la línea central
   - Guardado automático en base de datos
   - Visualización en panel de "Últimas Mediciones"

4. **Alertas**
   - Alerta visual y sonora cuando se supera el umbral
   - Registro automático en sistema de eventos

### Configuración Avanzada

El archivo `config/settings.json` permite configurar:

```json
{
  "camera": {
    "index": 0,
    "width": 1280,
    "height": 720,
    "fps": 30
  },
  "detection": {
    "model_path": "models/best_y8n.onnx",
    "confidence_threshold": 0.25,
    "iou_threshold": 0.50
  },
  "measurement": {
    "px_to_mm": 0.86,
    "alert_threshold_mm": 120.0,
    "ema_alpha": 0.35
  }
}
```

## 📊 Estructura de Datos

### Base de Datos
El sistema utiliza SQLite con las siguientes tablas principales:

- **measurements**: Mediciones continuas
- **oneshot_measurements**: Mediciones puntuales
- **alerts**: Registro de alertas
- **sessions**: Sesiones de trabajo
- **system_events**: Eventos del sistema

### Exportación
Los datos se pueden exportar en formatos:
- CSV para análisis en Excel/Python
- PDF para reportes ejecutivos
- JSON para integración con otros sistemas

## 🔍 Troubleshooting

### Problemas Comunes

**Error: Cámara no detectada**
```
Solución:
1. Verificar que la cámara esté conectada
2. Cerrar otras aplicaciones que usen la cámara
3. Probar diferentes índices de cámara (0, 1, 2...)
```

**Error: Modelo ONNX no cargado**
```
Solución:
1. Verificar ruta del modelo en configuración
2. Asegurar que el archivo .onnx existe
3. Verificar que ONNX Runtime está instalado
```

**Performance lento**
```
Solución:
1. Reducir resolución de cámara
2. Aumentar umbral de confianza
3. Verificar CPU/RAM disponible
```

### Logs del Sistema
Los logs se guardan en:
- `logs/cadix.log`: Log general
- `logs/cadix_errors.log`: Solo errores

## 🏭 Uso Industrial

### Recomendaciones de Despliegue

**Hardware Industrial**
- PC industrial con certificación IP65+
- Cámara industrial con iluminación LED
- UPS para protección contra cortes
- Monitor táctil resistente

**Configuración de Red**
- Conectividad para backup automático
- Acceso remoto para mantenimiento
- Integración con sistemas MES/ERP

**Mantenimiento**
- Backup automático configurado
- Logs rotativos para no saturar disco
- Alertas por email/SMS (configurables)

## 🤝 Soporte y Contribuciones

### Reportar Problemas
1. Crear issue en GitHub
2. Incluir logs relevantes
3. Describir pasos para reproducir

### Desarrollo
1. Fork del repositorio
2. Crear branch para nueva característica
3. Enviar pull request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## 📞 Contacto

**CADIX Industries**
- Email: soporte@cadix-industries.com
- Web: www.cadix-industries.com
- Teléfono: +XX XXX XXX XXXX

---

*CADIX v2.0 - Desarrollado para la industria moderna*