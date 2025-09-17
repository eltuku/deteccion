"""
CADIX - Detector Principal
Detector con manejo robusto de cámara, inferencia ONNX y medición one-shot.
Incluye modo "freeze" para fijar la medición en el instante del disparo.
"""

import time
import threading
import numpy as np
import cv2 as cv
import onnxruntime as ort
from typing import Tuple, List, Optional, Callable
from datetime import datetime
import platform

from src.core.logger import get_logger
from src.config.settings import SystemConfig


class CADIXDetector(threading.Thread):
    """
    Hilo de detección principal.
    - Lee cámara
    - Corre inferencia ONNX
    - Calcula medición (mm)
    - Dispara one-shot y congela valor si está habilitado
    - Envía frames y stats por callbacks
    """

    def __init__(
        self,
        config: SystemConfig,
        frame_callback: Callable,
        stats_callback: Callable,
        status_callback: Callable,
        shot_callback: Optional[Callable] = None,
        session_id: Optional[str] = None,
    ):
        super().__init__(daemon=True)
        self.config = config
        self.frame_cb = frame_callback
        self.stats_cb = stats_callback
        self.status_cb = status_callback
        self.shot_cb = shot_callback
        self.session_id = session_id or datetime.now().strftime("%Y%m%d-%H%M%S")

        self.logger = get_logger()
        self.stop_flag = threading.Event()

        # Variables de estado
        self.fps = 0.0
        self.ema_val = None
        self.hist: List[float] = []

        # Tracking
        self.lock_bbox: Optional[Tuple[int, int, int, int]] = None
        self.lock_bbox_smooth: Optional[Tuple[float, float, float, float]] = None
        self.lock_misses = 0
        self.prev_center: Optional[Tuple[int, int]] = None
        self.refractory = 0

        # One-shot freeze state
        self.frozen_value: Optional[float] = None
        self.frozen_frames: int = 0

        self.last_beep_t = 0.0

        # Métricas
        self.frame_count = 0
        self.detection_count = 0
        self.alert_count = 0
        self.avg_confidence = 0.0

        # Contadores FPS
        self._fps_n = 0
        self._fps_t0 = time.time()
        self._plot_tick = 0

        # Recursos
        self.cap: Optional[cv.VideoCapture] = None
        self.sess: Optional[ort.InferenceSession] = None
        self.input_name = None
        self.output_name = None

        # Inicializar
        self._init_onnx_session()
        self._init_camera()

        self.logger.info("Detector inicializado correctamente")

    # ----------------------------
    # Inicialización de ONNX
    # ----------------------------
    def _init_onnx_session(self):
        try:
            providers = ort.get_available_providers()
            if "CPUExecutionProvider" not in providers:
                providers.append("CPUExecutionProvider")

            model_path = self.config.detection.model_path
            self.sess = ort.InferenceSession(model_path, providers=providers)
            self.input_name = self.sess.get_inputs()[0].name
            self.output_name = self.sess.get_outputs()[0].name
            self.logger.info(f"Sesión ONNX OK: {model_path}")
        except Exception as e:
            self.logger.error(f"Error iniciando sesión ONNX: {e}")
            raise

    # ----------------------------
    # Inicialización de cámara (sin depender de camera.backend)
    # ----------------------------
    def _init_camera(self):
        try:
            idx = int(getattr(self.config.camera, "index", 0))

            if platform.system().lower().startswith("win"):
                api = cv.CAP_MSMF
            else:
                api = cv.CAP_ANY

            cap = cv.VideoCapture(idx, api)

            if not cap.isOpened():
                cap = cv.VideoCapture(idx, cv.CAP_ANY)

            if not cap.isOpened():
                # probar otros índices
                for k in range(1, 4):
                    cap = cv.VideoCapture(k, api)
                    if cap.isOpened():
                        idx = k
                        break

            if not cap.isOpened():
                raise RuntimeError("No se pudo abrir ninguna cámara")

            cap.set(cv.CAP_PROP_FPS, int(getattr(self.config.camera, "fps", 30)))
            cap.set(cv.CAP_PROP_FRAME_WIDTH, int(getattr(self.config.camera, "width", 1280)))
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, int(getattr(self.config.camera, "height", 720)))

            buf = getattr(self.config.camera, "buffersize", None)
            if buf is not None:
                try:
                    cap.set(cv.CAP_PROP_BUFFERSIZE, int(buf))
                except Exception:
                    pass

            self.cap = cap
            self.status_cb(f"Cámara abierta (index={idx})")

        except Exception as e:
            self.logger.error(f"Error iniciando cámara: {e}")
            raise

    # ----------------------------
    # Loop principal
    # ----------------------------
    def run(self):
        try:
            self.status_cb("Detector en ejecución")

            while not self.stop_flag.is_set():
                start_time = time.time()

                ok, frame = self.cap.read() if self.cap else (False, None)
                if not ok or frame is None:
                    time.sleep(0.01)
                    continue

                self.frame_count += 1

                # Mejoras de imagen
                bgr = self._enhance_image(frame)
                H, W = bgr.shape[:2]

                # Preparar input
                letterboxed, ratio, (dw, dh) = self._letterbox(
                    bgr, (self.config.detection.input_size, self.config.detection.input_size)
                )
                rgb = cv.cvtColor(letterboxed, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
                blob = np.transpose(rgb, (2, 0, 1))[None, :, :, :]

                # Inferencia
                try:
                    outputs = self.sess.run([self.output_name], {self.input_name: blob})
                    out = outputs[0] if outputs else None
                except Exception as e:
                    self.logger.error(f"Error en inferencia ONNX: {e}")
                    continue

                # Postproceso (asume [N,6] x1,y1,x2,y2,score,cls)
                boxes, scores = self._postprocess(out, ratio, dw, dh, W, H)

                # Tracking simple
                tracked_box, confidence = self._track_bbox(boxes, scores)

                # Medición
                mm_dist = None
                measurement_line = None

                if tracked_box is not None:
                    if self.frozen_value is not None:
                        mm_dist = self.frozen_value
                        measurement_line = None
                    else:
                        mm_dist, measurement_line = self._calculate_measurement(tracked_box, W, H)

                    # One-shot
                    self._check_oneshot(tracked_box, mm_dist, W, H)
                    # Alertas
                    self._check_alerts(mm_dist)

                # Dibujos
                output_image = bgr.copy()
                self._draw_overlays(output_image, boxes, scores, tracked_box, measurement_line, W, H)
                self._draw_info(output_image, mm_dist or 0, tracked_box is not None, W, H)

                # Frame callback
                try:
                    self.frame_cb(output_image)
                except Exception as e:
                    self.logger.error(f"Error en callback de frame: {e}")

                # Stats
                self._update_fps()
                self._update_stats(mm_dist, confidence)

                # Congelado
                if self.frozen_value is not None:
                    self.frozen_frames -= 1
                    if self.frozen_frames <= 0:
                        self.frozen_value = None

                # Ritmo
                elapsed = time.time() - start_time
                delay = max(0.0, (1.0 / max(1, self.config.camera.fps)) - elapsed)
                if delay > 0:
                    time.sleep(delay)

        except Exception as e:
            self.logger.error(f"Error en loop del detector: {e}")
        finally:
            try:
                if self.cap:
                    self.cap.release()
                    self.cap = None
                self.status_cb("Detector detenido")
            except Exception as e:
                self.logger.error(f"Error liberando cámara: {e}")

    def stop(self):
        self.stop_flag.set()

    # ----------------------------
    # Post-proceso de detecciones
    # ----------------------------
    def _postprocess(self, out: np.ndarray, ratio, dw, dh, W, H):
        if out is None or len(out) == 0:
            return [], []
        if out.ndim == 3:
            out = out[0]

        boxes = []
        scores = []
        for det in out:
            if len(det) < 6:
                continue
            x1, y1, x2, y2, sc, cls = det[:6]
            if sc < self.config.detection.confidence_threshold:
                continue

            # Deshacer letterbox
            x1 = (x1 - dw) / ratio
            y1 = (y1 - dh) / ratio
            x2 = (x2 - dw) / ratio
            y2 = (y2 - dh) / ratio

            x1 = int(np.clip(x1, 0, W - 1))
            y1 = int(np.clip(y1, 0, H - 1))
            x2 = int(np.clip(x2, 0, W - 1))
            y2 = int(np.clip(y2, 0, H - 1))

            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
                scores.append(float(sc))

        return boxes, scores

    # ----------------------------
    # Tracking simple por IoU
    # ----------------------------
    def _bbox_iou(self, a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        iw = max(0, inter_x2 - inter_x1)
        ih = max(0, inter_y2 - inter_y1)
        inter = iw * ih
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = max(area_a + area_b - inter, 1e-6)
        return inter / union

    def _track_bbox(self, boxes, scores):
        chosen = None
        conf = 0.0

        if not boxes:
            if self.lock_bbox is not None:
                self.lock_misses += 1
                if self.lock_misses > 5:
                    self.lock_bbox = None
                    self.lock_bbox_smooth = None
            return None, 0.0

        self.lock_misses = 0

        if self.lock_bbox is None:
            idx = int(np.argmax(scores))
            chosen = boxes[idx]
            conf = scores[idx]
            self.lock_bbox = chosen
            self.lock_bbox_smooth = chosen
        else:
            best_iou = -1.0
            best_idx = -1
            for i, b in enumerate(boxes):
                iou = self._bbox_iou(self.lock_bbox, b)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            if best_idx >= 0:
                chosen = boxes[best_idx]
                conf = scores[best_idx]
                self.lock_bbox = chosen

            if self.lock_bbox_smooth is None:
                self.lock_bbox_smooth = self.lock_bbox
            else:
                sx1, sy1, sx2, sy2 = self.lock_bbox_smooth
                x1, y1, x2, y2 = self.lock_bbox
                alpha = 0.4
                self.lock_bbox_smooth = (
                    (1 - alpha) * sx1 + alpha * x1,
                    (1 - alpha) * sy1 + alpha * y1,
                    (1 - alpha) * sx2 + alpha * x2,
                    (1 - alpha) * sy2 + alpha * y2,
                )

        if chosen is not None:
            self.detection_count += 1
        return chosen, conf

    # ----------------------------
    # Cálculo de medición (ejemplo)
    # ----------------------------
    def _calculate_measurement(self, bbox, W, H):
        x1, y1, x2, y2 = bbox
        px_width = max(1, x2 - x1)
        mm_per_px = float(self.config.measurement.px_to_mm)
        mm_dist = float(px_width * mm_per_px)
        measurement_line = ((x1, (y1 + y2) // 2), (x2, (y1 + y2) // 2))
        return mm_dist, measurement_line

    # ----------------------------
    # One-shot y freeze
    # ----------------------------
    def _check_oneshot(self, bbox, mm_dist, W, H):
        if bbox is None:
            self.prev_center = None
            return

        x1, y1, x2, y2 = bbox
        cx, cy = int(0.5 * (x1 + x2)), int(0.5 * (y1 + y2))

        gate_x = int(self.config.oneshot.gate_x_ratio * W)
        gate_y = H // 2
        y_tol = int(self.config.oneshot.gate_y_tolerance_fraction * H)

        if self.prev_center is None:
            self.prev_center = (cx, cy)
            return

        if self.refractory > 0:
            self.refractory -= 1

        prev_side = np.sign(self.prev_center[0] - gate_x)
        curr_side = np.sign(cx - gate_x)
        crossed = (prev_side != 0) and (curr_side != 0) and (prev_side != curr_side)

        shot_dir = self.config.oneshot.shot_direction
        dir_ok = (
            shot_dir == "any"
            or (shot_dir == "L2R" and self.prev_center[0] < gate_x <= cx)
            or (shot_dir == "R2L" and self.prev_center[0] > gate_x >= cx)
        )

        near_center_y = abs(cy - gate_y) <= y_tol

        if crossed and dir_ok and near_center_y and self.refractory == 0 and mm_dist is not None:
            value = float(mm_dist)
            direction = "L2R" if self.prev_center[0] < cx else "R2L"

            if self.shot_cb:
                try:
                    self.shot_cb(value, direction, self.session_id)
                except Exception as e:
                    self.logger.error(f"Error en callback de one-shot: {e}")

            # Siempre congelar la medición después del disparo para evitar mediciones continuas
            if getattr(self.config.oneshot, "freeze_after_shot", True):
                self.frozen_value = value
                self.frozen_frames = int(self.config.oneshot.refractory_frames * 2)  # Congelar por más tiempo
            
            # Liberar el lock del tracking para permitir nueva detección
            if getattr(self.config.oneshot, "release_lock_after_shot", True):
                self.lock_bbox = None
                self.lock_bbox_smooth = None

            self.refractory = int(self.config.oneshot.refractory_frames)
            self.logger.info(f"One-shot disparado: {value:.2f}mm ({direction})")

        self.prev_center = (cx, cy)

    # ----------------------------
    # Alertas
    # ----------------------------
    def _check_alerts(self, mm_dist):
        if mm_dist is None:
            return
        if mm_dist > self.config.measurement.alert_threshold_mm:
            now = time.time()
            if now - self.last_beep_t >= self.config.measurement.alert_cooldown_s:
                self._trigger_alert()
                self.last_beep_t = now
                self.alert_count += 1

    def _trigger_alert(self):
        try:
            if platform.system().lower().startswith("win"):
                try:
                    import winsound
                    winsound.Beep(1200, 150)
                except Exception:
                    print("\a", end="", flush=True)
            else:
                print("\a", end="", flush=True)
        except Exception as e:
            self.logger.error(f"Error reproduciendo alerta: {e}")

    # ----------------------------
    # Overlays / Info
    # ----------------------------
    def _draw_overlays(self, image, boxes, scores, tracked_bbox, measurement_line, W, H):
        try:
            for i, b in enumerate(boxes):
                x1, y1, x2, y2 = b
                color = (0, 200, 0) if (tracked_bbox and b == tracked_bbox) else (80, 80, 80)
                cv.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv.putText(image, f"{scores[i]:.2f}", (x1, y1 - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            if measurement_line is not None and self.frozen_value is None:
                (x1, y), (x2, _) = measurement_line
                cv.line(image, (x1, y), (x2, y), (50, 200, 255), 2)

            gate_x = int(self.config.oneshot.gate_x_ratio * W)
            cv.line(image, (gate_x, 0), (gate_x, H), (180, 180, 255), 1, cv.LINE_AA)
        except Exception as e:
            self.logger.error(f"Error dibujando overlays: {e}")

    def _draw_info(self, image, mm_dist, has_detection, W, H):
        try:
            status_text = "DETECTADO" if has_detection else "BUSCANDO"
            status_color = (60, 200, 120) if has_detection else (120, 120, 120)
            cv.putText(image, status_text, (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

            if mm_dist:
                txt = f"{mm_dist/10.0:.2f} cm  ({mm_dist:.1f} mm)"
                if self.ema_val:
                    txt += f"  EMA {self.ema_val:.1f}"
                if self.frozen_value is not None:
                    txt += "  [FREEZE]"
                cv.putText(image, txt, (10, 45), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 60), 2)

            cv.putText(image, f"FPS: {self.fps:.1f}", (10, 70), cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        except Exception as e:
            self.logger.error(f"Error dibujando info: {e}")

    # ----------------------------
    # Métricas
    # ----------------------------
    def _update_fps(self):
        self._fps_n += 1
        if self._fps_n >= 10:
            t1 = time.time()
            dt = t1 - self._fps_t0
            if dt > 0:
                self.fps = self._fps_n / dt
            self._fps_t0 = t1
            self._fps_n = 0

    def _update_stats(self, mm_dist, confidence):
        try:
            if mm_dist is not None:
                self.hist.append(float(mm_dist))
                if len(self.hist) > 200:
                    self.hist = self.hist[-200:]

            if confidence:
                if self.avg_confidence == 0.0:
                    self.avg_confidence = confidence
                else:
                    self.avg_confidence = 0.9 * self.avg_confidence + 0.1 * confidence

            avg_mm = float(np.mean(self.hist)) if self.hist else 0.0
            plot_update = (self._plot_tick % 8 == 0)
            self._plot_tick += 1

            self.stats_cb(
                mm_dist if mm_dist is not None else 0.0,
                self.ema_val if self.ema_val is not None else (mm_dist or 0.0),
                avg_mm,
                plot_update,
            )
        except Exception as e:
            self.logger.error(f"Error actualizando estadísticas: {e}")

    # ----------------------------
    # Utilidades
    # ----------------------------
    def _enhance_image(self, bgr):
        if self.config.image_processing.low_light_enhance:
            lab = cv.cvtColor(bgr, cv.COLOR_BGR2LAB)
            l, a, b = cv.split(lab)
            clip = max(0.1, float(self.config.image_processing.clahe_clip_limit))
            tile = self.config.image_processing.clahe_tile_size
            clahe = cv.createCLAHE(clipLimit=clip, tileGridSize=tuple(tile))
            cl = clahe.apply(l)
            limg = cv.merge((cl, a, b))
            bgr = cv.cvtColor(limg, cv.COLOR_LAB2BGR)
        return bgr

    def _letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114)):
        shape = img.shape[:2]  # (h, w)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            img = cv.resize(img, new_unpad, interpolation=cv.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, value=color)

        return img, r, (dw, dh)


__all__ = ["CADIXDetector"]
