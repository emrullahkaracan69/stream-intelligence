import logging

import cv2

from src.config import config
from src.detector.camera import CameraSource
from src.detector.yolo_detector import YoloDetector
from src.logging_config import setup_logging
from src.stream.youtube_stream import YouTubeStream

logger = logging.getLogger(__name__)


def draw_detections(frame, detections):
    """Draw bounding boxes and labels on frame for each detection."""
    for d in detections:
        # Bounding box çiz — yeşil renk, 2 piksel kalınlık
        cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), (0, 255, 0), 2)

        # Etiket metni — sınıf adı ve confidence yüzdesi
        label = f"{d.class_name} {d.confidence:.0%}"

        # Etiketi kutunun 10 piksel üstüne yaz
        cv2.putText(
            frame,
            label,
            (d.x1, d.y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
    return frame


def run():
    """Main pipeline: connect to source, detect objects, display results."""
    setup_logging()
    logger.info("Starting stream intelligence pipeline")

    # YoloDetector bir kez yüklenir — model GPU'ya alınır
    # Her frame'de yeniden yüklemek çok maliyetli olurdu
    detector = YoloDetector()

    # config.source_type .env'den geliyor — "camera" veya "youtube"
    # İkisi de aynı read_frame() interface'ini kullandığı için
    # döngü kodu değişmiyor, sadece kaynak değişiyor
    if config.source_type == "youtube":
        source = YouTubeStream(config.youtube_url)
    else:
        source = CameraSource(config.camera_index)

    # with bloğu — kaynak açılır, döngü biter veya hata olursa otomatik kapanır
    with source:
        logger.info("Connected to source, starting detection loop")

        while True:
            ret, frame, fps = source.read_frame()

            # ret False ise frame gelmedi — stream koptu olabilir
            if not ret:
                logger.warning("Failed to read frame, skipping")
                continue

            # Tespitleri al
            detections = detector.detect(frame)

            # Frame üzerine kutuları çiz
            frame = draw_detections(frame, detections)

            # FPS ve tespit sayısını sol üst köşeye yaz
            cv2.putText(
                frame,
                f"FPS: {fps:.1f} | Detections: {len(detections)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

            # Pencerede göster — "Stream Intelligence" pencere başlığı
            cv2.imshow("Stream Intelligence", frame)

            # waitKey(1) — 1ms bekle, klavye girişi kontrol et
            # 0xFF maskesi — farklı sistemlerde klavye kodları için güvenli karşılaştırma
            # q'ya basılırsa döngüden çık
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("Quit signal received, shutting down")
                break

    # Tüm OpenCV pencereleri kapat
    cv2.destroyAllWindows()
    logger.info("Pipeline stopped")


if __name__ == "__main__":
    run()
