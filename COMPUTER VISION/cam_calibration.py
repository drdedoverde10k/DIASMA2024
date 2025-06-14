import cv2 as cv
import numpy as np

# Constante física del ancho real de la tarjeta
REAL_WIDTH_MM = 85.6

# Inicializar cámara
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
if not cap.isOpened():
    print("❌ No se pudo abrir la cámara")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ No se pudo capturar el frame")
        continue

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv.threshold(blurred, 80, 255, cv.THRESH_BINARY_INV)

    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    pix_to_mm = None
    output = frame.copy()

    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < 5000:
            continue

        peri = cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            x, y, w, h = cv.boundingRect(approx)
            aspect_ratio = w / h

            if 1.4 < aspect_ratio < 1.7 and 180 < w < 800 and 100 < h < 500:
                roi = gray[y:y+h, x:x+w]
                mean_intensity = np.mean(roi)
                if mean_intensity > 180:
                    continue

                # Calibración
                pix_to_mm = REAL_WIDTH_MM / w
                ancho_cm = round(w * pix_to_mm / 10, 2)
                alto_cm = round(h * pix_to_mm / 10, 2)

                cv.drawContours(output, [approx], -1, (255, 0, 0), 2)
                cv.putText(output,
                           f"{w}px x {h}px ≈ {ancho_cm}cm x {alto_cm}cm",
                           (x, y - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                print(f"📏 Escala estimada: pix_to_mm = {pix_to_mm:.4f}")
                print(f"Rectángulo detectado: w = {w}px, h = {h}px")
                break

    cv.imshow("Calibración de cámara (tarjeta)", output)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()