import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os

# Ordena cuatro puntos de un cuadrilátero en el orden:
# [top‑left, top‑right, bottom‑right, bottom‑left]
def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    # la suma x+y será mínima en la esquina superior‑izquierda
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # la diferencia x‑y será mínima en la esquina superior‑derecha
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

# Clasificación avanzada de monedas chilenas usando diámetro, circularidad y tono (hue)
def clasificar_por_caracteristicas(diametro_mm: float,
                                   circularity: float,
                                   hue: float) -> str:

    # 1. detectar $50 por forma (no completamente circular) + diámetro
    if 23.5 <= diametro_mm <= 26.5 and 0.84 <= circularity <= 0.93 and hue < 25:
        return "$50"

    # 2. resto mediante diámetro + color aproximado
    if 15.0 <= diametro_mm <= 18.8:
        return "$1 / 5"
    elif 20.0 < diametro_mm <= 21.5:
        return "$10"
    elif 22.5 < diametro_mm <= 24.5 and hue > 25:
        # $100 actual
        return "$100"
    
    if 24.5 < diametro_mm <= 28.5:
        if diametro_mm >= 26.8:
            return '$100 Antigua'
        else:
            return "$500"
    
    return "?"

# Capture video (1280x720)
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)


if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Aplicar desenfoque para reducir ruido
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    
    # Aplicar detección de bordes
    canny = cv.Canny(blurred, 30, 150)

    # Convertir imagen a espacio HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Rango de color para tonos metálicos (grisáceos)
    lower_metall = np.array([10, 30, 100])   # H: cualquiera, S: baja, V: media
    upper_metall = np.array([40, 255, 255])  # H: cualquiera, S: baja, V: alta

    # Crear máscara
    mask = cv.inRange(hsv, lower_metall, upper_metall)

    # Aplicar la máscara a la imagen original
    masked = cv.bitwise_and(frame, frame, mask=mask)


    # Dibujar los círculos si se encontraron, filtrando por radio y circularidad
    output = frame.copy()

    # Desomentar para variar colores de cámara 
    frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    #lower_blue = np.array([90,  50,  50])
    #upper_blue = np.array([130,255,255])
    #creditcard_mask = cv.inRange(frame_hsv, lower_blue, upper_blue)

    creditcard_mask = cv.inRange(frame_hsv, np.array([0, 0, 0]), np.array([180, 255, 80]))
    contours, _ = cv.findContours(creditcard_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    pix_to_mm = None
    DEFAULT_PIX_TO_MM = 0.32  # valor aproximado para 1280x720
    pix_to_mm = DEFAULT_PIX_TO_MM
    for cnt in contours:
        # usar rectángulo rotado para mayor precisión
        rect = cv.minAreaRect(cnt)
        box = cv.boxPoints(rect).astype("float32")
        ordered = order_points(box)
        (tl, tr, br, bl) = ordered

        # long edges of the card
        widthA  = np.linalg.norm(br - bl)
        widthB  = np.linalg.norm(tr - tl)
        width_px = (widthA + widthB) / 2.0

        # short edges
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        height_px = (heightA + heightB) / 2.0

        # evitar división por cero si el contorno es degenerado
        if height_px == 0:
            continue

        aspect_ratio = width_px / height_px

        # filtro por proporción y tamaño
        if 1.3 < aspect_ratio < 1.8 and width_px > 100:
            pix_to_mm = 85.6 / float(width_px)  # ancho real de la tarjeta
            box = box.astype(int)   # np.int0 está deprecado en NumPy ≥1.26
            cv.drawContours(output, [box], 0, (255, 0, 0), 2)

            ancho_cm = round((width_px * pix_to_mm) / 10, 2)
            alto_cm  = round((height_px * pix_to_mm) / 10, 2)
            cv.putText(output,
                       f"{int(width_px)}px x {int(height_px)}px ~ {ancho_cm:.2f}cm x {alto_cm:.2f}cm",
                       (int(tl[0]) - 50, int(tl[1]) - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            break

    # factores de escala 
    if pix_to_mm is not None:
        px_per_mm   = 1.0 / pix_to_mm           # píxeles que ocupa 1 mm
        # diámetros de monedas chilenas más pequeñas y más grandes
        MIN_DIAM_MM = 17.0                      # $1 (≈19-20 mm)
        MAX_DIAM_MM = 27.0                      # $50/$500 (≈26-27 mm)

        # pasamos a radios y añadimos un margen ±10 %
        minRadius = int(0.45 * MIN_DIAM_MM * px_per_mm)   # (0.9·diam)/2
        maxRadius = int(0.55 * MAX_DIAM_MM * px_per_mm)   # (1.1·diam)/2
    else:
        # si la tarjeta no se ve, usa un rango amplio por defecto
        minRadius, maxRadius = 20, 45

    # Detección de círculos con Hough
    circles = cv.HoughCircles(
        blurred, 
        cv.HOUGH_GRADIENT, 
        dp=1.0, 
        minDist=50,
        param1=100, 
        param2=42, 
        minRadius=minRadius, 
        maxRadius=maxRadius
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        filtered_circles = []
        for (x, y, r) in circles:
            # descartar círculos fuera del rango dinámico
            if r < minRadius or r > maxRadius:
                continue
            mask = np.zeros(gray.shape, dtype="uint8")
            cv.circle(mask, (x, y), r, 255, -1)
            masked = cv.bitwise_and(gray, gray, mask=mask)
            contours_roi, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            for c in contours_roi:
                area = cv.contourArea(c)
                perimeter = cv.arcLength(c, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                if circularity < 0.70 or circularity > 0.98:
                    cv.putText(output, f"C={circularity:.2f}", (x + r + 5, y),
                                cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                    continue
                filtered_circles.append((x, y, r))

        conteo_monedas = {"$1 / 5": 0, "$10": 0, "$50": 0,
                          "$100": 0, "$100 Antigua": 0,
                          "$500": 0, "?": 0}
 

        for (x, y, r) in filtered_circles:
            cv.circle(output, (x, y), r, (0, 255, 0), 2)
            cv.circle(output, (x, y), 2, (0, 0, 255), 3)
            if pix_to_mm is not None:
                diameter_mm = round(2 * r * pix_to_mm, 1)

                # media del tono H dentro del ROI de la moneda
                hsv_roi = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                mask = np.zeros(gray.shape, dtype="uint8")
                cv.circle(mask, (x, y), r, 255, -1)
                hue_vals = hsv_roi[mask == 255][:, 0]
                hue_mean = np.mean(hue_vals) if hue_vals.size else 0

                # calcular circularity para este círculo
                mask_circ = np.zeros(gray.shape, dtype="uint8")
                cv.circle(mask_circ, (x, y), r, 255, -1)
                contours_roi, _ = cv.findContours(mask_circ, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                circularity = 0
                for c in contours_roi:
                    area = cv.contourArea(c)
                    perimeter = cv.arcLength(c, True)
                    if perimeter == 0:
                        continue
                    circularity = 4 * np.pi * (area / (perimeter * perimeter))
                    break

                # filtro por agujeros internos (botones)
                mask_roi = np.zeros(gray.shape, dtype="uint8")
                cv.circle(mask_roi, (x, y), int(r * 0.8), 255, -1)

                inner_blur = cv.GaussianBlur(gray, (3, 3), 0)
                inner_edges = cv.Canny(inner_blur, 50, 150)
                inner_edges = cv.bitwise_and(inner_edges, inner_edges, mask=mask_roi)

                inner_circles = cv.HoughCircles(
                    inner_edges,
                    cv.HOUGH_GRADIENT,
                    dp=1.0,
                    minDist=10,
                    param1=80,
                    param2=15,
                    minRadius=int(r * 0.05),
                    maxRadius=int(r * 0.35)
                )

                # Si se detecta al menos un círculo interno, lo descartamos como moneda
                if inner_circles is not None:
                    # cuenta de elemento desconocido
                    conteo_monedas["?"] += 1
                    cv.putText(output, "??", (x - 20, y),
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    continue

                etiqueta = clasificar_por_caracteristicas(diameter_mm, circularity, hue_mean)
                conteo_monedas[etiqueta] += 1
                cv.putText(output, f"{etiqueta} ({diameter_mm}mm) / r={r}px", (x - 40, y - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        if pix_to_mm is not None:
            y_offset = 30
            total_monedas = sum(conteo_monedas.values())
            for denom, count in conteo_monedas.items():
                texto = f"{denom}: {count}"
                cv.putText(output, texto, (10, y_offset), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 25
            cv.putText(output, f"Total: {total_monedas}", (10, y_offset),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Mostrar resultado con OpenCV
        # print('Coins in the image: ', len(filtered_circles))
        # for denom, count in conteo_monedas.items():
        #     print(f"{denom}: {count} monedas")

    # Mostrar resultado con OpenCV (siempre mostrar la ventana, incluso si no hay círculos detectados)
    #cv.imshow('Monedas detectadas', output)


    if pix_to_mm is not None:
        print(f"[DEBUG] pix_to_mm = {pix_to_mm:.3f} mm/px")
        print(f"[DEBUG] Ancho tarjeta detectada: {int(width_px)}px")
    else:
        print("[DEBUG] No se detectó tarjeta. No hay pix_to_mm.")

    cv.imshow('Monedas detectadas', output)
    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()