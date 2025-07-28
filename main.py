import os
import cv2
import sys
import numpy as np
from collections import Counter

# --- Первый способ: глобальный статистический анализ ---
def analyze_global_brightness(Y, R, B):
    """
    Глобальный статистический анализ яркости изображения.
    Возвращает:
    - средняя и медианная яркость
    - коэффициент вариации, квантили
    - цветовой баланс (теплота)
    - вывод о времени суток и освещенности
    """

    mean_brightness = np.mean(Y)
    median_brightness = np.median(Y)
    std_brightness = np.std(Y)
    coef_variation = std_brightness / (mean_brightness + 1e-5)

    q25 = np.percentile(Y, 25)
    q75 = np.percentile(Y, 75)

    percent_dark = np.sum(Y < 50) / Y.size * 100
    percent_very_dark = np.sum(Y < 30) / Y.size * 100
    percent_bright = np.sum(Y > 180) / Y.size * 100
    percent_very_bright = np.sum(Y > 220) / Y.size * 100

    warm_ratio = np.mean(R) / (np.mean(B) + 1e-5)

    if mean_brightness < 40:
        day_period = "ночь"
    elif mean_brightness < 70:
        day_period = "сумерки"
    elif mean_brightness < 137:
        day_period = "утро или вечер"
    else:
        day_period = "день"

    if mean_brightness < 50 and percent_very_dark > 30:
        light_status = "темно"
    elif mean_brightness > 160:
        light_status = "светло"
    else:
        light_status = "приглушённый свет"

    return {
        "средняя_яркость": round(mean_brightness, 2),
        "медиана_яркости": round(median_brightness, 2),
        "стандартное_отклонение": round(std_brightness, 2),
        "коэффициент_вариации": round(coef_variation, 3),
        "квантиль_25%": round(q25, 2),
        "квантиль_75%": round(q75, 2),
        "процент_очень_тёмных": round(percent_very_dark, 2),
        "процент_тёмных": round(percent_dark, 2),
        "процент_ярких": round(percent_bright, 2),
        "процент_очень_ярких": round(percent_very_bright, 2),
        "теплота_цвета (R/B)": round(warm_ratio, 3),
        "время_суток": day_period,
        "освещенность": light_status
    }

# --- Второй способ: локальный блочный анализ ---
def local_block_analysis(Y, blocks_x=6, blocks_y=4):
    h, w = Y.shape
    block_h = h // blocks_y
    block_w = w // blocks_x
    local_means = []

    for i in range(blocks_y):
        for j in range(blocks_x):
            block = Y[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
            local_mean = np.mean(block)
            local_means.append(local_mean)

    local_means = np.array(local_means)
    avg_local = np.mean(local_means)
    block_std = np.std(local_means)
    block_min = np.min(local_means)
    block_max = np.max(local_means)
    block_range = block_max - block_min

    if avg_local < 40 and block_max < 70:
        local_day_period = "ночь"
    elif avg_local < 70 and block_range > 50:
        local_day_period = "сумерки"
    elif avg_local > 160 and block_range < 40:
        local_day_period = "день"
    elif 70 <= avg_local <= 160 and block_range >= 30:
        local_day_period = "утро или вечер"
    else:
        local_day_period = "неопределённо"

    if block_range < 20:
            illumination_comment = "Освещение равномерное"
    elif block_range > 50 and block_min < 40:
        illumination_comment = "Заметны тени или яркие фонари"
    else:
        illumination_comment = "Частично освещено (контрастная сцена)"

    return {
        "лок_яркость_мин": round(block_min, 2),
        "лок_яркость_макс": round(block_max, 2),
        "лок_яркость_std": round(block_std, 2),
        "лок_яркость_разброс": round(block_range, 2),
        "время_суток_по_локальному_анализу": local_day_period,
        "локальный_анализ": illumination_comment
    }

# --- Третий способ: анализ по цветовым моделям ---    
def brightness_from_color_models(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    mean_v = np.mean(v_channel)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    mean_l = np.mean(l_channel)

    def classify(value):
        if value < 50:
            return "очень темно"
        elif value < 100:
            return "темно"
        elif value < 170:
            return "приглушённый свет"
        elif value < 220:
            return "светло"
        else:
            return "очень светло"

    brightness_hsv = classify(mean_v)
    brightness_lab = classify(mean_l)

    if brightness_hsv == brightness_lab:
        combined = brightness_hsv
    else:
        scores = {
            "очень темно": 0,
            "темно": 1,
            "приглушённый свет": 2,
            "светло": 3,
            "очень светло": 4
        }
        avg_score = (scores[brightness_hsv] + scores[brightness_lab]) / 2
        combined = [k for k, v in scores.items() if round(avg_score) == v][0]

    return {
        "освещенность_HSV": brightness_hsv,
        "освещенность_LAB": brightness_lab,
        "средняя_V": round(mean_v, 2),
        "средняя_L": round(mean_l, 2),
        "освещенность_по_цветовым_моделям": combined
    }

def most_common_vote(votes):
    counter = Counter(votes)
    most_common = counter.most_common(1)
    return most_common[0][0] if most_common else "неопределённо"

def enhanced_brightness_analysis(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Не удалось загрузить изображение.")

    YUV = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    Y, _, _ = cv2.split(YUV)
    B, _, R = cv2.split(image)

    # --- Первый способ: глобальный статистический анализ ---
    global_stats = analyze_global_brightness(Y, R, B)

    # --- Второй способ: локальный блочный анализ ---
    local_stats = local_block_analysis(Y)

    # --- Третий способ: анализ по цветовым моделям ---
    color_model_stats = brightness_from_color_models(image)
    
    # Итоговое время суток
    time_votes = [
        global_stats["время_суток"],
        local_stats["время_суток_по_локальному_анализу"]
    ]
    final_time = most_common_vote(time_votes)

    # Итоговая освещённость
    light_votes = [
        global_stats["освещенность"],
        color_model_stats["освещенность_по_цветовым_моделям"]
    ]
    final_light = most_common_vote(light_votes)

    # Дополнительная корректировка, если сцена контрастная
    has_shadows = "тени" in local_stats["локальный_анализ"].lower()
    high_contrast = global_stats["коэффициент_вариации"] > 0.6
    if final_light == "приглушённый свет" and (has_shadows or high_contrast):
        final_light = "контрастная сцена"

    return {
        **global_stats,
        **local_stats,
        **color_model_stats,
        "итоговое_время_суток": final_time,
        "итоговая_освещенность": final_light
    }

def analyze_directory(folder_path):
    supported_ext = ('.jpg', '.jpeg', '.png')
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(supported_ext)]

    global_counter = Counter()
    local_counter = Counter()
    summary_counter = Counter()
    final_light_counter = Counter()

    for filename in files:
        path = os.path.join(folder_path, filename)
        try:
            result = enhanced_brightness_analysis(path)
            global_counter[result["время_суток"]] += 1
            local_counter[result["время_суток_по_локальному_анализу"]] += 1
            summary_counter[result["итоговое_время_суток"]] += 1
            final_light_counter[result["итоговая_освещенность"]] += 1

            print(f"[OK] {filename}: {result['время_суток']} / {result['время_суток_по_локальному_анализу']} / {result['итоговая_освещенность']}")
        except Exception as e:
            print(f"[ERROR] {filename}: {e}")

    print("\n[] Итоги по времени суток (глобальный анализ):")
    for k, v in global_counter.items():
        print(f"\t{k}: {v} шт.")

    print("\n[] Итоги по времени суток (локальный блочный анализ):")
    for k, v in local_counter.items():
        print(f"\t{k}: {v} шт.")

    print("\n[] Итоговая классификация времени суток:")
    for k, v in summary_counter.items():
        print(f"\t{k}: {v} шт.")

    print("\n[] Итоговая классификация освещенности:")
    for k, v in final_light_counter.items():
        print(f"\t{k}: {v} шт.")


if __name__ == "__main__":
    """
    image_path = "img1.png"
    result = enhanced_brightness_analysis(image_path)
    for key, value in result.items():
        print(f"{key}: {value}")
    """
    if len(sys.argv) < 2:
        print("Использование: python3 main.py <путь_к_папке_с_изображениями>")
        sys.exit(1)

    folder = sys.argv[1]

    if not os.path.isdir(folder):
        print(f"Ошибка: папка '{folder}' не существует или недоступна.")
        sys.exit(1)
        
    analyze_directory(folder)
