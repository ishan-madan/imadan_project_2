# ==============================
# IMPORTS
# ==============================

import os
import time
import math
import json
import re
import csv
from datetime import datetime

import requests
import joblib
import board
import adafruit_dht
import busio
import adafruit_mpu6050

from gpiozero import LED, Button
from lcd_i2c import LCD_I2C
from picamzero import Camera


# ==============================
# CONFIG
# ==============================

MODEL_PATH = "sleep_quality_model.pkl"
SCALER_PATH = "sleep_quality_scaler.pkl"

LOG_FILE = "sleep_final_log.csv"

DHT_PIN = board.D16
LCD_ADDRESS = 39

GREEN_LED_PIN = 21
RED_LED_PIN = 20
START_BUTTON_PIN = 24

PREDICTION_INTERVAL = 5.0

DUKE_CHAT_URL = "https://litellm.oit.duke.edu/chat/completions"
MODEL = "gpt-5-mini"
TIMEOUT = 30

DEFAULT_LLM_RESPONSE = {
    "action": "NOOP",
    "brightness": 0.0,
    "sleep_score": 50,
    "advice": "No advice (LLM unreachable)"
}

ALLOWED_ACTIONS = {"LED_ON", "LED_OFF", "NOOP"}

HISTORY_LENGTH = 10  # number of previous readings to consider
history = []  # store dicts with temp, humidity, motion, cam_motion, score, long_tip


# ==============================
# I/O INITIALIZATION
# ==============================

def initialize_dht():
    return adafruit_dht.DHT11(DHT_PIN)


def initialize_mpu():
    i2c = busio.I2C(board.SCL, board.SDA)
    return adafruit_mpu6050.MPU6050(i2c)


def initialize_lcd():
    lcd = LCD_I2C(LCD_ADDRESS, 16, 2)
    lcd.backlight.on()
    lcd.clear()
    return lcd


def initialize_leds():
    return LED(GREEN_LED_PIN), LED(RED_LED_PIN)


def initialize_camera():
    cam = Camera()
    return cam


# ==============================
# SENSOR READING
# ==============================

def read_dht(sensor):
    try:
        return sensor.temperature, sensor.humidity
    except RuntimeError:
        return 22.0, 50.0  # fallback


def read_motion(mpu):
    ax, ay, az = mpu.acceleration
    return math.sqrt(ax**2 + ay**2 + az**2)


def get_camera_motion(cam, last_photo_path="motion_prev.jpg"):
    import cv2
    temp_path = "motion_curr.jpg"

    time.sleep(0.2)
    cam.take_photo(temp_path)

    if not os.path.exists(last_photo_path):
        os.rename(temp_path, last_photo_path)
        return 0.0

    frame1 = cv2.imread(last_photo_path)
    frame2 = cv2.imread(temp_path)

    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

    os.replace(temp_path, last_photo_path)

    return thresh.sum() / 1000000  # normalize


# ==============================
# BASIC MODEL
# ==============================

def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def predict_sleep(model, scaler, temp, humidity, motion):
    features = [[temp, humidity, motion]]
    scaled = scaler.transform(features)
    return model.predict_proba(scaled)[0][1]


# ==============================
# TREND ANALYSIS
# ==============================

def add_to_history(temp, humidity, motion, cam_motion, score, long_tip):
    history.append({
        "temp": temp,
        "humidity": humidity,
        "motion": motion,
        "cam_motion": cam_motion,
        "score": score,
        "long_tip": long_tip
    })
    if len(history) > HISTORY_LENGTH:
        history.pop(0)


def summarize_history():
    if not history:
        return {}

    temps = [h["temp"] for h in history]
    humid = [h["humidity"] for h in history]
    motion = [h["motion"] for h in history]
    cam_motion = [h["cam_motion"] for h in history]
    scores = [h["score"] for h in history]

    summary = {
        "avg_temp": sum(temps)/len(temps),
        "avg_humidity": sum(humid)/len(humid),
        "avg_motion": sum(motion)/len(motion),
        "avg_cam_motion": sum(cam_motion)/len(cam_motion),
        "avg_score": sum(scores)/len(scores),
        "motion_high_count": sum(1 for m in motion if m > 15),
        "cam_motion_high_count": sum(1 for cm in cam_motion if cm > 50),
        "temp_high_count": sum(1 for t in temps if t > 22),
        "temp_low_count": sum(1 for t in temps if t < 18),
        "humidity_high_count": sum(1 for h in humid if h > 60),
        "humidity_low_count": sum(1 for h in humid if h < 40),
    }
    return summary


# ==============================
# LLM INTEGRATION
# ==============================

def get_token():
    token = os.getenv("LITELLM_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Missing LITELLM_TOKEN")
    return token


def call_llm(prompt, retries=3, timeout=30):
    """
    Calls the Duke LLM with the given prompt.
    Retries on failure and returns a safe default if all attempts fail.
    """
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "x-litellm-api-key": get_token()
    }

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}]
    }

    for attempt in range(1, retries + 1):
        try:
            r = requests.post(DUKE_CHAT_URL, headers=headers, json=payload, timeout=timeout)
            r.raise_for_status()  # raise an error for HTTP errors

            # Extract LLM response
            response_text = r.json()["choices"][0]["message"]["content"]
            print(f"LLM response received (attempt {attempt})")
            return response_text

        except requests.exceptions.Timeout:
            print(f"LLM timeout on attempt {attempt}/{retries} (waited {timeout}s)... retrying")
            time.sleep(2)
        except requests.exceptions.RequestException as e:
            print(f"LLM request error on attempt {attempt}/{retries}: {e}")
            time.sleep(2)
        except (KeyError, json.JSONDecodeError) as e:
            print(f"LLM returned invalid JSON on attempt {attempt}/{retries}: {e}")
            time.sleep(2)

    # If all retries fail, return default safe JSON string
    print("LLM unreachable or error persists, returning default safe response.")
    return json.dumps(DEFAULT_LLM_RESPONSE)


def build_prompt(temp, humidity, motion, cam_motion, prob):
    summary = summarize_history()

    trend_text = f"""
Recent trends (last {HISTORY_LENGTH} readings):
Avg Temp: {summary.get('avg_temp', temp):.1f}°C
Avg Humidity: {summary.get('avg_humidity', humidity):.1f}%
Avg Motion: {summary.get('avg_motion', motion):.2f}
Avg Cam Motion: {summary.get('avg_cam_motion', cam_motion):.2f}
Temp High: {summary.get('temp_high_count',0)}, Temp Low: {summary.get('temp_low_count',0)}
Humidity High: {summary.get('humidity_high_count',0)}, Humidity Low: {summary.get('humidity_low_count',0)}
Motion spikes: {summary.get('motion_high_count',0)}, Cam motion spikes: {summary.get('cam_motion_high_count',0)}
"""

    return f"""
You are a sleep optimization AI.

Sensor data:
Temperature: {temp}
Humidity: {humidity}
Gyro motion: {motion}
Camera motion: {cam_motion}
Sleep probability: {prob}

{trend_text}

Return ONLY JSON:
{{
  "action": "LED_ON | LED_OFF | NOOP",
  "brightness": 0.0 to 1.0,
  "sleep_score": 0 to 100,
  "short_tip": "16 chars max tip for LCD",
  "long_tip": "detailed advice to store for trend analysis"
}}

Rules:
- LED_ON means the RED led should be turned on and the green off (due to poor sleep quality, which is classified as under 60/100 quality)
- LED_OFF means the GREEN led should be turned ON and the red off (due to good sleep quality, which is classified as over 60/100 quality)
- Brightness is always set to 1.0
- Sleep score should be a score between 0-100 with deductions from the total of 100 made accordingly based on the instructions below
- The ideal average temperature is 18-23 C. Every 1 degree deviation from this range should result in a 5 point drop in sleep quality.
- The ideal humidity is 40-60%. Every 1% deviation from this range should result in a 1 point drop in sleep quality.
- The base gyro acceleration is 10.15 (this is due to gravity). The value that is being inputted is calculated with sqrt(ax^2 + ay^2 + az^2). Every increase in gyro motion by 2 should result in a 5 point deduction in sleep quality.
- Camera motion is a value calculated by taking the pixel difference between 2 photos. This value is then divided by 1,000,000. This makes the range a value from 0-1250. Follow the rules below for acceptable motion ranges with recommended point deductions. AS a general rule, more motion is bad and should result in lower point values/higher deductions.
- A camera motion of 0-75 entails light motion. This could be something such a small movement during sleep (foot moves marginally). This should NOT result in a point deduction.
- A camera motion of 75-150 entails some motion. This could be something such as light tossing. This should result in a point deduction of 1-4 points (depending on severity of motion value)
- A camera motion of 150-350 entails heavy motion. This could be something like heavy tossing and turning. This should result in a point deduction of 4-10 points (depending on severity of motion value)
- A camera motion of 350+ entails very heavy motion. This could be something like gettign up out of bed or a pet coming onto the bed. This should result in a point deduction of at least 10 points. Scale this value up based on how high the camera motion value is.
- Sleep probability is calculated using a logistic regression model. This model only utilizes the numeric data values of Temp, Humidity, and Gyro. Use this as a baseline if and only if your predictions are inconclusive based on your own analysis of the data when using this information alongside the camera motion info.
- Short tip should be a short tip outputted to the user. It should be <16 chars max tip for LCD. The tip should tell the user to either increase or decrease the following things: Temp, Humidity, Motion. The tip can tell the user to decrease one, two, or all three of these things. In order to keep things shorter, please use "^" instead of the word increase and use "v" instead of the word decrease.
- Long tip should be a long comprehensive tip of the current and past trend of data. This tip should comprehensively explain what happened in the surret sensor read, as well as what has been going on during the saved past sensor reads. It should then provide options for how the user can aim to improve the sleep quality, such as suggesting tips based on what the data can tell from the sensor readings. 
"""


def validate(raw):
    DEFAULT = {
        "action": "NOOP",
        "brightness": 0.0,
        "sleep_score": 50,
        "short_tip": "No tip",
        "long_tip": "No advice"
    }

    try:
        json_text = re.search(r"\{.*\}", raw, re.DOTALL).group()
        data = json.loads(json_text)
    except:
        return DEFAULT

    action = str(data.get("action", "NOOP")).upper()
    if action not in ALLOWED_ACTIONS:
        action = "NOOP"

    brightness = float(data.get("brightness", 0.0))
    brightness = max(0.0, min(1.0, brightness))

    score = int(data.get("sleep_score", 50))
    score = max(0, min(100, score))

    short_tip = str(data.get("short_tip", "No tip"))[:16]
    long_tip = str(data.get("long_tip", "No advice"))

    return {
        "action": action,
        "brightness": brightness,
        "sleep_score": score,
        "short_tip": short_tip,
        "long_tip": long_tip
    }


# ==============================
# LOGGING
# ==============================

def init_log():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time","temp","humidity","motion","cam_motion","score"])


def log_data(temp, humidity, motion, cam_motion, score):
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            temp,
            humidity,
            motion,
            cam_motion,
            score
        ])


# ==============================
# MAIN
# ==============================

def main():
    print("Starting Sleep AI Final...")

    model, scaler = load_model()
    dht = initialize_dht()
    mpu = initialize_mpu()
    lcd = initialize_lcd()
    green_led, red_led = initialize_leds()
    cam = initialize_camera()

    init_log()
    button = Button(START_BUTTON_PIN)
    lcd.write_text("Press button")

    while not button.is_pressed:
        time.sleep(0.1)

    lcd.clear()
    last_photo_path = "motion_prev.jpg"

    try:
        while True:
            temp, humidity = read_dht(dht)
            motion = read_motion(mpu)
            cam_motion = get_camera_motion(cam, last_photo_path)

            prob = predict_sleep(model, scaler, temp, humidity, motion)

            prompt = build_prompt(temp, humidity, motion, cam_motion, prob)
            raw = call_llm(prompt)
            decision = validate(raw)

            # LED control
            if decision["action"] == "LED_ON":
                red_led.on()
                green_led.off()
            elif decision["action"] == "LED_OFF":
                green_led.on()
                red_led.off()

            # Add current reading to history
            add_to_history(temp, humidity, motion, cam_motion, decision["sleep_score"], decision["long_tip"])

            # LCD display
            lcd.clear()
            lcd.cursor.setPos(0, 0)
            lcd.write_text(f"Score:{decision['sleep_score']} {decision['short_tip']}"[:16])
            lcd.cursor.setPos(1, 0)
            lcd.write_text(decision["long_tip"][:16])

            log_data(temp, humidity, motion, cam_motion, decision["sleep_score"])
            print(decision)
            time.sleep(PREDICTION_INTERVAL)

    except KeyboardInterrupt:
        print("Shutting down...")
        green_led.off()
        red_led.off()
        lcd.clear()


if __name__ == "__main__":
    main()