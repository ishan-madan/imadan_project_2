# Sleep Optimization AI (Final Project)

## Overview
This project is a real-time **AI-powered sleep optimization system** deployed on a Raspberry Pi. It integrates multiple environmental sensors, a camera, a machine learning model, and a large language model (LLM) to analyze sleep conditions and provide actionable feedback.

The system continuously monitors:
- Temperature
- Humidity
- Physical motion (accelerometer)
- Visual motion (camera-based detection)

It then:
1. Uses a **logistic regression model** to estimate sleep probability.
2. Uses an **LLM (GPT-5-mini via Duke LiteLLM API)** to interpret sensor data and trends.
3. Outputs:
   - A **sleep quality score (0–100)**
   - **Short actionable tips** (LCD display)
   - **Detailed long-form advice** (trend-aware)
   - **Physical feedback via LEDs**

This project satisfies all requirements of the AIPI 590 final project:
- Multi-sensor input (including camera)
- Edge ML model deployed on Raspberry Pi
- LLM integration via API
- Physical interface (LCD + button)
- Physical environment manipulation (LEDs)

---

## Features

### Sensor Integration
- **DHT11** → Temperature & Humidity
- **MPU6050** → Motion detection via acceleration magnitude
- **Pi Camera (picamzero)** → Visual motion detection using frame differencing

### AI + Intelligence
- **Logistic Regression Model**
  - Predicts baseline sleep probability
  - Uses temperature, humidity, motion
- **LLM Agent**
  - Performs reasoning on:
    - Current sensor data
    - Historical trends
  - Outputs structured JSON decisions

### Trend Awareness
- Stores last **10 readings**
- Computes:
  - Averages
  - Threshold violations
  - Motion spikes
- LLM uses this to generate **context-aware advice**

### Outputs
- **LCD Display (16x2)**
  - Sleep score
  - Short tip (<16 chars)
- **LED Feedback**
  - Green = Good sleep quality
  - Red = Poor sleep quality
- **CSV Logging**
  - Persistent dataset for analysis

---

## Hardware Components

- Raspberry Pi (with GPIO + I2C)
- DHT11 Temperature & Humidity Sensor (GPIO 16)
- MPU6050 Accelerometer (I2C)
- Pi Camera Module (via picamzero)
- 16x2 LCD with I2C (address 39)
- Green LED (GPIO 21)
- Red LED (GPIO 20)
- Push Button (GPIO 24)
- Breadboard + resistors + jumper wires

---

## Wiring Description

### Core Connections
- **DHT11**
  - Data → GPIO 16

- **MPU6050 (I2C)**
  - SDA → SDA
  - SCL → SCL

- **LCD (I2C)**
  - SDA → SDA
  - SCL → SCL

- **Camera**
  - Connected via CSI port (handled by picamzero)

- **LEDs**
  - Green LED → GPIO 21 → resistor → GND
  - Red LED → GPIO 20 → resistor → GND

- **Button**
  - GPIO 24 → Button → GND (pull-up logic)

---

## System Architecture

### Data Flow
1. Sensors collect real-time environmental data
2. Data is passed to:
   - ML model → sleep probability
   - LLM → reasoning + decision-making
3. Outputs are generated:
   - LCD display
   - LED signals
   - CSV logging

---

## Code Structure

The entire system is implemented in a single Python script and structured as follows:

### 1. Configuration
Defines:
- File paths (model, scaler, logs)
- GPIO pins
- LLM API settings
- History tracking parameters

---

### 2. Initialization Functions
- `initialize_dht()`
- `initialize_mpu()`
- `initialize_lcd()`
- `initialize_leds()`
- `initialize_camera()`

---

### 3. Sensor Reading
- `read_dht()` → temperature & humidity
- `read_motion()` → acceleration magnitude
- `get_camera_motion()` → image difference-based motion

Camera motion pipeline:
- Capture image
- Compare with previous frame
- Convert to grayscale
- Blur + threshold
- Compute pixel difference

---

### 4. Machine Learning Model
- `load_model()` → loads trained model + scaler
- `predict_sleep()` → outputs sleep probability

---

### 5. Trend Analysis
- Maintains rolling history (last 10 readings)
- `add_to_history()` → stores readings
- `summarize_history()` → computes:
  - averages
  - high/low counts
  - motion spikes

---

### 6. LLM Integration

#### Prompt Engineering
- Combines:
  - Current sensor data
  - Historical trends
  - Explicit scoring rules

#### `call_llm()`
- Sends request to Duke LiteLLM endpoint
- Includes:
  - retries
  - timeout handling
  - fallback response

#### Output Format (JSON)
```json
{
"action": "LED_ON | LED_OFF | NOOP",
"brightness": 0.0-1.0,
"sleep_score": 0-100,
"short_tip": "...",
"long_tip": "..."
}
```


#### Validation
- Ensures:
  - Correct JSON format
  - Safe bounds
  - Allowed actions only

---

### 7. Logging
- `init_log()` → creates CSV
- `log_data()` → appends readings

Columns:
- timestamp
- temperature
- humidity
- motion
- camera motion
- sleep score

---

### 8. Main Loop
1. Wait for button press
2. Continuously:
   - Read sensors
   - Compute motion
   - Predict sleep probability
   - Call LLM
   - Validate response
   - Update LEDs
   - Update LCD
   - Store history
   - Log data

---

## Installation

### 1. Clone Repository

### 2. Install Dependencies
```bash
pip install adafruit-circuitpython-dht
pip install adafruit-circuitpython-mpu6050
pip install gpiozero
pip install joblib
pip install requests
pip install opencv-python
```


### 3. Environment Variable
Set your API key:
```bash
export LITELLM_TOKEN=your_token_here
```


### 4. Required Files
- `sleep_quality_model.pkl`
- `sleep_quality_scaler.pkl`

---

## Usage

Run:
```bash
python3 sleep_predictor.py
```


Steps:
1. LCD displays: "Press button"
2. Press button to start
3. System begins monitoring every 5 seconds
4. Outputs:
   - LCD → score + short tip
   - LEDs → sleep quality
   - CSV → logged data

---

## Example Output

LCD:
```bash
Score:72 
v Temp, v Humid
```


Terminal:
```json
{
"action": "LED_OFF",
"sleep_score": 72,
"short_tip": "vTemp vHum",
"long_tip": "Temperature has been slightly high..."
}
```


---

## Design Decisions

- **LLM used for reasoning instead of thresholds**
  - Enables dynamic interpretation
- **Camera motion added**
  - Captures behavior not detectable via accelerometer
- **Trend-based feedback**
  - Provides more meaningful advice than single readings
- **Fail-safe LLM handling**
  - Prevents crashes on API failure

---

## Challenges

- Camera motion calibration
- Handling LLM failures robustly
- Designing effective prompt constraints
- Keeping LCD output within 16 characters
- Synchronizing multiple sensors

---

## Future Improvements

- Replace heuristic scoring with learned model that adapts to the user's behaviors
- Add sound/microphone input
- Mobile dashboard for visualization
- Adaptive learning based on user feedback

---

## Files Included

- `sleep_predictor.py` → full system script
- `sleep_quality_model.pkl`
- `sleep_quality_scaler.pkl`
- `sleep_final_log.csv`
- Circuit diagram image
- README.md

---

## Citation & Acknowledgements
- AI assistants were used to compile this information into this README file. Data was given by me, but formatting was done by ChatGPT.

---

## Notes
- Ensure correct wiring before running
- Use resistors for LEDs
- Ensure stable camera positioning