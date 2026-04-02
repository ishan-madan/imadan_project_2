# Sleep Optimization AI – Project Proposal

## 1. Motivation

Sleep quality is heavily influenced by environmental factors such as temperature, humidity, and movement, but these effects are often difficult to quantify. This project was inspired by a personal experience while tenting for a Duke MBB game, where I consistently felt poorly rested but had no way to measure or understand why. 

The goal is to build a system that can **objectively evaluate sleep conditions and provide actionable feedback**.

---

## 2. Objective

This project aims to develop a **Raspberry Pi-based edge AI system** that:
- Collects environmental and motion data using sensors
- Uses a machine learning model to estimate sleep quality
- Uses a large language model (LLM) to interpret results and generate advice
- Provides real-time feedback through an LCD and LEDs

---

## 3. Proposed Approach

### Data Collection
- Temperature & humidity (DHT11)
- Motion (MPU6050)
- Visual movement (camera)

### Modeling
- Train a **logistic regression model** to predict sleep quality probability

### LLM Integration
- Combine current readings and recent trends into a prompt
- Generate:
  - Sleep score (0–100)
  - Short tip (LCD)
  - Detailed advice

### Output
- LCD displays score and short recommendation
- LEDs indicate sleep quality (green = good, red = poor)
- Data logged to CSV for analysis

---

## 4. Expected Challenges

- Sensor noise and calibration
- Interpreting camera-based motion
- Designing reliable LLM prompts
- Maintaining real-time performance

---

## 5. Definition of Success

The project will be successful if it:
- Accurately collects and processes sensor data
- Produces reasonable sleep quality estimates
- Generates useful, actionable feedback
- Successfully runs end-to-end on the Raspberry Pi

---

## 6. Conclusion

This project combines sensors, machine learning, and LLMs to create a system that transforms environmental data into meaningful sleep insights. It aims to better understand and improve sleep quality in real-world conditions.