/*
  Requires: Encoder library by Paul Stoffregen (install via Library Manager)

  arduino_mdd10_motor_encoder_analog_pedal_controller.ino

  Arduino side for:
    Raspberry Pi -> USB serial -> Arduino -> MDD10
    Arduino reads:
      - encoder A/B
      - analog pedal voltage

  Your pedal values:
    released/off  ≈ 0.8 V
    full pressed  ≈ 4.21 V

  Arduino ADC:
    0..5 V -> analogRead 0..1023

  Approx ADC values:
    0.8 V  -> 0.8 / 5.0 * 1023 ≈ 164
    4.21 V -> 4.21 / 5.0 * 1023 ≈ 861

  Responsibilities:
    - Receive motor commands from Raspberry Pi:
        M <pwm> <dir>

      Example:
        M 120 1

    - Control MDD10:
        D9 -> MDD10 PWM
        D8 -> MDD10 DIR

    - Read encoder:
        D2 -> Encoder A
        D3 -> Encoder B

    - Read analog pedal:
        A0 -> pedal analog signal

    - Optional pedal status output:
        D7 -> pedal pressed output, open-drain style, Raspberry Pi safe

    - Send telemetry:
        T <rpm> <speed_mps> <pulse_count> <direction> <pedal_percent> <pedal_pressed> <pedal_adc> <pedal_voltage>

      Example:
        T 398.50 0.996 1327 1 52.4 1 529 2.59

  IMPORTANT SAFETY:
    - Arduino Uno/Nano analog input can read 0..5 V.
    - Raspberry Pi GPIO CANNOT read 5 V.
    - If you use PEDAL_OUT to Raspberry Pi, this code uses open-drain style:
        pedal pressed     -> D7 pulls LOW
        pedal not pressed -> D7 floats
      Raspberry Pi should use pull-up to 3.3 V.
*/


#include <Encoder.h>


// ============================================================
// Pin config
// ============================================================
const int PIN_MOTOR_PWM = 9;
const int PIN_MOTOR_DIR = 8;

const int PIN_ENC_A = 2;
const int PIN_ENC_B = 3;

const int PIN_PEDAL_ANALOG = A0;

// Optional pedal output to Raspberry Pi.
// Open-drain style output. Safe if Pi uses pull-up to 3.3 V.
const int PIN_PEDAL_OUT = 7;
const bool USE_PEDAL_OUT = true;


// ============================================================
// Motor config
// ============================================================
const int MAX_PWM_LIMIT = 255;
const unsigned long COMMAND_TIMEOUT_MS = 2000;  // stop motor if Pi command disappears (2s for 4-7 FPS loop)


// ============================================================
// Encoder config
// ============================================================
const float ENCODER_PPR = 44.0;                 // measured: 120 FALLING-A+B==LOW per wheel-rev × 4 edges ÷ gear ratio 11 ≈ 44
const float TOTAL_GEAR_RATIO = 11.0;            // motor shaft → wheel: (gearbox 19:1) × (chain 38T/22T) = 19×(22/38)^-1 = 11
const float WHEEL_CIRCUMFERENCE_M = 1.596;      // 20-inch wheel: pi * (20 * 0.0254)
const unsigned long TELEMETRY_INTERVAL_MS = 100;


// ============================================================
// Analog pedal calibration
// ============================================================
// Recalibrated from live telemetry (2026-06-18):
// Resting / not pressed: ADC ~198–208 (0.97–1.02 V)
// Full physical press:   ADC ~665     (3.26 V)
//
// PEDAL_RELEASED_ADC set just below observed minimum (198) for 0% reference.
// PEDAL_FULL_ADC set to observed physical maximum (665) for 100% reference.
const int PEDAL_RELEASED_ADC = 180;
const int PEDAL_FULL_ADC = 665;

// Below this percent, treat pedal as not pressed.
// Resting noise (ADC 198–208) maps to ~3.7–5.8% — 8% deadzone covers it.
const float PEDAL_DEADZONE_PERCENT = 8.0;

// Above this percent, pedal safety override stops the motor.
// 80% = ADC ~568 — well into deliberate press territory.
const float PEDAL_OVERRIDE_PERCENT = 80.0;

// Smoothing factor for pedal ADC.
// Higher = smoother but slower.
// 0.20 is a decent start.
// Very heavy smoothing: motor vibration at high PWM induces ADC noise spikes
// that briefly exceed the override threshold and trigger PEDAL STOP.
// At ~10 kHz loop rate, alpha=0.02 gives ~5 ms time constant — enough to
// reject brief motor-noise spikes while still tracking real pedal presses.
const float PEDAL_SMOOTH_ALPHA = 0.02;


// ============================================================
// State variables
// ============================================================
String inputLine = "";

// Encoder library handles both A/B interrupts and validates quadrature transitions.
// EMI that hits only one channel, or creates invalid state sequences, is rejected.
Encoder enc(PIN_ENC_A, PIN_ENC_B);

int currentPwm = 0;
int currentDir = 1;

unsigned long lastCommandTime = 0;
unsigned long lastTelemetryTime = 0;

long lastPulseCountForSpeed = 0;

float pedalAdcFiltered = PEDAL_RELEASED_ADC;
float pedalPercent = 0.0;
bool pedalPressed = false;
bool lastPedalPressed = false;


// ============================================================
// Pedal reading
// ============================================================
void updatePedal() {
  int raw = analogRead(PIN_PEDAL_ANALOG);

  pedalAdcFiltered =
      (PEDAL_SMOOTH_ALPHA * raw) +
      ((1.0 - PEDAL_SMOOTH_ALPHA) * pedalAdcFiltered);

  float pct =
      (pedalAdcFiltered - PEDAL_RELEASED_ADC) *
      100.0 /
      (PEDAL_FULL_ADC - PEDAL_RELEASED_ADC);

  pct = constrain(pct, 0.0, 100.0);

  if (pct < PEDAL_DEADZONE_PERCENT) {
    pct = 0.0;
  }

  pedalPercent = pct;
  pedalPressed = pedalPercent >= PEDAL_OVERRIDE_PERCENT;

  // Hardware safety: analog pedal override stops motor immediately.
  if (pedalPressed && currentPwm > 0) {
    stopMotor();
    Serial.println("PEDAL STOP");
  }

  updatePedalOutput();

  if (pedalPressed != lastPedalPressed) {
    Serial.print("P ");
    Serial.print(pedalPressed ? 1 : 0);
    Serial.print(" ");
    Serial.print(pedalPercent, 1);
    Serial.print(" ");
    Serial.println((int)pedalAdcFiltered);
    lastPedalPressed = pedalPressed;
  }
}


// Open-drain style output:
// - pedal pressed: D7 OUTPUT LOW
// - pedal not pressed: D7 INPUT floating
//
// Connect Raspberry Pi GPIO with internal pull-up to 3.3 V.
// This avoids Arduino sending 5 V HIGH into Pi.
void updatePedalOutput() {
  if (!USE_PEDAL_OUT) {
    return;
  }

  if (pedalPressed) {
    pinMode(PIN_PEDAL_OUT, OUTPUT);
    digitalWrite(PIN_PEDAL_OUT, LOW);
  } else {
    pinMode(PIN_PEDAL_OUT, INPUT);
  }
}


// ============================================================
// Motor control
// ============================================================
void setMotor(int pwm, int dir) {
  pwm = constrain(pwm, 0, MAX_PWM_LIMIT);
  dir = (dir != 0) ? 1 : 0;

  // Pedal safety overrides Pi motor command.
  if (pedalPressed && pwm > 0) {
    pwm = 0;
  }

  currentPwm = pwm;
  currentDir = dir;

  digitalWrite(PIN_MOTOR_DIR, currentDir == 1 ? HIGH : LOW);
  analogWrite(PIN_MOTOR_PWM, currentPwm);
}


void stopMotor() {
  currentPwm = 0;
  analogWrite(PIN_MOTOR_PWM, 0);
}


// ============================================================
// Serial command parsing
// ============================================================
// Commands:
//   M 120 1       motor PWM 120, direction 1
//   M 0 1         motor stop
//   STOP          stop motor
//   S             stop motor
//   PING          connection check
//   E?            immediate telemetry
//   P?            pedal status only
//
// Responses:
//   OK M 120 1
//   OK M 0 1 PEDAL_BLOCKED
//   OK STOP
//   OK PONG
//   T ...
//   P ...
// ============================================================
void handleCommand(String line) {
  line.trim();

  if (line.length() == 0) {
    return;
  }

  line.toUpperCase();

  if (line == "STOP" || line == "S") {
    stopMotor();
    lastCommandTime = millis();
    Serial.println("OK STOP");
    return;
  }

  if (line == "PING") {
    Serial.println("OK PONG");
    return;
  }

  if (line == "E?") {
    sendTelemetry(true);
    return;
  }

  if (line == "P?") {
    Serial.print("P ");
    Serial.print(pedalPressed ? 1 : 0);
    Serial.print(" ");
    Serial.print(pedalPercent, 1);
    Serial.print(" ");
    Serial.print((int)pedalAdcFiltered);
    Serial.print(" ");
    Serial.println(adcToVoltage(pedalAdcFiltered), 2);
    return;
  }

  if (!line.startsWith("M ")) {
    Serial.println("ERR BAD_COMMAND");
    return;
  }

  int firstSpace = line.indexOf(' ');
  int secondSpace = line.indexOf(' ', firstSpace + 1);

  if (firstSpace < 0 || secondSpace < 0) {
    Serial.println("ERR BAD_M_FORMAT");
    return;
  }

  int pwm = line.substring(firstSpace + 1, secondSpace).toInt();
  int dir = line.substring(secondSpace + 1).toInt();

  pwm = constrain(pwm, 0, MAX_PWM_LIMIT);
  dir = (dir != 0) ? 1 : 0;

  setMotor(pwm, dir);
  lastCommandTime = millis();

  Serial.print("OK M ");
  Serial.print(currentPwm);
  Serial.print(" ");
  Serial.print(dir);

  if (pedalPressed && pwm > 0) {
    Serial.print(" PEDAL_BLOCKED");
  }

  Serial.println();
}


// ============================================================
// Telemetry
// ============================================================
float adcToVoltage(float adc) {
  return adc * (5.0 / 1023.0);
}


void sendTelemetry(bool forceSend) {
  unsigned long now = millis();

  if (!forceSend && (now - lastTelemetryTime < TELEMETRY_INTERVAL_MS)) {
    return;
  }

  unsigned long dtMs = now - lastTelemetryTime;
  if (dtMs == 0) {
    return;
  }

  long pulseCountSnapshot = enc.read();
  long deltaPulses = pulseCountSnapshot - lastPulseCountForSpeed;

  float dtSec = dtMs / 1000.0;
  float motorRevs = abs(deltaPulses) / ENCODER_PPR;   // motor shaft revolutions
  float wheelRevs = motorRevs / TOTAL_GEAR_RATIO;     // wheel revolutions
  float rpm = (wheelRevs / dtSec) * 60.0;             // wheel RPM
  float speedMps = (wheelRevs * WHEEL_CIRCUMFERENCE_M) / dtSec;

  int signedDirection = 0;
  if (deltaPulses > 0) {
    signedDirection = 1;
  } else if (deltaPulses < 0) {
    signedDirection = -1;
  } else {
    signedDirection = currentDir == 1 ? 1 : -1;
  }

  Serial.print("T ");
  Serial.print(rpm, 2);
  Serial.print(" ");
  Serial.print(speedMps, 3);
  Serial.print(" ");
  Serial.print(pulseCountSnapshot);
  Serial.print(" ");
  Serial.print(signedDirection);
  Serial.print(" ");
  Serial.print(pedalPercent, 1);
  Serial.print(" ");
  Serial.print(pedalPressed ? 1 : 0);
  Serial.print(" ");
  Serial.print((int)pedalAdcFiltered);
  Serial.print(" ");
  Serial.println(adcToVoltage(pedalAdcFiltered), 2);

  lastPulseCountForSpeed = pulseCountSnapshot;
  lastTelemetryTime = now;
}


// ============================================================
// Setup / loop
// ============================================================
void setup() {
  pinMode(PIN_MOTOR_PWM, OUTPUT);
  pinMode(PIN_MOTOR_DIR, OUTPUT);

  // Encoder library sets INPUT_PULLUP on PIN_ENC_A/B automatically.

  pinMode(PIN_PEDAL_ANALOG, INPUT);

  if (USE_PEDAL_OUT) {
    pinMode(PIN_PEDAL_OUT, INPUT);  // open-drain inactive
  }

  stopMotor();
  digitalWrite(PIN_MOTOR_DIR, LOW);

  // Fast PWM 62.5 kHz on D9/D10 (Timer 1). Reduces EMI coupling into encoder cables.
  // Encoder library uses quadrature validation for additional noise rejection.
  // millis() uses Timer 0 — not affected.
  TCCR1A |= 0x01;                              // WGM10 = 1 (fast PWM, 8-bit)
  TCCR1B = (TCCR1B & 0b11100111) | 0x09;      // WGM12=1, prescaler=1 → 62.5 kHz

  Serial.begin(115200);
  Serial.setTimeout(20);

  // Encoder library attaches interrupts on PIN_ENC_A and PIN_ENC_B automatically.

  // Initialize pedal filter with real current value.
  pedalAdcFiltered = analogRead(PIN_PEDAL_ANALOG);
  updatePedal();

  lastCommandTime = millis();
  lastTelemetryTime = millis();

  Serial.println("READY ARDUINO MDD10 ENCODER ANALOG PEDAL CONTROLLER");

  Serial.print("PEDAL_CAL ADC_RELEASED=");
  Serial.print(PEDAL_RELEASED_ADC);
  Serial.print(" ADC_FULL=");
  Serial.print(PEDAL_FULL_ADC);
  Serial.print(" OVERRIDE_PERCENT=");
  Serial.println(PEDAL_OVERRIDE_PERCENT, 1);

  sendTelemetry(true);
}


void loop() {
  updatePedal();

  while (Serial.available() > 0) {
    char c = (char)Serial.read();

    if (c == '\n' || c == '\r') {
      if (inputLine.length() > 0) {
        handleCommand(inputLine);
        inputLine = "";
      }
    } else {
      inputLine += c;

      if (inputLine.length() > 80) {
        inputLine = "";
        Serial.println("ERR LINE_TOO_LONG");
      }
    }
  }

  // Safety timeout
  if (millis() - lastCommandTime > COMMAND_TIMEOUT_MS) {
    if (currentPwm != 0) {
      stopMotor();
      Serial.println("SAFETY STOP TIMEOUT");
    }
  }

  sendTelemetry(false);
}
