// PPR Test - Spin wheel ONE full rotation, read the count
// Encoder A → Pin 2, Encoder B → Pin 3
//
// INPUT_PULLUP holds pins HIGH by default.
// When encoder fires, it pulls line LOW → FALLING edge on A.
// B == LOW means encoder is actively driving B (real quadrature signal).
// If count stays 0 while spinning, swap FALLING→RISING and LOW→HIGH.

#define ENC_A 2
#define ENC_B 3

volatile long pulseCount = 0;

void onPulse() {
  if (digitalRead(ENC_B) == LOW) {
    pulseCount++;
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(ENC_A, INPUT_PULLUP);
  pinMode(ENC_B, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(ENC_A), onPulse, FALLING);

  Serial.println("=== PPR Tester ===");
  Serial.println("Send 'r' to reset, spin ONE full rotation, read the result.");
  Serial.println("If count stays 0 while spinning, swap FALLING->RISING and LOW->HIGH.\n");
}

void loop() {
  static unsigned long lastPrint = 0;
  if (millis() - lastPrint >= 200) {
    lastPrint = millis();
    long count;
    noInterrupts();
    count = pulseCount;
    interrupts();
    Serial.print("PPR: ");
    Serial.println(count);
  }

  if (Serial.available()) {
    char c = Serial.read();
    if (c == 'r' || c == 'R') {
      noInterrupts();
      pulseCount = 0;
      interrupts();
      Serial.println("--- Reset! Spin now ---");
    }
  }
}
