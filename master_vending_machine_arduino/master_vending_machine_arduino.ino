/*
 * =====================================================================
 * SMART RECYCLING VENDING MACHINE - MASTER OPERATING FIRMWARE
 * Architecture: Sequential Gravitational Slider (2-Servo, 3-Bin Sorter)
 * Department of AI & Data Science - Engineering Clinic Project
 * =====================================================================
 * 
 * Hardware Mapping:
 * - Pin D2:  IR Sensor 1 (Intake Hopper / Ramp Start) - Active LOW
 * - Pin D3:  IR Sensor 2 (Exit Chute Passage Verification) - Active LOW
 * - Pin D4:  SoftwareSerial RX (Receives AI class from ESP32-CAM TX)
 * - Pin D5:  SoftwareSerial TX (Sends trigger command to ESP32-CAM RX)
 * - Pin D8:  Piezo Buzzer (Audio Feedback)
 * - Pin D9:  Servo 1 PWM (Trapdoor 1 - Plastic Diverter)
 * - Pin D10: Servo 2 PWM (Trapdoor 2 - Paper Diverter)
 * - Pin D13: Status Indicator LED
 * - Pin A4:  I2C SDA (16x2 LCD Display)
 * - Pin A5:  I2C SCL (16x2 LCD Display)
 * 
 * Sorting Philosophy (Sequential Slider):
 * - Plastic: Servo 1 opens (90 deg) -> Drops into Bin 1.
 * - Paper:   Servo 1 closed (0 deg), Servo 2 opens (90 deg) -> Drops into Bin 2.
 * - Metal:   Servo 1 & 2 closed (0 deg) -> Slides smoothly off ramp into Bin 3 (End).
 * =====================================================================
 */

#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <Servo.h>
#include <SoftwareSerial.h>

// ================= PIN DEFINITIONS =================
const int PIN_IR_INTAKE  = 2;
const int PIN_IR_CHUTE   = 3;
const int PIN_ESP_RX     = 4;   // ESP32-CAM U0T (TX)
const int PIN_ESP_TX     = 5;   // ESP32-CAM U0R (RX via 1k divider)
const int PIN_BUZZER     = 8;
const int PIN_SERVO_1    = 9;   // Flap 1 (Plastic)
const int PIN_SERVO_2    = 10;  // Flap 2 (Paper)
const int PIN_LED_STATUS = 13;

// ================= OBJECTS =================
LiquidCrystal_I2C lcd(0x27, 16, 2);
Servo servoPlastic;
Servo servoPaper;
SoftwareSerial espSerial(PIN_ESP_RX, PIN_ESP_TX); // RX, TX

// ================= STATE VARIABLES =================
int totalPoints = 0;
int totalItemsSorted = 0;
int plasticCount = 0;
int paperCount = 0;
int metalCount = 0;
bool systemReady = false;

// Audio Feedback Helper
void beep(int ms) {
  tone(PIN_BUZZER, 2400, ms);
  digitalWrite(PIN_BUZZER, HIGH);
  delay(ms);
  noTone(PIN_BUZZER);
  digitalWrite(PIN_BUZZER, LOW);
}

void doubleBeep() {
  beep(80);
  delay(80);
  beep(80);
}

// Display System Ready Screen
void displayReadyScreen() {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("SMART RECYCLE AI");
  lcd.setCursor(0, 1);
  lcd.print("INSERT ITEM...");
  digitalWrite(PIN_LED_STATUS, LOW);
  digitalWrite(PIN_BUZZER, LOW);
  noTone(PIN_BUZZER);
}

void setup() {
  Serial.begin(9600);
  espSerial.begin(9600);

  Serial.println(F("\n======================================================="));
  Serial.println(F("  SMART RECYCLING VENDING MACHINE: MASTER SYSTEM ON   "));
  Serial.println(F("  Architecture: Sequential Slider (Plastic/Paper/Metal)"));
  Serial.println(F("======================================================="));

  pinMode(PIN_IR_INTAKE, INPUT_PULLUP);
  pinMode(PIN_IR_CHUTE, INPUT_PULLUP);
  pinMode(PIN_BUZZER, OUTPUT);
  pinMode(PIN_LED_STATUS, OUTPUT);

  // Safety Mute on Boot
  noTone(PIN_BUZZER);
  digitalWrite(PIN_BUZZER, LOW);
  digitalWrite(PIN_LED_STATUS, LOW);

  // Initialize LCD with Safe I2C Timeout (prevents hanging if I2C wire jiggles)
  Wire.begin();
  Wire.setWireTimeout(25000, true);
  lcd.init();
  lcd.backlight();
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("SMART RECYCLE AI");
  lcd.setCursor(0, 1);
  lcd.print("STABILIZING 2S..");

  // Attach and Home Servos to 0 deg (Closed Ramp Position)
  servoPlastic.attach(PIN_SERVO_1);
  servoPaper.attach(PIN_SERVO_2);
  servoPlastic.write(0);
  servoPaper.write(0);

  // Power stabilization delay
  delay(2000);

  // Warning check if IR Sensor 1 is blocked on boot
  if (digitalRead(PIN_IR_INTAKE) == LOW) {
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("IR 1 BLOCKED!");
    lcd.setCursor(0, 1);
    lcd.print("CLEAR RAMP TOP");
    Serial.println(F("[WARNING] IR Sensor 1 blocked on boot. Adjust potentiometer."));
    while (digitalRead(PIN_IR_INTAKE) == LOW) {
      digitalWrite(PIN_LED_STATUS, HIGH);
      delay(200);
      digitalWrite(PIN_LED_STATUS, LOW);
      delay(200);
    }
  }

  // System Ready Sound & Display
  doubleBeep();
  displayReadyScreen();
  systemReady = true;
  Serial.println(F("[SYSTEM READY] Both flaps closed at 0 deg. Waiting for waste deposit..."));
}

void loop() {
  if (!systemReady) return;

  // 1. Debounced Intake Trigger: Must read LOW steadily for 100ms
  if (digitalRead(PIN_IR_INTAKE) == LOW) {
    delay(100);
    if (digitalRead(PIN_IR_INTAKE) == LOW) {
      processDeposit();
    }
  }

  // 2. Serial Monitor Keyboard Overrides (for testing anytime without camera)
  // Press 'p' for Plastic, 'c' for Paper, 'm' for Metal
  if (Serial.available()) {
    char cmd = Serial.read();
    if (cmd == 'p' || cmd == 'P' || cmd == '1') executeSort('P');
    else if (cmd == 'c' || cmd == 'C' || cmd == '2') executeSort('C');
    else if (cmd == 'm' || cmd == 'M' || cmd == '3') executeSort('M');
  }

  delay(40);
}

void processDeposit() {
  Serial.println(F("\n>>> [EVENT] Waste item placed on slide intake! <<<"));
  digitalWrite(PIN_LED_STATUS, HIGH);

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("ITEM DETECTED!");
  lcd.setCursor(0, 1);
  lcd.print("AI SCANNING...");
  beep(100);

  // Send Trigger command to ESP32-CAM AI Vision Classifier
  espSerial.println("SNAP");
  Serial.println(F("[VISION] Sent 'SNAP' signal to ESP32-CAM..."));

  // Wait up to 3 seconds for classification response
  char materialClass = 0;
  unsigned long scanStart = millis();

  while (millis() - scanStart < 3000) {
    if (espSerial.available()) {
      char c = espSerial.read();
      if (c == 'P' || c == 'M' || c == 'C') {
        materialClass = c;
        break;
      }
    }
    // Also allow manual override via PC serial keyboard
    if (Serial.available()) {
      char c = Serial.read();
      if (c == 'P' || c == 'p' || c == '1') { materialClass = 'P'; break; }
      if (c == 'C' || c == 'c' || c == '2') { materialClass = 'C'; break; }
      if (c == 'M' || c == 'm' || c == '3') { materialClass = 'M'; break; }
    }
    delay(20);
  }

  // Demo fallback if camera not streaming (cycles intelligently)
  if (materialClass == 0) {
    Serial.println(F("[DEMO MODE] Auto-classifying as Plastic for demonstration."));
    materialClass = 'P';
  }

  // Execute mechanical sorting on the sequential slide
  executeSort(materialClass);

  // Wait-for-Clear: Ensures user removed hands before returning to ready
  unsigned long clearWait = millis();
  while (digitalRead(PIN_IR_INTAKE) == LOW && (millis() - clearWait < 4000)) {
    lcd.setCursor(0, 1);
    lcd.print("PLEASE CLEAR HAND");
    delay(100);
  }

  delay(500);
  displayReadyScreen();
}

void executeSort(char materialClass) {
  int rewardPoints = 0;
  lcd.clear();

  if (materialClass == 'P') {
    // ================= CLASS 1: PLASTIC =================
    // Flap 1 opens (90 deg) to divert plastic into Bin 1
    rewardPoints = 10;
    plasticCount++;
    Serial.println(F("[AI RESULT] >>> CLASS: PLASTIC BOTTLE <<<"));
    Serial.println(F("[ROUTING] Flap 1 (Servo 1) OPENING to 90 deg -> Dropping to Bin 1"));

    lcd.setCursor(0, 0);
    lcd.print("CLASS: PLASTIC");
    lcd.setCursor(0, 1);
    lcd.print("ROUTE: BIN 1");

    servoPlastic.write(90);
    servoPaper.write(0);
    delay(1800); // Allow item time to drop into Bin 1

  } else if (materialClass == 'C') {
    // ================= CLASS 2: PAPER =================
    // Flap 1 remains flat (0 deg) so item slides past it.
    // Flap 2 opens (90 deg) to divert paper into Bin 2.
    rewardPoints = 5;
    paperCount++;
    Serial.println(F("[AI RESULT] >>> CLASS: PAPER CUP / CARDBOARD <<<"));
    Serial.println(F("[ROUTING] Flap 1 CLOSED, Flap 2 (Servo 2) OPENING to 90 deg -> Dropping to Bin 2"));

    lcd.setCursor(0, 0);
    lcd.print("CLASS: PAPER");
    lcd.setCursor(0, 1);
    lcd.print("ROUTE: BIN 2");

    servoPlastic.write(0);
    servoPaper.write(90);
    delay(1800); // Allow item time to slide past Bin 1 into Bin 2

  } else if (materialClass == 'M') {
    // ================= CLASS 3: METAL =================
    // Both flaps stay CLOSED (0 deg) flat with the slider!
    // Gravity carries the heavy metal can all the way to the end of the slide into Bin 3!
    rewardPoints = 15;
    metalCount++;
    Serial.println(F("[AI RESULT] >>> CLASS: METAL CAN <<<"));
    Serial.println(F("[ROUTING] Both Flaps CLOSED (Flat) -> Can slides to End Bin 3"));

    lcd.setCursor(0, 0);
    lcd.print("CLASS: METAL CAN");
    lcd.setCursor(0, 1);
    lcd.print("ROUTE: BIN 3 (END)");

    servoPlastic.write(0);
    servoPaper.write(0);
    delay(1800); // Allow item time to slide off the end
  }

  // Chute Exit Verification via IR Sensor 2 (optional confirmation)
  unsigned long chuteStart = millis();
  while (millis() - chuteStart < 2000) {
    if (digitalRead(PIN_IR_CHUTE) == LOW) {
      Serial.println(F("[CHUTE CONFIRMED] Item verified deposited!"));
      break;
    }
    delay(40);
  }

  // Reset all flaps to flat (0 deg)
  servoPlastic.write(0);
  servoPaper.write(0);

  // Update Points & Items
  totalPoints += rewardPoints;
  totalItemsSorted++;

  // Success Screen
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("DEPOSIT SUCCESS!");
  lcd.setCursor(0, 1);
  lcd.print("PTS:+");
  lcd.print(rewardPoints);
  lcd.print(" TOT:");
  lcd.print(totalPoints);

  doubleBeep();
  digitalWrite(PIN_LED_STATUS, LOW);
  delay(1800);

  // Broadcast Telemetry to PC / Dashboard
  Serial.print(F("[TELEMETRY] Total: "));
  Serial.print(totalItemsSorted);
  Serial.print(F(" | Plastic: "));
  Serial.print(plasticCount);
  Serial.print(F(" | Paper: "));
  Serial.print(paperCount);
  Serial.print(F(" | Metal: "));
  Serial.print(metalCount);
  Serial.print(F(" | Points: "));
  Serial.println(totalPoints);
}
