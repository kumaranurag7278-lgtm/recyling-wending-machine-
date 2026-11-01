/*
 * SMART RECYCLING VENDING MACHINE - MASTER CONTROLLER FIRMWARE
 * Department of AI & Data Science - Engineering Clinic
 * 
 * Hardware Mapping:
 * - Pin D2: IR Sensor 1 (Intake Staging Tray) - Active LOW
 * - Pin D3: IR Sensor 2 (Exit Chute Verification) - Active LOW
 * - Pin D4: SoftwareSerial RX (Receives AI class from ESP32-CAM TX)
 * - Pin D5: SoftwareSerial TX (Sends trigger command to ESP32-CAM RX)
 * - Pin D8: Buzzer (Active/Passive)
 * - Pin D9: Servo PWM (Central Carousel Turntable Indexer: 0° / 90° / 180°)
 * - Pin D13: Status Indicator LED
 * - Pin A4: I2C SDA (16x2 LCD)
 * - Pin A5: I2C SCL (16x2 LCD)
 */

#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <Servo.h>
#include <SoftwareSerial.h>

// ================= PIN DEFINITIONS =================
const int PIN_IR_INTAKE  = 2;
const int PIN_IR_CHUTE   = 3;
const int PIN_ESP_RX     = 4;   // Connects to ESP32-CAM U0T (TX)
const int PIN_ESP_TX     = 5;   // Connects to ESP32-CAM U0R (RX via 1k divider)
const int PIN_BUZZER     = 8;
const int PIN_SERVO      = 9;   // Single Central Carousel Servo
const int PIN_LED_STATUS = 13;

// ================= OBJECTS =================
LiquidCrystal_I2C lcd(0x27, 16, 2);
Servo carouselServo;
SoftwareSerial espSerial(PIN_ESP_RX, PIN_ESP_TX); // RX, TX

// State variables
int totalPoints = 0;
int totalItemsSorted = 0;
bool systemReady = false;

void beepSound(int ms) {
  tone(PIN_BUZZER, 2400, ms);
  digitalWrite(PIN_BUZZER, HIGH);
  delay(ms);
  noTone(PIN_BUZZER);
  digitalWrite(PIN_BUZZER, LOW);
}

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

  pinMode(PIN_IR_INTAKE, INPUT_PULLUP);
  pinMode(PIN_IR_CHUTE, INPUT_PULLUP);
  pinMode(PIN_BUZZER, OUTPUT);
  pinMode(PIN_LED_STATUS, OUTPUT);

  // Ensure buzzer & LED are strictly OFF on boot
  noTone(PIN_BUZZER);
  digitalWrite(PIN_BUZZER, LOW);
  digitalWrite(PIN_LED_STATUS, LOW);

  // Initialize LCD
  lcd.init();
  lcd.backlight();
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("SMART RECYCLE AI");
  lcd.setCursor(0, 1);
  lcd.print("STABILIZING 2S..");

  // Attach and zero-calibrate Single Servo (0 deg = Plastic default)
  carouselServo.attach(PIN_SERVO);
  carouselServo.write(0);

  // Power stabilization delay (prevents adapter surge false triggers)
  delay(2000);

  // Check if IR sensor is stuck triggered at boot
  if (digitalRead(PIN_IR_INTAKE) == LOW) {
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("IR 1 BLOCKED!");
    lcd.setCursor(0, 1);
    lcd.print("ADJUST TRIMPOT");
    Serial.println(F("[WARNING] IR Sensor 1 is reading LOW on startup! Adjust potentiometer."));
    while (digitalRead(PIN_IR_INTAKE) == LOW) {
      // Flash LED warning without sound so it doesn't irritate
      digitalWrite(PIN_LED_STATUS, HIGH);
      delay(200);
      digitalWrite(PIN_LED_STATUS, LOW);
      delay(200);
    }
  }

  // Double beep when ready
  beepSound(80);
  delay(100);
  beepSound(80);

  displayReadyScreen();
  systemReady = true;
  Serial.println(F("[SYSTEM READY] Carousel initialized at 0 deg. Waiting for waste deposit..."));
}

void loop() {
  if (!systemReady) return;

  // Debounced intake trigger: must read LOW steadily for 100ms
  if (digitalRead(PIN_IR_INTAKE) == LOW) {
    delay(100); // Debounce noise
    if (digitalRead(PIN_IR_INTAKE) == LOW) {
      processDeposit();
    }
  }

  // Serial Monitor Keyboard Overrides (p = Plastic, m = Metal, c = Paper)
  if (Serial.available()) {
    char cmd = Serial.read();
    if (cmd == 'p' || cmd == 'P') executeSort('P');
    else if (cmd == 'm' || cmd == 'M') executeSort('M');
    else if (cmd == 'c' || cmd == 'C') executeSort('C');
  }

  delay(50);
}

void processDeposit() {
  Serial.println(F("\n>>> [EVENT] Item detected at intake hopper!"));
  digitalWrite(PIN_LED_STATUS, HIGH);

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("ITEM DETECTED!");
  lcd.setCursor(0, 1);
  lcd.print("AI SCANNING...");
  beepSound(100);

  // Trigger ESP32-CAM
  espSerial.println("SNAP");
  Serial.println(F("[COMMAND] Sent 'SNAP' trigger to ESP32-CAM..."));

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
    if (Serial.available()) {
      char c = Serial.read();
      if (c == 'P' || c == 'p') { materialClass = 'P'; break; }
      if (c == 'M' || c == 'm') { materialClass = 'M'; break; }
      if (c == 'C' || c == 'c') { materialClass = 'C'; break; }
    }
    delay(20);
  }

  // If no response from ESP32, default demo cycle
  if (materialClass == 0) {
    Serial.println(F("[NOTICE] ESP32 standby. Auto-classifying as Plastic for demo."));
    materialClass = 'P';
  }

  executeSort(materialClass);

  // CRITICAL: Wait until user removes hands / item clears IR sensor
  // Prevents infinite loop re-triggering and non-stop buzzing!
  unsigned long clearWait = millis();
  while (digitalRead(PIN_IR_INTAKE) == LOW && (millis() - clearWait < 5000)) {
    lcd.setCursor(0, 1);
    lcd.print("PLEASE CLEAR HAND");
    delay(100);
  }

  delay(500);
  displayReadyScreen();
}

void executeSort(char materialClass) {
  int rewardPoints = 0;
  int targetAngle = 0;

  lcd.clear();

  if (materialClass == 'P') {
    rewardPoints = 10;
    targetAngle = 0; // 0 deg = Plastic Compartment
    Serial.println(F("[AI RESULT] >>> CLASS: PLASTIC BOTTLE (Angle 0 deg) <<<"));
    lcd.setCursor(0, 0);
    lcd.print("CLASS: PLASTIC");
    lcd.setCursor(0, 1);
    lcd.print("ROTATE: 0 DEG");
  } else if (materialClass == 'M') {
    rewardPoints = 15;
    targetAngle = 90; // 90 deg = Metal Compartment
    Serial.println(F("[AI RESULT] >>> CLASS: METAL CAN (Angle 90 deg) <<<"));
    lcd.setCursor(0, 0);
    lcd.print("CLASS: METAL CAN");
    lcd.setCursor(0, 1);
    lcd.print("ROTATE: 90 DEG");
  } else if (materialClass == 'C') {
    rewardPoints = 5;
    targetAngle = 180; // 180 deg = Paper Compartment
    Serial.println(F("[AI RESULT] >>> CLASS: PAPER (Angle 180 deg) <<<"));
    lcd.setCursor(0, 0);
    lcd.print("CLASS: PAPER CUP");
    lcd.setCursor(0, 1);
    lcd.print("ROTATE: 180 DEG");
  }

  // Move single servo to index carousel
  carouselServo.write(targetAngle);
  delay(600); // Allow servo time to rotate

  // Verify item passage through chute exit IR sensor
  Serial.println(F("[CHUTE] Waiting for item to drop into compartment..."));
  unsigned long chuteStart = millis();
  while (millis() - chuteStart < 2500) {
    if (digitalRead(PIN_IR_CHUTE) == LOW) {
      Serial.println(F("[CHUTE CONFIRMED] Item verified in collection bin!"));
      break;
    }
    delay(50);
  }

  // Update Score and Show Confirmation
  totalPoints += rewardPoints;
  totalItemsSorted++;

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("DEPOSIT SUCCESS!");
  lcd.setCursor(0, 1);
  lcd.print("PTS:+");
  lcd.print(rewardPoints);
  lcd.print(" TOT:");
  lcd.print(totalPoints);

  beepSound(150);
  digitalWrite(PIN_LED_STATUS, LOW);
  delay(1500);

  // Return carousel to 0 deg home
  carouselServo.write(0);
}
