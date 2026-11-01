/*
 * SMART RECYCLING VENDING MACHINE - MASTER CONTROLLER FIRMWARE
 * Department of AI & Data Science - Engineering Clinic
 * 
 * Hardware Mapping:
 * - Pin D2: IR Sensor 1 (Intake Staging Tray)
 * - Pin D3: IR Sensor 2 (Exit Chute Verification)
 * - Pin D4: SoftwareSerial RX (Receives AI class from ESP32-CAM TX)
 * - Pin D5: SoftwareSerial TX (Sends trigger command to ESP32-CAM RX)
 * - Pin D8: Piezo Buzzer
 * - Pin D9: Servo 1 PWM (Upper Plastic Deflector Flap)
 * - Pin D10: Servo 2 PWM (Lower Metal / Paper Selector Flap)
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
const int PIN_SERVO_1    = 9;
const int PIN_SERVO_2    = 10;
const int PIN_LED_STATUS = 13;

// ================= OBJECTS =================
LiquidCrystal_I2C lcd(0x27, 16, 2);
Servo servo1;
Servo servo2;
SoftwareSerial espSerial(PIN_ESP_RX, PIN_ESP_TX); // RX, TX

// State variables
int totalPoints = 0;
int totalItemsSorted = 0;

void beepSound(int ms) {
  tone(PIN_BUZZER, 2500, ms);
  digitalWrite(PIN_BUZZER, HIGH);
  delay(ms);
  noTone(PIN_BUZZER);
  digitalWrite(PIN_BUZZER, LOW);
}

void setup() {
  Serial.begin(9600);       // PC Debug Serial
  espSerial.begin(9600);    // ESP32-CAM Communication Serial

  Serial.println(F("\n=================================================="));
  Serial.println(F("  SMART RECYCLING VENDING MACHINE: MASTER SYSTEM  "));
  Serial.println(F("=================================================="));

  pinMode(PIN_IR_INTAKE, INPUT);
  pinMode(PIN_IR_CHUTE, INPUT);
  pinMode(PIN_BUZZER, OUTPUT);
  pinMode(PIN_LED_STATUS, OUTPUT);

  digitalWrite(PIN_BUZZER, LOW);
  digitalWrite(PIN_LED_STATUS, LOW);

  // Initialize LCD
  lcd.init();
  lcd.backlight();
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("SMART RECYCLE AI");
  lcd.setCursor(0, 1);
  lcd.print("INITIALIZING...");

  // Attach and zero-calibrate Servos
  servo1.attach(PIN_SERVO_1);
  servo2.attach(PIN_SERVO_2);
  servo1.write(0);
  servo2.write(0);
  delay(500);

  // Ready Screen
  displayReadyScreen();
  Serial.println(F("[SYSTEM READY] Waiting for waste deposit..."));
}

void displayReadyScreen() {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("SMART RECYCLE AI");
  lcd.setCursor(0, 1);
  lcd.print("INSERT ITEM...");
  digitalWrite(PIN_LED_STATUS, LOW);
}

void loop() {
  // Check if waste item is placed in intake hopper (IR 1 triggered LOW)
  if (digitalRead(PIN_IR_INTAKE) == LOW) {
    processDeposit();
  }

  // Also accept manual keyboard override commands from PC Serial Monitor:
  // Type 'p' for Plastic, 'm' for Metal, 'c' for Paper/Cup in Serial Monitor
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

  // Trigger ESP32-CAM to capture image and run MobileNetV2 inference
  espSerial.println("SNAP");
  Serial.println(F("[COMMAND] Sent 'SNAP' trigger to ESP32-CAM..."));

  // Wait for classification result with a 4-second timeout
  char materialClass = 0;
  unsigned long scanStart = millis();

  while (millis() - scanStart < 4000) {
    // Check if ESP32-CAM sent back classification
    if (espSerial.available()) {
      char c = espSerial.read();
      if (c == 'P' || c == 'M' || c == 'C') {
        materialClass = c;
        break;
      }
    }
    // Also allow PC Serial monitor input
    if (Serial.available()) {
      char c = Serial.read();
      if (c == 'P' || c == 'p') { materialClass = 'P'; break; }
      if (c == 'M' || c == 'm') { materialClass = 'M'; break; }
      if (c == 'C' || c == 'c') { materialClass = 'C'; break; }
    }
    delay(20);
  }

  // If no response from ESP32, default demo fallback (auto-cycles or defaults to Plastic)
  if (materialClass == 0) {
    Serial.println(F("[NOTICE] ESP32 standby. Auto-classifying as Plastic for test demo."));
    materialClass = 'P';
  }

  executeSort(materialClass);
}

void executeSort(char materialClass) {
  int rewardPoints = 0;

  lcd.clear();

  if (materialClass == 'P') {
    // ================= CLASS 1: PLASTIC =================
    rewardPoints = 10;
    Serial.println(F("[AI RESULT] >>> CLASS: PLASTIC BOTTLE <<<"));
    lcd.setCursor(0, 0);
    lcd.print("CLASS: PLASTIC");
    lcd.setCursor(0, 1);
    lcd.print("ROUTE: BIN 1");

    // Servo 1 tilts 45 deg to divert plastic directly into Bin 1
    servo1.write(45);
    servo2.write(0);

  } else if (materialClass == 'M') {
    // ================= CLASS 2: METAL =================
    rewardPoints = 15;
    Serial.println(F("[AI RESULT] >>> CLASS: METAL CAN <<<"));
    lcd.setCursor(0, 0);
    lcd.print("CLASS: METAL CAN");
    lcd.setCursor(0, 1);
    lcd.print("ROUTE: BIN 2");

    // Servo 1 drops down (90 deg), Servo 2 diverts 45 deg into Bin 2
    servo1.write(90);
    delay(250);
    servo2.write(45);

  } else if (materialClass == 'C') {
    // ================= CLASS 3: PAPER =================
    rewardPoints = 5;
    Serial.println(F("[AI RESULT] >>> CLASS: PAPER CUP <<<"));
    lcd.setCursor(0, 0);
    lcd.print("CLASS: PAPER CUP");
    lcd.setCursor(0, 1);
    lcd.print("ROUTE: BIN 3");

    // Servo 1 drops down (90 deg), Servo 2 diverts 135 deg into Bin 3
    servo1.write(90);
    delay(250);
    servo2.write(135);
  }

  // Monitor IR Sensor 2 at chute exit to confirm physical bin entry
  Serial.println(F("[CHUTE] Waiting for item to pass IR Sensor 2 into bin..."));
  unsigned long chuteStart = millis();
  bool passed = false;

  while (millis() - chuteStart < 3500) {
    if (digitalRead(PIN_IR_CHUTE) == LOW) {
      passed = true;
      Serial.println(F("[CHUTE CONFIRMED] Item verified in collection bin!"));
      break;
    }
    delay(50);
  }

  // Reset Servos to 0 deg neutral position
  servo1.write(0);
  servo2.write(0);

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

  beepSound(200);
  digitalWrite(PIN_LED_STATUS, LOW);
  delay(2500);

  // Return to ready
  displayReadyScreen();
}
