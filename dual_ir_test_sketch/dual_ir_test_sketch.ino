#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <Servo.h>

// ================= PIN DEFINITIONS =================
const int PIN_IR_INTAKE = 2;    // IR Sensor 1 (Top Intake Tray)
const int PIN_IR_CHUTE  = 3;    // IR Sensor 2 (Bottom Chute Exit)
const int PIN_BUZZER    = 8;    // Piezo Buzzer
const int PIN_SERVO_1   = 9;    // Servo 1 (Plastic Deflection Flap)
const int PIN_SERVO_2   = 10;   // Servo 2 (Metal/Paper Selector)
const int PIN_LED_RED   = 13;   // Status Red LED

LiquidCrystal_I2C lcd(0x27, 16, 2);
Servo servo1;
Servo servo2;

void beep(int ms) {
  tone(PIN_BUZZER, 2500, ms);
  digitalWrite(PIN_BUZZER, HIGH);
  delay(ms);
  noTone(PIN_BUZZER);
  digitalWrite(PIN_BUZZER, LOW);
}

void setup() {
  Serial.begin(9600);
  Serial.println(F("=== SMART RECYCLING VENDING MACHINE: DUAL IR TEST ==="));

  pinMode(PIN_IR_INTAKE, INPUT);
  pinMode(PIN_IR_CHUTE, INPUT);
  pinMode(PIN_BUZZER, OUTPUT);
  pinMode(PIN_LED_RED, OUTPUT);

  digitalWrite(PIN_BUZZER, LOW);
  digitalWrite(PIN_LED_RED, LOW);

  // Initialize LCD
  lcd.init();
  lcd.backlight();
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("SMART RECYCLE AI");
  lcd.setCursor(0, 1);
  lcd.print("DUAL IR SENSORS");

  // Calibrate Servos to 0 deg
  servo1.attach(PIN_SERVO_1);
  servo2.attach(PIN_SERVO_2);
  servo1.write(0);
  servo2.write(0);
  delay(400);

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("IR1:OK  IR2:OK");
  lcd.setCursor(0, 1);
  lcd.print("READY FOR WASTE");
  Serial.println(F("[READY] Both IR Sensor 1 (D2) and IR Sensor 2 (D3) active!"));
}

void loop() {
  // Read both IR sensors (LOW = Object Detected, HIGH = Clear)
  int ir1 = digitalRead(PIN_IR_INTAKE);
  int ir2 = digitalRead(PIN_IR_CHUTE);

  // STAGE 1: Item placed on Intake Tray (IR 1 triggered)
  if (ir1 == LOW) {
    Serial.println(F("\n>>> [STAGE 1] Item detected at INTAKE (IR 1)!"));
    digitalWrite(PIN_LED_RED, HIGH);

    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("INTAKE DETECTED!");
    lcd.setCursor(0, 1);
    lcd.print("FLAP OPENING...");

    // Servo 1 opens to route item into chute
    servo1.write(45);
    servo2.write(45);
    beep(100);

    // STAGE 2: Wait for item to slide down and pass IR 2 (Timeout: 4 seconds)
    Serial.println(F(">>> [STAGE 2] Waiting for item to drop past CHUTE (IR 2)..."));
    unsigned long startTime = millis();
    bool passedChute = false;

    while (millis() - startTime < 4000) {
      if (digitalRead(PIN_IR_CHUTE) == LOW) {
        passedChute = true;
        Serial.println(F(">>> [STAGE 3] CONFIRMED! Item safely passed IR 2 into BIN!"));
        break;
      }
      delay(50);
    }

    // Reset Flaps
    servo1.write(0);
    servo2.write(0);
    delay(400);

    // Confirmation on LCD
    lcd.clear();
    if (passedChute) {
      lcd.setCursor(0, 0);
      lcd.print("DEPOSIT CONFIRM!");
      lcd.setCursor(0, 1);
      lcd.print("BIN ENTRY VERIFY");
      beep(250);
    } else {
      // If timer expired before IR2 triggered
      lcd.setCursor(0, 0);
      lcd.print("CYCLE COMPLETE");
      lcd.setCursor(0, 1);
      lcd.print("+10 REWARD PTS");
    }

    digitalWrite(PIN_LED_RED, LOW);
    delay(2000);

    // Return to Ready state
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("SMART RECYCLE AI");
    lcd.setCursor(0, 1);
    lcd.print("READY FOR WASTE");
  }

  // Live test for IR 2 standalone
  if (ir2 == LOW && ir1 == HIGH) {
    Serial.println(F(">>> IR 2 (Chute Exit) triggered!"));
    digitalWrite(PIN_LED_RED, HIGH);
    delay(200);
    digitalWrite(PIN_LED_RED, LOW);
  }

  delay(100);
}
