#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <Servo.h>

// ================= PIN DEFINITIONS =================
const int PIN_IR_INTAKE = 2;    // IR Proximity Sensor OUT
const int PIN_BUZZER    = 8;    // Piezo Buzzer (D8)
const int PIN_SERVO_1   = 9;    // Servo 1 (Plastic Deflection Flap)
const int PIN_SERVO_2   = 10;   // Servo 2 (Metal/Paper Selector)
const int PIN_LED_RED   = 13;   // Status Red LED

LiquidCrystal_I2C lcd(0x27, 16, 2);
Servo servo1;
Servo servo2;

// Loud beep function supporting BOTH Active and Passive Buzzers!
void beep(int ms) {
  // Method 1: Tone frequency (for passive buzzers)
  tone(PIN_BUZZER, 2500, ms);
  // Method 2: Direct High (for active buzzers)
  digitalWrite(PIN_BUZZER, HIGH);
  delay(ms);
  noTone(PIN_BUZZER);
  digitalWrite(PIN_BUZZER, LOW);
}

void setup() {
  Serial.begin(9600);
  Serial.println(F("=== SMART RECYCLING VENDING MACHINE: TEST RUN ==="));

  pinMode(PIN_IR_INTAKE, INPUT);
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
  lcd.print("SYSTEM READY :)");

  // Power-on Beep
  digitalWrite(PIN_LED_RED, HIGH);
  beep(250);
  digitalWrite(PIN_LED_RED, LOW);

  // Initialize Servos
  servo1.attach(PIN_SERVO_1);
  servo2.attach(PIN_SERVO_2);
  servo1.write(0);
  servo2.write(0);
  delay(300);

  Serial.println(F("[OK] System Ready. Bring hand in front of IR Sensor to test!"));
}

void loop() {
  int irState = digitalRead(PIN_IR_INTAKE);

  if (irState == LOW) {
    Serial.println(F(">>> ITEM DETECTED BY IR SENSOR!"));
    digitalWrite(PIN_LED_RED, HIGH);

    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("ITEM DETECTED!");
    lcd.setCursor(0, 1);
    lcd.print("SORTING WASTE...");

    // Double Beep
    beep(150);
    delay(100);
    beep(150);

    // Test Servo 1 (Plastic Route)
    Serial.println(F("Moving Servo 1 to 45 deg..."));
    servo1.write(45);
    delay(1000);

    // Test Servo 2 (Metal Route)
    Serial.println(F("Moving Servo 2 to 45 deg..."));
    servo2.write(45);
    delay(1000);

    // Reset Servos
    servo1.write(0);
    servo2.write(0);
    delay(500);

    // Confirmation Beep & LCD
    beep(300);
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("PLASTIC ACCEPTED");
    lcd.setCursor(0, 1);
    lcd.print("+10 POINTS REWARD");
    Serial.println(F("[SUCCESS] Sorting Cycle Complete!\n"));

    digitalWrite(PIN_LED_RED, LOW);
    delay(2500);

    // Reset back to Idle
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("SMART RECYCLE AI");
    lcd.setCursor(0, 1);
    lcd.print("INSERT RECYCLE");
  }

  delay(100);
}
