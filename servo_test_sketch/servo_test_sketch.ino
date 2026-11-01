#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <Servo.h>

LiquidCrystal_I2C lcd(0x27, 16, 2);
Servo servo1;
Servo servo2;

const int PIN_SERVO_1 = 9;   // Orange wire of Servo 1
const int PIN_SERVO_2 = 10;  // Orange wire of Servo 2
const int PIN_LED     = 13;

void setup() {
  Serial.begin(9600);
  Serial.println(F("\n=== DEDICATED SERVO MOTOR TEST ==="));

  pinMode(PIN_LED, OUTPUT);
  digitalWrite(PIN_LED, HIGH);

  lcd.init();
  lcd.backlight();
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("SERVO SWEEP TEST");
  lcd.setCursor(0, 1);
  lcd.print("PIN 9 & PIN 10");

  Serial.println(F("Attaching Servo 1 to Pin 9..."));
  servo1.attach(PIN_SERVO_1);
  Serial.println(F("Attaching Servo 2 to Pin 10..."));
  servo2.attach(PIN_SERVO_2);

  delay(1000);
}

void loop() {
  // Test 1: Neutral Position (0 deg)
  Serial.println(F(">>> POS 1: Both Servos to 0 DEGREES (Neutral)"));
  lcd.clear();
  lcd.setCursor(0, 0); lcd.print("SERVOS -> 0 DEG");
  lcd.setCursor(0, 1); lcd.print("NEUTRAL CLOSED");
  servo1.write(0);
  servo2.write(0);
  digitalWrite(PIN_LED, HIGH);
  delay(1500);

  // Test 2: Plastic Sorting Position (45 deg)
  Serial.println(F(">>> POS 2: Servo 1 to 45 DEGREES (Plastic Divert)"));
  lcd.clear();
  lcd.setCursor(0, 0); lcd.print("SERVO 1 -> 45 DEG");
  lcd.setCursor(0, 1); lcd.print("PLASTIC CHUTE");
  servo1.write(45);
  digitalWrite(PIN_LED, LOW);
  delay(1500);

  // Test 3: Drop Gate Open (90 deg)
  Serial.println(F(">>> POS 3: Both Servos to 90 DEGREES (Drop Gate Open)"));
  lcd.clear();
  lcd.setCursor(0, 0); lcd.print("SERVOS -> 90 DEG");
  lcd.setCursor(0, 1); lcd.print("DROP GATE OPEN");
  servo1.write(90);
  servo2.write(90);
  digitalWrite(PIN_LED, HIGH);
  delay(1500);

  // Test 4: Paper Sorting Position (135 deg)
  Serial.println(F(">>> POS 4: Servo 2 to 135 DEGREES (Paper Divert)"));
  lcd.clear();
  lcd.setCursor(0, 0); lcd.print("SERVO 2 -> 135 DEG");
  lcd.setCursor(0, 1); lcd.print("PAPER CHUTE");
  servo2.write(135);
  digitalWrite(PIN_LED, LOW);
  delay(1500);
}
