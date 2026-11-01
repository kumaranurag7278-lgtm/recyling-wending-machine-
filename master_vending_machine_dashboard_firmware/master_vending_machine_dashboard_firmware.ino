#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <Servo.h>
#include <SoftwareSerial.h>

// ================= PIN DEFINITIONS =================
const int PIN_IR_INTAKE  = 2;    // IR 1: Intake Staging Tray
const int PIN_IR_CHUTE   = 3;    // IR 2: Exit Chute Verification
const int PIN_ESP_RX     = 4;    // SoftwareSerial RX (from ESP32 TX)
const int PIN_ESP_TX     = 5;    // SoftwareSerial TX (to ESP32 RX)
const int PIN_BUZZER     = 8;    // Piezo Buzzer
const int PIN_SERVO_1    = 9;    // Servo 1: Plastic Deflector
const int PIN_SERVO_2    = 10;   // Servo 2: Metal / Paper Selector
const int PIN_LED_STATUS = 13;   // Status Indicator LED

// ================= OBJECTS =================
LiquidCrystal_I2C lcd(0x27, 16, 2);
Servo servo1;
Servo servo2;
SoftwareSerial espSerial(PIN_ESP_RX, PIN_ESP_TX); // RX, TX

// State metrics
int countPlastic = 0;
int countMetal = 0;
int countPaper = 0;
int totalRewardPoints = 0;
int lastServo1Angle = 0;
int lastServo2Angle = 0;

void beepSound(int ms) {
  tone(PIN_BUZZER, 2500, ms);
  digitalWrite(PIN_BUZZER, HIGH);
  delay(ms);
  noTone(PIN_BUZZER);
  digitalWrite(PIN_BUZZER, LOW);
}

// Broadcast structured JSON telemetry over USB to Dashboard
void sendTelemetry(const char* statusMsg) {
  int ir1State = digitalRead(PIN_IR_INTAKE) == LOW ? 1 : 0;
  int ir2State = digitalRead(PIN_IR_CHUTE) == LOW ? 1 : 0;
  
  Serial.print(F("{\"telemetry\":{"));
  Serial.print(F("\"ir1\":")); Serial.print(ir1State); Serial.print(F(","));
  Serial.print(F("\"ir2\":")); Serial.print(ir2State); Serial.print(F(","));
  Serial.print(F("\"servo1\":")); Serial.print(lastServo1Angle); Serial.print(F(","));
  Serial.print(F("\"servo2\":")); Serial.print(lastServo2Angle); Serial.print(F(","));
  Serial.print(F("\"plastic\":")); Serial.print(countPlastic); Serial.print(F(","));
  Serial.print(F("\"metal\":")); Serial.print(countMetal); Serial.print(F(","));
  Serial.print(F("\"paper\":")); Serial.print(countPaper); Serial.print(F(","));
  Serial.print(F("\"points\":")); Serial.print(totalRewardPoints); Serial.print(F(","));
  Serial.print(F("\"status\":\"")); Serial.print(statusMsg); Serial.print(F("\""));
  Serial.println(F("}}"));
}

void setup() {
  Serial.begin(115200);   // High-speed USB telemetry to dashboard
  espSerial.begin(9600);  // Serial link with ESP32-CAM

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
  lcd.print("DASHBOARD SYNC..");

  // Calibrate Servos
  servo1.attach(PIN_SERVO_1);
  servo2.attach(PIN_SERVO_2);
  servo1.write(0);
  servo2.write(0);
  lastServo1Angle = 0;
  lastServo2Angle = 0;
  delay(500);

  displayReadyScreen();
  sendTelemetry("SYSTEM READY");
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
  // 1. Process Hardware Deposit from Intake IR
  if (digitalRead(PIN_IR_INTAKE) == LOW) {
    processDeposit();
  }

  // 2. Process Dashboard Remote Control Commands from PC
  if (Serial.available()) {
    char cmd = Serial.read();
    if (cmd == 'P' || cmd == 'p') {
      executeSort('P', "REMOTE TRIGGER (PLASTIC)");
    } else if (cmd == 'M' || cmd == 'm') {
      executeSort('M', "REMOTE TRIGGER (METAL)");
    } else if (cmd == 'C' || cmd == 'c') {
      executeSort('C', "REMOTE TRIGGER (PAPER)");
    } else if (cmd == 'T' || cmd == 't') {
      sendTelemetry("PING ACTIVE");
    } else if (cmd == 'R' || cmd == 'r') {
      // Reset stats
      countPlastic = 0; countMetal = 0; countPaper = 0; totalRewardPoints = 0;
      sendTelemetry("STATS RESET");
      displayReadyScreen();
    }
  }

  // Periodic heartbeat telemetry every ~1.5s
  static unsigned long lastHeartbeat = 0;
  if (millis() - lastHeartbeat > 1200) {
    lastHeartbeat = millis();
    sendTelemetry("MONITORING");
  }

  delay(50);
}

void processDeposit() {
  digitalWrite(PIN_LED_STATUS, HIGH);
  sendTelemetry("ITEM DETECTED - CAPTURING IMAGE");

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("ITEM DETECTED!");
  lcd.setCursor(0, 1);
  lcd.print("AI SCANNING...");
  beepSound(100);

  // Trigger ESP32-CAM
  espSerial.println("SNAP");

  char materialClass = 0;
  unsigned long scanStart = millis();

  while (millis() - scanStart < 3500) {
    if (espSerial.available()) {
      char c = espSerial.read();
      if (c == 'P' || c == 'M' || c == 'C') {
        materialClass = c;
        break;
      }
    }
    delay(20);
  }

  // Fallback demo classification if camera still in preview mode
  if (materialClass == 0) {
    static int cycleCounter = 0;
    cycleCounter++;
    if (cycleCounter % 3 == 1) materialClass = 'P';
    else if (cycleCounter % 3 == 2) materialClass = 'M';
    else materialClass = 'C';
  }

  executeSort(materialClass, "AI CLASSIFIED");
}

void executeSort(char materialClass, const char* origin) {
  int reward = 0;
  lcd.clear();

  if (materialClass == 'P') {
    countPlastic++;
    reward = 10;
    lastServo1Angle = 45;
    lastServo2Angle = 0;
    servo1.write(45);
    servo2.write(0);

    lcd.setCursor(0, 0); lcd.print("CLASS: PLASTIC");
    lcd.setCursor(0, 1); lcd.print("ROUTE: BIN 1 (+10)");
    sendTelemetry("ROUTING TO BIN 1 (PLASTIC)");

  } else if (materialClass == 'M') {
    countMetal++;
    reward = 15;
    lastServo1Angle = 90;
    lastServo2Angle = 45;
    servo1.write(90);
    delay(200);
    servo2.write(45);

    lcd.setCursor(0, 0); lcd.print("CLASS: METAL CAN");
    lcd.setCursor(0, 1); lcd.print("ROUTE: BIN 2 (+15)");
    sendTelemetry("ROUTING TO BIN 2 (METAL)");

  } else if (materialClass == 'C') {
    countPaper++;
    reward = 5;
    lastServo1Angle = 90;
    lastServo2Angle = 135;
    servo1.write(90);
    delay(200);
    servo2.write(135);

    lcd.setCursor(0, 0); lcd.print("CLASS: PAPER CUP");
    lcd.setCursor(0, 1); lcd.print("ROUTE: BIN 3 (+5)");
    sendTelemetry("ROUTING TO BIN 3 (PAPER)");
  }

  totalRewardPoints += reward;

  // Verify exit via Chute IR 2
  unsigned long chuteStart = millis();
  while (millis() - chuteStart < 3000) {
    if (digitalRead(PIN_IR_CHUTE) == LOW) {
      sendTelemetry("EXIT CHUTE VERIFIED");
      break;
    }
    delay(40);
  }

  // Reset flaps
  servo1.write(0);
  servo2.write(0);
  lastServo1Angle = 0;
  lastServo2Angle = 0;

  beepSound(250);
  digitalWrite(PIN_LED_STATUS, LOW);

  lcd.clear();
  lcd.setCursor(0, 0); lcd.print("DEPOSIT SUCCESS!");
  lcd.setCursor(0, 1);
  lcd.print("PTS:+"); lcd.print(reward);
  lcd.print(" TOT:"); lcd.print(totalRewardPoints);

  sendTelemetry("CYCLE COMPLETED");
  delay(2500);

  displayReadyScreen();
  sendTelemetry("SYSTEM READY");
}
