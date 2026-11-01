void setup() {
  Serial.begin(9600);
  Serial.println(F("=== DIAGNOSTIC TEST: LED PIN 13 & BUZZER PIN 8 ==="));
  pinMode(13, OUTPUT);
  pinMode(8, OUTPUT);
}

void loop() {
  Serial.println(F("TURN ON: LED (13) + BUZZER (8)"));
  digitalWrite(13, HIGH);
  digitalWrite(8, HIGH);
  tone(8, 2000);
  delay(1000);

  Serial.println(F("TURN OFF: LED (13) + BUZZER (8)"));
  digitalWrite(13, LOW);
  digitalWrite(8, LOW);
  noTone(8);
  delay(1000);
}
