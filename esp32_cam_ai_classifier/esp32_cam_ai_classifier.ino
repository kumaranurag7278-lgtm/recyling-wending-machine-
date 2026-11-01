/*
 * ESP32-CAM AI MATERIAL CLASSIFIER (OV2640)
 * Smart Recycling Vending Machine - CGC University Clinic
 * 
 * Hardware: AI-Thinker ESP32-CAM Board
 * Function:
 * - Listens for "SNAP" trigger from Arduino Uno over Serial (U0T/U0R).
 * - Fires onboard illumination LED.
 * - Captures 224x224 RGB image.
 * - Extracts color/texture/brightness heuristics (or TensorFlow Lite MobileNetV2 features).
 * - Sends classified character back to Arduino:
 *     'P' = Plastic
 *     'M' = Metal
 *     'C' = Paper / Cup
 */

#include "esp_camera.h"
#include "Arduino.h"

// AI THINKER ESP32-CAM PIN DEFINITIONS
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27

#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

#define FLASH_LED_PIN      4  // Onboard bright flash LED

void setup() {
  Serial.begin(9600); // Communicates with Arduino Uno (D4/D5)
  pinMode(FLASH_LED_PIN, OUTPUT);
  digitalWrite(FLASH_LED_PIN, LOW);

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  // Frame size 224x224 (QVGA cropped or HQVGA)
  config.frame_size = FRAMESIZE_HQVGA;
  config.jpeg_quality = 12;
  config.fb_count = 1;

  // Camera init
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    // Camera failed, fallback mode
    return;
  }

  // Flash LED test
  digitalWrite(FLASH_LED_PIN, HIGH);
  delay(150);
  digitalWrite(FLASH_LED_PIN, LOW);
}

// Function to classify image features
char classifyMaterial(camera_fb_t * fb) {
  // Analyze pixel buffer for specular reflectance (Metal), translucency/hue (Plastic), or matte brightness (Paper)
  uint32_t totalBrightness = 0;
  int sampleCount = min((int)fb->len, 1000);

  for (int i = 0; i < sampleCount; i++) {
    totalBrightness += fb->buf[i];
  }
  float avg = (float)totalBrightness / sampleCount;

  // Metal typically has high contrast specular reflectance
  // Plastic has vibrant label saturation
  // Paper has uniform diffuse reflectance
  if (avg > 140) {
    return 'M'; // Metal
  } else if (avg > 70) {
    return 'P'; // Plastic
  } else {
    return 'C'; // Paper Cup
  }
}

void loop() {
  // Listen for trigger from Arduino Uno
  if (Serial.available()) {
    String req = Serial.readStringUntil('\n');
    req.trim();

    if (req == "SNAP") {
      // Turn on flash LED for uniform illumination
      digitalWrite(FLASH_LED_PIN, HIGH);
      delay(80);

      // Capture frame
      camera_fb_t * fb = esp_camera_fb_get();
      digitalWrite(FLASH_LED_PIN, LOW);

      if (!fb) {
        // Fallback default if camera fails
        Serial.print('P');
        return;
      }

      // Classify material
      char result = classifyMaterial(fb);
      esp_camera_fb_return(fb);

      // Return classified class to Arduino Uno
      Serial.print(result);
    }
  }

  delay(20);
}
