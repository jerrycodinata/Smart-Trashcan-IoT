#define BLYNK_TEMPLATE_ID           "TEMPLATE ID"
#define BLYNK_TEMPLATE_NAME         "TEMPLATE NAME"
#define BLYNK_AUTH_TOKEN            "AUTH TOKEN"
#define BLYNK_PRINT Serial

#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <BlynkSimpleEsp32.h>

#define TRIG_PIN 22
#define ECHO_PIN 21
#define PROXIMITY_PIN 23
#define LED_PIN 2
#define TRASH_CAN_DEPTH_CM 17.0

// WiFi Credentials
char ssid[] = "Wifi SSID";  
char pass[] = "Wifi Password";

// Google App Script Variables
const char* GAS_HOST = "script.google.com"; 
const int HTTPS_PORT = 443; 

const char* GAS_URL_PREFIX = "GAS URL PREFIX";

// Global Variables
int latestDistance = 0;
int latestLidState = 0;
float latestCpuTemp = 0.0;
float latestUptimeHour = 0.0;
int latestFullnessPercent = 0;
int previousLidState = 0;
int googleSheetTimerId;

BlynkTimer timer;
WiFiClientSecure client;

// --- SENSOR READ FUNCTIONS --
void readUltrasonicSensor() {
  if (latestLidState == HIGH) {
    return; 
  }

  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  long duration = pulseIn(ECHO_PIN, HIGH, 25000);
  latestDistance = duration * 0.0343 / 2;

  // --- Calculate Fullness Percentage ---
  if (latestDistance <= 0 || latestDistance > TRASH_CAN_DEPTH_CM) {
    latestFullnessPercent = 0;
  } else {
    latestFullnessPercent = 100.0 * (TRASH_CAN_DEPTH_CM - latestDistance) / TRASH_CAN_DEPTH_CM;
  }

  if (latestFullnessPercent < 0) latestFullnessPercent = 0;
  if (latestFullnessPercent > 100) latestFullnessPercent = 100;
}

void readProximitySensor() {
  latestLidState = digitalRead(PROXIMITY_PIN);
}

void readCpuTemp() {
  latestCpuTemp = temperatureRead();
}

void readUpTime() {
  latestUptimeHour = millis() / 3600000.0;
}

void readAllSensorData() {
  readProximitySensor();
  readUltrasonicSensor();
  readCpuTemp();
}

void printDataToSerial() {
  Serial.println("===========================================================");
  Serial.println("Distance: " + String(latestDistance) + "cm");
  Serial.println("Full: " + String(latestFullnessPercent) + "%");
  Serial.println("Lid: " + String((latestLidState == LOW) ? "CLOSED" : "OPEN"));
  Serial.println("Temp: " + String(latestCpuTemp, 1) + "C");
  Serial.println("Uptime: " + String(latestUptimeHour, 2) + "h");
  Serial.println("===========================================================");
  Serial.println();

}

void sendDataToBlynk() {
  Blynk.virtualWrite(V7, latestDistance);
  Blynk.virtualWrite(V8, latestLidState);
  Blynk.virtualWrite(V5, latestCpuTemp);
  Blynk.virtualWrite(V11, latestUptimeHour);
  Blynk.virtualWrite(V12, latestFullnessPercent);
}

void sendDataToGoogleSheet() {
  digitalWrite(LED_PIN, HIGH);

  client.setInsecure();

  if (!client.connect(GAS_HOST, HTTPS_PORT)) {
    Serial.println("GAS connection failed");
    digitalWrite(LED_PIN, LOW);
    return;
  }

  String lidStatusText = (latestLidState == LOW) ? "CLOSED" : "OPEN";

  String url = String(GAS_URL_PREFIX) + "?distance=" + String(latestDistance) +
               "&fullness=" + String(latestFullnessPercent) +
               "&lid_status=" + lidStatusText +
               "&cpu_temp=" + String(latestCpuTemp, 2);

  client.print(String("GET ") + url + " HTTP/1.1\r\n" +
               "Host: " + GAS_HOST + "\r\n" +
               "User-Agent: ESP32_TrashCan_Sensor\r\n" +
               "Connection: close\r\n\r\n");

  Serial.println("--> Sent data to Google Sheet");
  
  client.stop();
  delay(100);
  digitalWrite(LED_PIN, LOW);
}

void checkLidStateChange() {
  int currentLidState = digitalRead(PROXIMITY_PIN);

  if (currentLidState != previousLidState) {
    Serial.println("!!! Lid state change detected. Logging event now. !!!");

    latestLidState = currentLidState;
    
    readUltrasonicSensor();
    readCpuTemp();
    
    sendDataToGoogleSheet();
    timer.restartTimer(googleSheetTimerId);
    
    previousLidState = currentLidState;
  }
}

void setup() {
  Serial.begin(115200);
  Serial.println("Starting up...");

  Blynk.begin(BLYNK_AUTH_TOKEN, ssid, pass);

  if (Blynk.connected()) {
    Serial.println("Successfully connected to Blynk!");
  } else {
    Serial.println("Failed to connect to Blynk. Will continue without it.");
  }

  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  pinMode(PROXIMITY_PIN, INPUT);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  previousLidState = digitalRead(PROXIMITY_PIN);

  // --- TIMER SETUP ---
  timer.setInterval(1000L, readAllSensorData);
  timer.setInterval(30000L, readUpTime);
  timer.setInterval(2000L, sendDataToBlynk);    
  timer.setInterval(2100L, printDataToSerial);   

  googleSheetTimerId = timer.setInterval(15000L, sendDataToGoogleSheet);
}

void loop() {
  Blynk.run();
  timer.run();
  checkLidStateChange();
}
