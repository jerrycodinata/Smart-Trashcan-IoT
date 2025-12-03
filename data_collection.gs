// --- SCRIPT CONFIGURATION ---
const SHEET_ID = "GS Sheet ID";
const SHEET_NAME = "DataLog";

/**
 * Handles GET requests from the ESP32 to log sensor data.
 */
function doGet(e) {
  if (!e.parameter) {
    return ContentService.createTextOutput(
      "Error: No parameters provided."
    ).setMimeType(ContentService.MimeType.TEXT);
  }

  try {
    const sheet = SpreadsheetApp.openById(SHEET_ID).getSheetByName(SHEET_NAME);

    if (!sheet) {
      SpreadsheetApp.openById(SHEET_ID).insertSheet(SHEET_NAME);
    }

    if (sheet.getLastRow() === 0) {
      const headers = [
        "Timestamp (ISO 8601)",
        "Distance (cm)",
        "Fullness (%)",
        "Lid Status",
        "CPU Temp (°C)",
      ];
      sheet.appendRow(headers);
    }

    const timestamp = e.parameter.timestamp
      ? e.parameter.timestamp
      : new Date().toISOString();

    const distance = e.parameter.distance
      ? parseFloat(e.parameter.distance)
      : null;
    const fullness = e.parameter.fullness
      ? parseFloat(e.parameter.fullness)
      : null;
    const lidStatus = e.parameter.lid_status ? e.parameter.lid_status : null;
    const cpuTemp = e.parameter.cpu_temp
      ? parseFloat(e.parameter.cpu_temp)
      : null;

    sheet.appendRow([timestamp, distance, fullness, lidStatus, cpuTemp]);

    return ContentService.createTextOutput("Success: Row added.").setMimeType(
      ContentService.MimeType.TEXT
    );
  } catch (error) {
    Logger.log(error.toString());
    return ContentService.createTextOutput(
      "Error: " + error.toString()
    ).setMimeType(ContentService.MimeType.TEXT);
  }
}
