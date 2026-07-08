// Hide the console window in release builds.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use base64::Engine;

/// Writes base64-encoded bytes to a user-chosen path. The path always comes from
/// the native save dialog, so the user has explicitly picked the destination.
#[tauri::command]
fn write_file(path: String, data_base64: String) -> Result<(), String> {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(data_base64)
        .map_err(|e| e.to_string())?;
    std::fs::write(&path, bytes).map_err(|e| e.to_string())
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![write_file])
        .run(tauri::generate_context!())
        .expect("error while running Noise Contour Mapper");
}
