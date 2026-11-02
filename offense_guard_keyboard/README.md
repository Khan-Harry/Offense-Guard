# 🛡️ Offense Guard - Android System-Wide Smart Keyboard

This folder contains the **native Android System InputMethodService (Keyboard)** for **Offense Guard**.

When installed on any Android device, this keyboard registers directly inside the Android operating system settings (`Settings -> System -> Languages & Input -> Manage on-screen keyboards`). It can be used as the default keyboard inside **Instagram, WhatsApp, TikTok, YouTube, Facebook, Twitter, Chrome, and any other Android app**.

---

## ✨ Features

1. **System-Wide Protection:**
   - Runs in any Android application that opens a keyboard (Instagram comments, WhatsApp chats, TikTok video replies).
2. **Real-Time Smart Safety Bar:**
   - Sits on top of the keyboard layout.
   - 🟢 **Clean & Safe:** Displays green badge when safe text is typed.
   - 🔴 **Offensive Detected:** Real-time red warning alert showing the flagged term.
3. **Instant 0ms Edge Detection + Server Verification:**
   - Built-in high-priority Roman Urdu & English offline abusive lexicon for instant 0ms latency response.
   - Background asynchronous communication with the Offense Guard Flask server (`/api/runtime_check`).
4. **One-Tap In-Place Censor Action:**
   - When an offensive word is detected, pressing the **`[ 🛡️ Censor (***) ]`** button automatically replaces the flagged word with asterisks directly inside the target app's input box (e.g. inside an Instagram comment field).

---

## 🚀 How to Build & Install the APK

### Method 1: Android Studio (Easiest & Recommended for FYP)
1. Open **Android Studio**.
2. Click **File -> Open...** and select this directory:
   `d:\Semesters\BSE-6\FYP 2\FYP_Project\offense_guard_keyboard`
3. Wait for Gradle to sync dependencies.
4. Click **Build -> Build Bundle(s) / APK(s) -> Build APK(s)**.
5. Once build completes, click **locate** to get `app-debug.apk`.
6. Transfer `app-debug.apk` to your phone via USB or WhatsApp/Google Drive and install it.

### Method 2: Command Line (Gradle)
```bash
cd offense_guard_keyboard
./gradlew assembleDebug
```
The output APK will be located at:
`offense_guard_keyboard/app/build/outputs/apk/debug/app-debug.apk`

---

## 📱 How to Activate on Your Phone

1. Install and open the **Offense Guard Keyboard** app on your phone.
2. Tap **Step 1: Enable in Settings** -> Toggle on **Offense Guard Keyboard**.
3. Tap **Step 2: Select Active Keyboard** -> Choose **Offense Guard Keyboard**.
4. Open **Instagram / WhatsApp / TikTok** and tap any comment or chat box:
   - Type normal words: Status stays green 🟢
   - Type abusive/offensive words: Safety bar turns red 🔴 with alert!
   - Tap **"Censor (***)"**: The word is replaced by `***` before posting!
