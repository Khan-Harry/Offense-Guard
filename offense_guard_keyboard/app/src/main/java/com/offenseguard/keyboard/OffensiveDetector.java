package com.offenseguard.keyboard;

import android.os.Handler;
import android.os.Looper;
import android.util.Log;

import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.HashSet;
import java.util.Locale;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Hybrid Detection Engine:
 * 1. Instant 0ms local edge lexicon (via WordLists) — English, Roman Urdu, Urdu Script.
 * 2. Background async verification with Offense Guard Flask Multi-Model AI Server.
 */
public class OffensiveDetector {
    private static final String TAG = "OffensiveDetector";

    // Flask Backend API Endpoint (Public HTTPS Live Tunnel)
    public static String SERVER_URL = "https://offense-guard.vercel.app/api/runtime_check";

    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private final Handler mainHandler = new Handler(Looper.getMainLooper());
    private final Set<String> oneTimeAllowedWords = new HashSet<>();
    private final Set<String> permanentAllowedWords = new HashSet<>();

    public interface DetectionCallback {
        void onResult(boolean isOffensive, String flaggedWord);
    }

    // ─── Text normalization helpers ───────────────────────────────────────────

    /**
     * Strips Arabic/Urdu tashkeel (harakat / diacritical marks) so that
     * "کَمِینَہ" matches "کمینہ" in the blacklist.
     */
    private static String stripTashkeel(String text) {
        if (text == null) return "";
        return text.replaceAll(
            "[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]",
            ""
        );
    }

    /**
     * Strips invisible Unicode characters that can break exact-string matching:
     * Zero-width non-joiner, Zero-width joiner, directional markers, BOM.
     */
    private static String stripInvisible(String text) {
        if (text == null) return "";
        return text.replaceAll("[\u200B-\u200F\u202A-\u202E\uFEFF]", "");
    }

    /** Full Urdu normalization: strip invisible chars then strip tashkeel. */
    private static String normalizeUrdu(String text) {
        return stripTashkeel(stripInvisible(text)).trim();
    }

    // ─── Allow-list management ────────────────────────────────────────────────

    /** Allows a word for THIS session/message only (one-time). */
    public void allowWordOnce(String word) {
        if (word != null) {
            oneTimeAllowedWords.add(normalizeUrdu(word).toLowerCase(Locale.ROOT));
        }
    }

    /** Clears one-time allowances (call on new input session). */
    public void clearOneTimeAllowed() {
        oneTimeAllowedWords.clear();
    }

    /** Permanently whitelists a word (persists via SharedPreferences). */
    public void addPermanentAllowedWord(String word) {
        if (word != null) {
            permanentAllowedWords.add(normalizeUrdu(word).toLowerCase(Locale.ROOT));
        }
    }

    /** Restores permanent whitelist from SharedPreferences on startup. */
    public void setPermanentAllowedWords(Set<String> words) {
        permanentAllowedWords.clear();
        if (words != null) {
            for (String w : words) {
                if (w != null && !w.trim().isEmpty()) {
                    permanentAllowedWords.add(normalizeUrdu(w).toLowerCase(Locale.ROOT));
                }
            }
        }
    }

    public boolean isWordAllowed(String word) {
        if (word == null) return false;
        String norm = normalizeUrdu(word).toLowerCase(Locale.ROOT);
        return oneTimeAllowedWords.contains(norm) || permanentAllowedWords.contains(norm);
    }

    @Deprecated
    public void ignoreWord(String word) {
        allowWordOnce(word);
    }

    // ─── Main detection ───────────────────────────────────────────────────────

    /**
     * Checks text for offensive language using hybrid local + server logic.
     * Calls callback on main thread.
     */
    public void analyzeText(final String text, final DetectionCallback callback) {
        if (text == null || text.trim().isEmpty()) {
            callback.onResult(false, "");
            return;
        }

        final String cleanText = stripInvisible(text).trim();
        final String lowerText = cleanText.toLowerCase(Locale.ROOT);
        final String normalizedUrduText = normalizeUrdu(cleanText);

        // Check if current text contains factual zoological markers
        boolean hasZoologicalContext = false;
        for (String zm : WordLists.ZOOLOGICAL_MARKERS) {
            if (cleanText.contains(zm) || normalizedUrduText.contains(zm) || lowerText.contains(zm)) {
                hasZoologicalContext = true;
                break;
            }
        }

        // ── 1a. Roman / English check (case-insensitive) ──────────────────────
        String[] romanWords = lowerText.split("\\s+");
        for (String w : romanWords) {
            String sanitized = w.replaceAll("[.,/#!$%^&*;:{}=\\-_`~()?\"'۔،؟!]", "").trim();
            if (!sanitized.isEmpty() && WordLists.ROMAN_BLACKLIST.contains(sanitized) && !isWordAllowed(sanitized)) {
                if (hasZoologicalContext && WordLists.CONTEXTUAL_ANIMALS.contains(sanitized)) {
                    continue;
                }
                callback.onResult(true, sanitized);
                return;
            }
        }

        // ── 1b. Urdu Script check ──
        // 1b-i: Multi-word phrase substring search
        for (String phrase : WordLists.URDU_PHRASES) {
            String normPhrase = normalizeUrdu(phrase).toLowerCase(Locale.ROOT);
            if ((normalizedUrduText.contains(normPhrase) || lowerText.contains(normPhrase)) && !isWordAllowed(normPhrase)) {
                callback.onResult(true, phrase);
                return;
            }
        }

        // 1b-ii: Individual Urdu word check
        String[] urduWords = normalizedUrduText.split("[\\s\u200C\u200D،۔؟!.]+");
        for (String w : urduWords) {
            String sanitized = w.replaceAll("[.,/#!$%^&*;:{}=\\-_`~()?\"']", "").trim();
            if (sanitized.isEmpty()) continue;

            if (WordLists.URDU_BLACKLIST.contains(sanitized) && !isWordAllowed(sanitized)) {
                if (hasZoologicalContext && WordLists.CONTEXTUAL_ANIMALS.contains(sanitized)) {
                    continue;
                }
                callback.onResult(true, sanitized);
                return;
            }
        }

        // If text is very short, skip server call
        if (cleanText.length() < 3) {
            callback.onResult(false, "");
            return;
        }

        // ── 2. Async Server AI Runtime Check ─────────────────────────────────
        executor.execute(() -> {
            HttpURLConnection conn = null;
            try {
                URL url = new URL(SERVER_URL);
                conn = (HttpURLConnection) url.openConnection();
                conn.setRequestMethod("POST");
                conn.setRequestProperty("Content-Type", "application/json; utf-8");
                conn.setRequestProperty("Accept", "application/json");
                conn.setConnectTimeout(1500);
                conn.setReadTimeout(1500);
                conn.setDoOutput(true);

                JSONObject jsonInput = new JSONObject();
                jsonInput.put("text", cleanText);

                try (OutputStream os = conn.getOutputStream()) {
                    byte[] input = jsonInput.toString().getBytes(StandardCharsets.UTF_8);
                    os.write(input, 0, input.length);
                }

                int code = conn.getResponseCode();
                if (code == 200) {
                    BufferedReader br = new BufferedReader(
                        new InputStreamReader(conn.getInputStream(), StandardCharsets.UTF_8));
                    StringBuilder response = new StringBuilder();
                    String line;
                    while ((line = br.readLine()) != null) {
                        response.append(line.trim());
                    }

                    JSONObject resObj = new JSONObject(response.toString());
                    boolean isOffensive = resObj.optBoolean("is_offensive", false);
                    String flaggedWord = resObj.optString("flagged_word", "");

                    if (isWordAllowed(flaggedWord)) {
                        isOffensive = false;
                        flaggedWord = "";
                    }

                    final boolean finalIsOffensive = isOffensive;
                    final String finalFlaggedWord = flaggedWord;
                    mainHandler.post(() -> callback.onResult(finalIsOffensive, finalFlaggedWord));
                }
            } catch (Exception e) {
                Log.d(TAG, "Server check skipped/offline: " + e.getMessage());
            } finally {
                if (conn != null) conn.disconnect();
            }
        });
    }
}
