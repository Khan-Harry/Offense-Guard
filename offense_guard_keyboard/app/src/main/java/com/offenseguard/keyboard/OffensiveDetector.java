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
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Hybrid Detection Engine:
 * 1. Instant 0ms local edge lexicon — English, Roman Urdu, Urdu Script.
 * 2. Background async verification with Offense Guard Flask Multi-Model AI Server.
 *
 * Urdu detection improvements:
 *  - Separate ROMAN_BLACKLIST (latin) vs URDU_BLACKLIST (Arabic script)
 *  - stripTashkeel() removes harakat so typed اعراب don't break matching
 *  - stripInvisible() removes zero-width joiners / BOM characters
 *  - Multi-word Urdu phrase substring search (URDU_PHRASES[])
 *  - "بہن چود" (space-separated) is caught via phrases, not word split
 */
public class OffensiveDetector {
    private static final String TAG = "OffensiveDetector";

    // Flask Backend API Endpoint (Public HTTPS Live Tunnel)
    public static String SERVER_URL = "https://offense-guard.vercel.app/api/runtime_check";

    // ─── Roman Urdu / English blacklist (matched after toLowerCase) ───────────
    // IMPORTANT: Every word must include ALL common phonetic/spelling variations
    // because Roman Urdu has no standard spelling — people write the same word
    // many different ways (e.g. kameena / kamina / kameene / kaminey etc.)
    private static final Set<String> ROMAN_BLACKLIST = new HashSet<>(Arrays.asList(

        // ── کمینہ (kameena / kamina) — all forms ──
        "kameena", "kameene", "kameeni", "kameeno",
        "kamina", "kamine", "kaminey", "kamini",
        "kmina", "kmeena",

        // ── حرامی (harami) ──
        "harami", "haramee", "haramy", "hrami", "harammi",
        "haramzada", "haramzaada", "haramzade", "haramzadey", "haramzadi",
        "haramkhor", "haramkhor",

        // ── چوتیہ (chutiya) ──
        "chutiya", "chootiya", "chutia", "chootia", "chotia",
        "chootiye", "chutiye", "chutiyas", "chutiyapa",
        "chutmarika", "choot",

        // ── گانڈو / گاند (gaandu / gaand) ──
        "gaandu", "gandu", "gandoo", "gaando",
        "gaand", "gaand", "gand",
        "gandoo", "gandou",

        // ── بہنچود (behenchod / bhenchod) ──
        "behenchod", "bhenchod", "behanchod", "benachod",
        "bhen chod", "bhenchod", "bhnchd",
        "bc", "b.c",

        // ── مادرچود (madarchod) ──
        "madarchod", "maderchod", "madarchaud", "madarchod",
        "madar chod", "mc",

        // ── لنڈ / لوڑا (land / lora) ──
        "lund", "lun", "land", "landa",
        "loray", "lorhay", "lora", "lawda", "landa",

        // ── کتا / کتیا (kutta / kutti) ──
        // NOTE: standalone "kutta/kuttay" are in CONTEXTUAL_ANIMALS — they are
        // only flagged here if they appear WITHOUT a zoological marker.
        "kutta", "kuttay", "kutte", "kutti", "kutia",
        "kuta", "kutey",

        // ── گدھا (gadha / gadhe) ──
        "gadha", "gadhe", "gadhi", "gadhey", "gadha",
        "gadday", "gadday",

        // ── رنڈی / گشتی (randi / gashti) ──
        "randi", "randy", "rande", "randwa", "randia",
        "gashti", "gashty", "gashte",

        // ── کنجر / دلال (kanjar / dalla) ──
        "kanjar", "kanjri", "kanjari", "kanjar",
        "dalla", "dalley", "dala",

        // ── بکواس (bakwas) ──
        "bakwas", "bakwaas", "bakawaas", "bakwasss",

        // ── پاگل (pagal) ──
        "pagal", "paagal", "pagla", "pagli",

        // ── گدھا / الو (ullo / gadhe) ──
        "ullo", "ullu", "ulu", "ulloo",
        "gadhe", "gadha",

        // ── بدتمیز (badtameez) ──
        "badtameez", "badtamiz", "bad tameez", "bad tamiz",

        // ── احمق / بیوقوف (ahmaq / bewaqoof) ──
        "ahmaq", "ahmak",
        "bewaqoof", "bewakoof", "bewakuf", "bewkoof",
        "bawakoof", "baywakoof",

        // ── جاہل (jahil) ──
        "jahil", "jaahil", "jaheel",

        // ── بھڑوا / چنال (bharwa / chinal) ──
        "bharwa", "bharway", "bhadwa", "bhadway",
        "chinal", "channl",

        // ── سور (suar) ──
        "suar", "suwar", "soar",

        // ── لعنت (lanat) ──
        "lanat", "laanat", "laannat",

        // ── گھٹیا (ghatiya) ──
        "ghatiya", "ghateya", "ghatiha",

        // ── سالا / سالی (sala / sali) ──
        "sala", "saala", "saale", "sale",
        "sali", "saali",

        // ── حرام خور (haraam khor) ──
        "haramkhor", "haram khor", "haramkhori",

        // ── گھنٹا (ghanta) ──
        "ghanta", "ghantaa", "gante",

        // ── کمبخت (kambakht) ──
        "kambakht", "kambakth", "kambakht",

        // ── نالائق (nalayiq) ──
        "nalayiq", "nalayak", "nalaik", "nalyak",

        // ── بدمعاش (badmaash) ──
        "badmaash", "badmash", "badmaas",

        // ── لفنگا (lafanga) ──
        "lafanga", "lafange", "lafangi",

        // ── بے غیرت (be-ghairat) ──
        "beghairat", "be ghairat", "beghayrat",
        "besharam", "be sharam",
        "behaya", "be haya",

        // ── بکوُاس abbreviations ──
        "bkl", "bsdk", "bund",

        // ── English swear words ──
        "fuck", "fucking", "fucked", "fucker", "fucks", "fck", "f*ck", "fu*k",
        "shit", "shitting", "shitty", "shits", "sh*t",
        "bitch", "bitches", "bitch", "b*tch",
        "bastard", "bastards",
        "asshole", "asshat", "ass",
        "cunt", "cunts",
        "dick", "dickhead", "dicks",
        "pussy", "pussies",
        "cock", "cocks",
        "whore", "whores",
        "slut", "sluts",
        "faggot", "fag",
        "idiot", "idiots",
        "stupid", "stupidity",
        "crap", "crappy",
        "dumbass", "dumb",
        "jackass", "jerk",
        "bullshit", "bs",
        "shithead", "motherfucker", "motherfucking",
        "fuckyou", "fuckoff", "fucku",
        "wtf", "stfu"
    ));


    // ─── Urdu Script blacklist (WITHOUT tashkeel — matched after normalizeUrdu) ─
    private static final Set<String> URDU_BLACKLIST = new HashSet<>(Arrays.asList(
        // جانوروں سے تشبیہ
        "کتا", "کتیا", "کتے", "کتی",
        "سور", "خنزیر",
        "گدھا", "گدھے", "گدھی",
        "الو",
        // جنسی گالیاں
        "حرامی", "حرامزادہ", "حرامکور",
        "کمینہ", "کمینے", "کمینی",
        "چوتیہ", "چوت", "چت",
        "گانڈو", "گانڈ", "گاند",
        "لنڈ", "لن", "لوڑا",
        "بھوسڑی", "بھوسڑا",
        "مادرچود", "بہنچود",
        "رنڈی", "رنڈوا",
        "گشتی", "چنال", "چنڈال",
        "دلا", "دلال", "کنجر", "کنجری", "کنجروں",
        "بھڑوا", "بھڑوے",
        "فاحشہ", "زانی", "زانیہ",
        "بدکردار", "بدکار",
        // توہین آمیز
        "بکواس", "لعنت",
        "خبیث", "ذلیل",
        "پاگل",
        "جاہل", "بدتمیز", "گھٹیا",
        "بدمعاش", "لفنگا", "لفنگے",
        "بے غیرت", "بے شرم", "بے حیا",
        "حرام خور",
        "سالا", "سالی",
        "نالائق", "احمق", "بیوقوف",
        "کمبخت", "بدبخت",
        "شیطان",
        "مردود", "ملعون", "لعین",
        "گنجا", "گدی",
        "چوڑا", "چوڑی",
        "ذلت"
    ));

    // ─── Multi-word Urdu & Roman Urdu phrases (substring search) ─────────
    private static final String[] URDU_PHRASES = {
        "سور کا بچہ", "الو کا پٹھا",
        "بہن چود", "بہنچود",
        "ماں چود", "ماں کی",
        "باپ کا", "ماں کا",
        "حرام زادہ", "حرام کا",
        "بے غیرت", "بے شرم", "بے حیا",
        "حرام خور",
        "کمینہ آدمی", "ذلیل آدمی", "نالائق آدمی",
        "کتے کی اولاد", "گدھے کی اولاد",
        // Roman Urdu possessive slurs & idioms (always offensive)
        "kutta k bachay", "kuttay k bachay", "kutta kay bachay", "kuttay kay bachay",
        "kutta ki aulad", "kutte ki aulad", "kuttay ki aulad",
        "gadha k bachay", "gadhe k bachay", "gadha ki aulad", "gadhe ki aulad",
        "ullu ka patha", "ullu ka pattha", "teri maa", "teri behan"
    };

    // ─── Zoological / Biological Factual Context Markers (Prevent False Positives) ───
    private static final List<String> ZOOLOGICAL_MARKERS = Arrays.asList(
        "جانور", "janwar", "وفادار", "wafadar", "پالتو", "palto", "paaltu",
        "ایک جانور", "ek janwar", "aik janwar", "animal", "pet", "pets",
        "جنگلی", "jungli", "چوپایہ", "دودھ", "breed", "nasl", "نسل", "چرند", "پرند", "بیمار"
    );

    // Contextual animal words that are safe in zoological narrative
    private static final Set<String> CONTEXTUAL_ANIMALS = new HashSet<>(Arrays.asList(
        "کتا", "کتیا", "کتے", "کتی", "سور", "خنزیر", "گدھا", "گدھے", "گدھی", "الو", "بندر",
        "kutta", "kuttay", "kutte", "kutti", "kutia", "kuta", "kutey",
        "gadha", "gadhe", "gadhi", "gadhey", "suar", "suwar", "ullo", "ullu", "ulu", "bandar"
    ));

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
     * Unicode ranges covered:
     *   U+0610–U+061A  Extended Arabic marks
     *   U+064B–U+065F  Tashkeel (fatha, kasra, damma, shadda, tanwin, etc.)
     *   U+0670         Superscript alef
     *   U+06D6–U+06DC  Extended Urdu marks
     *   U+06DF–U+06E4  More extended marks
     *   U+06E7–U+06E8  Rafe / wasla
     *   U+06EA–U+06ED  More extended marks
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
     * Zero-width non-joiner (U+200C), Zero-width joiner (U+200D),
     * Directional markers (U+200E–U+200F, U+202A–U+202E), BOM (U+FEFF).
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

        // Check if current text contains factual zoological markers (e.g. "ایک جانور ھے", "جانور ہے", "animal", etc.)
        boolean hasZoologicalContext = false;
        for (String zm : ZOOLOGICAL_MARKERS) {
            if (cleanText.contains(zm) || normalizedUrduText.contains(zm) || lowerText.contains(zm)) {
                hasZoologicalContext = true;
                break;
            }
        }

        // ── 1a. Roman / English check (case-insensitive) ──────────────────────
        String[] romanWords = lowerText.split("\\s+");
        for (String w : romanWords) {
            String sanitized = w.replaceAll("[.,/#!$%^&*;:{}=\\-_`~()?\"'۔،؟!]", "").trim();
            if (!sanitized.isEmpty() && ROMAN_BLACKLIST.contains(sanitized) && !isWordAllowed(sanitized)) {
                // If this is an animal word in factual zoological context, do not flag!
                if (hasZoologicalContext && CONTEXTUAL_ANIMALS.contains(sanitized)) {
                    continue;
                }
                callback.onResult(true, sanitized);
                return;
            }
        }

        // ── 1b. Urdu Script check (normalize → tashkeel-stripped comparison) ──
        // 1b-i: Multi-word phrase substring search (catches "بہن چود" with space, and Roman phrases like "kuttay k bachay")
        for (String phrase : URDU_PHRASES) {
            String normPhrase = normalizeUrdu(phrase).toLowerCase(Locale.ROOT);
            if ((normalizedUrduText.contains(normPhrase) || lowerText.contains(normPhrase)) && !isWordAllowed(normPhrase)) {
                callback.onResult(true, phrase);
                return;
            }
        }

        // 1b-ii: Individual Urdu word check
        // Split on whitespace, Urdu punctuation, and zero-width chars
        String[] urduWords = normalizedUrduText.split("[\\s\u200C\u200D،۔؟!.]+");
        for (String w : urduWords) {
            // Remove any remaining ASCII punctuation
            String sanitized = w.replaceAll("[.,/#!$%^&*;:{}=\\-_`~()?\"']", "").trim();
            if (sanitized.isEmpty()) continue;

            if (URDU_BLACKLIST.contains(sanitized) && !isWordAllowed(sanitized)) {
                // If this is an animal word in factual zoological context, do not flag!
                if (hasZoologicalContext && CONTEXTUAL_ANIMALS.contains(sanitized)) {
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
                // Network unavailable or offline — fail silently, do NOT emit false!
                Log.d(TAG, "Server check skipped/offline: " + e.getMessage());
            } finally {
                if (conn != null) conn.disconnect();
            }
        });
    }
}
