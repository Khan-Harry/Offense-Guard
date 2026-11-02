package com.offenseguard.keyboard;

import android.content.SharedPreferences;
import android.graphics.Color;
import android.inputmethodservice.InputMethodService;
import android.os.Handler;
import android.os.Looper;
import android.view.KeyEvent;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.View;
import android.view.inputmethod.EditorInfo;
import android.view.inputmethod.InputConnection;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

import org.json.JSONArray;
import org.json.JSONException;

public class OffenseGuardInputMethodService extends InputMethodService {

    private static final String PREFS_NAME = "offense_guard_prefs";
    private static final String PREFS_KEY_ALLOWED = "always_allowed_words";

    private LinearLayout safetyBar;
    private TextView tvSafetyStatus;

    // In-Keyboard ReThink Warning Popup Card
    private LinearLayout cardReThinkWarning;
    private TextView tvFlaggedWordCard;
    private Button btnCardClear;      // 🗑️ Delete Word
    private Button btnCardUseAnyway;  // ⚡ Use Anyway (one-time)
    private Button btnCardAllow;      // ⭐ Always Allow (permanent)
    private Button btnDialogDismiss;

    private LinearLayout layoutAlpha;
    private LinearLayout layoutNumbers;
    private Button btnModeSwitch;
    private Button btnGlobe;
    private Button btnSpace;
    private Button btnPeriod;

    private boolean isShift = false;
    private boolean isNumbersMode = false;
    private boolean isUrduMode = false; // Urdu / English toggle
    private String currentFlaggedWord = "";

    private View keyboardRootView;
    private final OffensiveDetector detector = new OffensiveDetector();
    private final List<Button> letterButtons = new ArrayList<>();
    private final Handler debounceHandler = new Handler(Looper.getMainLooper());
    private Runnable debounceRunnable;

    // Continuous backspace repeat handler and runnable
    private final Handler repeatHandler = new Handler(Looper.getMainLooper());
    private final Runnable repeatRunnable = new Runnable() {
        @Override
        public void run() {
            handleBackspace();
            repeatHandler.postDelayed(this, 50); // Delete 1 character every 50ms while held
        }
    };

    @Override
    public View onCreateInputView() {
        LayoutInflater inflater = getLayoutInflater();
        keyboardRootView = inflater.inflate(R.layout.keyboard_view, null);

        // Bind Safety Bar
        safetyBar = keyboardRootView.findViewById(R.id.safetyBar);
        tvSafetyStatus = keyboardRootView.findViewById(R.id.tvSafetyStatus);

        // Bind In-Keyboard ReThink Warning Card
        cardReThinkWarning = keyboardRootView.findViewById(R.id.cardReThinkWarning);
        tvFlaggedWordCard = keyboardRootView.findViewById(R.id.tvFlaggedWordCard);
        btnCardClear = keyboardRootView.findViewById(R.id.btnCardClear);
        btnCardUseAnyway = keyboardRootView.findViewById(R.id.btnCardUseAnyway);
        btnCardAllow = keyboardRootView.findViewById(R.id.btnCardAllow);
        btnDialogDismiss = keyboardRootView.findViewById(R.id.btnDialogDismiss);

        // Load persistent always-allowed whitelist into detector
        loadPermanentAllowedWords();

        // Bind Layout containers
        layoutAlpha = keyboardRootView.findViewById(R.id.layoutAlpha);
        layoutNumbers = keyboardRootView.findViewById(R.id.layoutNumbers);
        btnModeSwitch = keyboardRootView.findViewById(R.id.btnModeSwitch);
        btnGlobe = keyboardRootView.findViewById(R.id.btnGlobe);
        btnSpace = keyboardRootView.findViewById(R.id.btnSpace);
        btnPeriod = keyboardRootView.findViewById(R.id.btnPeriod);
        Button btnEnter = keyboardRootView.findViewById(R.id.btnEnter);

        // Setup In-Keyboard Popup Card Listeners
        btnCardClear.setOnClickListener(v -> handleClearWord());
        btnCardUseAnyway.setOnClickListener(v -> handleUseAnyway());
        btnCardAllow.setOnClickListener(v -> handleAlwaysAllow());
        btnDialogDismiss.setOnClickListener(v -> {
            if (cardReThinkWarning != null) cardReThinkWarning.setVisibility(View.GONE);
        });

        // Tapping safety bar opens/closes popup warning card
        safetyBar.setOnClickListener(v -> {
            if (currentFlaggedWord != null && !currentFlaggedWord.isEmpty()) {
                if (cardReThinkWarning != null) {
                    int vis = cardReThinkWarning.getVisibility() == View.VISIBLE ? View.GONE : View.VISIBLE;
                    cardReThinkWarning.setVisibility(vis);
                }
            }
        });

        // Setup Alphabet Layout Rows
        setupAlphaRows(keyboardRootView);

        // Setup Numbers Layout Rows
        setupNumberRows(keyboardRootView);

        // Setup Bottom Control Keys
        btnModeSwitch.setOnClickListener(v -> toggleKeyboardMode());

        // Setup 🌐 Globe Language Switcher (English <-> Urdu)
        btnGlobe.setOnClickListener(v -> toggleLanguage());

        btnSpace.setOnClickListener(v -> {
            commitText(" ");
            checkTextRealtime();
        });

        btnSpace.setOnLongClickListener(v -> {
            toggleLanguage();
            return true;
        });

        btnPeriod.setOnClickListener(v -> {
            commitText(isUrduMode ? "۔" : ".");
            checkTextRealtime();
        });

        btnEnter.setOnClickListener(v -> {
            InputConnection ic = getCurrentInputConnection();
            if (ic != null) {
                EditorInfo info = getCurrentInputEditorInfo();
                if (info != null && (info.imeOptions & EditorInfo.IME_MASK_ACTION) != EditorInfo.IME_ACTION_NONE) {
                    ic.performEditorAction(info.imeOptions & EditorInfo.IME_MASK_ACTION);
                } else {
                    ic.sendKeyEvent(new KeyEvent(KeyEvent.ACTION_DOWN, KeyEvent.KEYCODE_ENTER));
                    ic.sendKeyEvent(new KeyEvent(KeyEvent.ACTION_UP, KeyEvent.KEYCODE_ENTER));
                }
            }
        });

        return keyboardRootView;
    }

    private void setupAlphaRows(View root) {
        letterButtons.clear();
        LinearLayout row1 = root.findViewById(R.id.row1);
        LinearLayout row2 = root.findViewById(R.id.row2);
        LinearLayout row3 = root.findViewById(R.id.row3);

        row1.removeAllViews();
        row2.removeAllViews();
        row3.removeAllViews();

        String[] r1;
        String[] r2;
        String[] r3;

        if (isUrduMode) {
            // Urdu Phonetic Layout (اردو کی بورڈ)
            if (isShift) {
                r1 = new String[]{"ض", "ص", "ث", "ف", "غ", "خ", "ح", "ڈ", "ٹ", "ژ"};
                r2 = new String[]{"آ", "ذ", "ڑ", "ظ", "ط", "ث", "گ", "ک", "ھ"};
                r3 = new String[]{"ں", "ؤ", "ئ", "ء", "؟", "،", "۔"};
            } else {
                r1 = new String[]{"ق", "و", "ع", "ر", "ت", "ے", "ء", "ی", "ہ", "پ"};
                r2 = new String[]{"ا", "س", "د", "ف", "گ", "ح", "ج", "ک", "ل"};
                r3 = new String[]{"ز", "ش", "چ", "ط", "ب", "ن", "م"};
            }
        } else {
            // Standard English QWERTY
            r1 = new String[]{"q", "w", "e", "r", "t", "y", "u", "i", "o", "p"};
            r2 = new String[]{"a", "s", "d", "f", "g", "h", "j", "k", "l"};
            r3 = new String[]{"z", "x", "c", "v", "b", "n", "m"};
        }

        for (String c : r1) {
            Button b = createKeyButton(c, 1.0f);
            row1.addView(b);
            letterButtons.add(b);
        }

        for (String c : r2) {
            Button b = createKeyButton(c, 1.0f);
            row2.addView(b);
            letterButtons.add(b);
        }

        // Shift button on left of Row 3
        Button btnShift = createSpecialButton(isShift ? "⬆" : "⇧", 1.25f);
        btnShift.setOnClickListener(v -> toggleShift());
        row3.addView(btnShift);

        for (String c : r3) {
            Button b = createKeyButton(c, 1.0f);
            row3.addView(b);
            letterButtons.add(b);
        }

        // Backspace button on right of Row 3 with continuous repeating delete on long-press
        Button btnBack = createSpecialButton("⌫", 1.25f);
        setupBackspaceButton(btnBack);
        row3.addView(btnBack);
    }

    private void setupNumberRows(View root) {
        LinearLayout row1 = root.findViewById(R.id.numRow1);
        LinearLayout row2 = root.findViewById(R.id.numRow2);
        LinearLayout row3 = root.findViewById(R.id.numRow3);

        row1.removeAllViews();
        row2.removeAllViews();
        row3.removeAllViews();

        String[] r1 = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "0"};
        String[] r2 = {"@", "#", "$", "%", "&", "*", "-", "+", "=", "/"};
        String[] r3 = isUrduMode 
            ? new String[]{"؟", "،", "۔", "!", ":", ";", "(", ")"}
            : new String[]{"(", ")", "!", "?", ":", ";", "\"", "'"};

        for (String c : r1) row1.addView(createKeyButton(c, 1.0f));
        for (String c : r2) row2.addView(createKeyButton(c, 1.0f));

        for (String c : r3) row3.addView(createKeyButton(c, 1.0f));

        // Backspace button on right of Row 3 with continuous repeating delete
        Button btnBack = createSpecialButton("⌫", 1.25f);
        setupBackspaceButton(btnBack);
        row3.addView(btnBack);
    }

    /**
     * Attaches continuous repeat deletion listener to the backspace button.
     * Single tap = 1 character deleted.
     * Long press = continuous deletion every 50ms after initial 300ms delay.
     */
    private void setupBackspaceButton(Button btnBack) {
        btnBack.setOnTouchListener((v, event) -> {
            switch (event.getAction()) {
                case MotionEvent.ACTION_DOWN:
                    handleBackspace(); // delete first char immediately
                    repeatHandler.removeCallbacks(repeatRunnable);
                    repeatHandler.postDelayed(repeatRunnable, 300); // start fast repeat after 300ms
                    v.setPressed(true);
                    return true;
                case MotionEvent.ACTION_UP:
                case MotionEvent.ACTION_CANCEL:
                    repeatHandler.removeCallbacks(repeatRunnable); // stop repeating
                    v.setPressed(false);
                    return true;
            }
            return false;
        });
    }

    private Button createKeyButton(final String text, float weight) {
        Button b = new Button(this);
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.MATCH_PARENT, weight);
        params.setMargins(2, 0, 2, 0);
        b.setLayoutParams(params);
        b.setText(text);
        b.setTextSize(isUrduMode ? 21 : 18);
        b.setTextColor(Color.parseColor("#F1F3F4"));
        b.setBackgroundResource(R.drawable.key_background);
        b.setPadding(0, 0, 0, 0);
        b.setAllCaps(false);
        b.setMinWidth(0);
        b.setMinimumWidth(0);
        b.setMinHeight(0);
        b.setMinimumHeight(0);
        b.setStateListAnimator(null);
        b.setElevation(2f);

        b.setOnClickListener(v -> {
            String charToCommit = b.getText().toString();
            commitText(charToCommit);
            if (isShift) {
                toggleShift();
            }
            checkTextRealtime();
        });

        return b;
    }

    private Button createSpecialButton(String text, float weight) {
        Button b = new Button(this);
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.MATCH_PARENT, weight);
        params.setMargins(2, 0, 2, 0);
        b.setLayoutParams(params);
        b.setText(text);
        b.setTextSize(17);
        b.setTextColor(Color.parseColor("#E8EAED"));
        b.setBackgroundResource(R.drawable.key_special_background);
        b.setPadding(0, 0, 0, 0);
        b.setMinWidth(0);
        b.setMinimumWidth(0);
        b.setMinHeight(0);
        b.setMinimumHeight(0);
        b.setStateListAnimator(null);
        b.setElevation(1f);
        return b;
    }

    private void commitText(String text) {
        InputConnection ic = getCurrentInputConnection();
        if (ic != null) {
            ic.commitText(text, 1);
        }
    }

    private void handleBackspace() {
        InputConnection ic = getCurrentInputConnection();
        if (ic != null) {
            ic.deleteSurroundingText(1, 0);
            checkTextRealtime();
        }
    }

    private void toggleShift() {
        isShift = !isShift;
        if (isUrduMode) {
            setupAlphaRows(keyboardRootView);
        } else {
            for (Button b : letterButtons) {
                String txt = b.getText().toString();
                b.setText(isShift ? txt.toUpperCase(Locale.ROOT) : txt.toLowerCase(Locale.ROOT));
            }
        }
    }

    private void toggleLanguage() {
        isUrduMode = !isUrduMode;
        isShift = false;
        if (btnSpace != null) {
            btnSpace.setText(isUrduMode ? "اردو" : "English");
        }
        if (btnGlobe != null) {
            btnGlobe.setText(isUrduMode ? "EN" : "🌐");
        }
        if (btnPeriod != null) {
            btnPeriod.setText(isUrduMode ? "۔" : ".");
        }
        if (keyboardRootView != null) {
            setupAlphaRows(keyboardRootView);
            setupNumberRows(keyboardRootView);
        }
    }

    private void toggleKeyboardMode() {
        isNumbersMode = !isNumbersMode;
        if (isNumbersMode) {
            layoutAlpha.setVisibility(View.GONE);
            layoutNumbers.setVisibility(View.VISIBLE);
            btnModeSwitch.setText("ABC");
        } else {
            layoutAlpha.setVisibility(View.VISIBLE);
            layoutNumbers.setVisibility(View.GONE);
            btnModeSwitch.setText("123");
        }
    }

    private void checkTextRealtime() {
        InputConnection ic = getCurrentInputConnection();
        if (ic == null) return;

        CharSequence textBefore = ic.getTextBeforeCursor(150, 0);
        if (textBefore == null || textBefore.length() == 0) {
            setSafetyStatus(false, "");
            return;
        }

        final String sentence = textBefore.toString();

        if (debounceRunnable != null) {
            debounceHandler.removeCallbacks(debounceRunnable);
        }

        debounceRunnable = () -> detector.analyzeText(sentence, this::setSafetyStatus);
        debounceHandler.postDelayed(debounceRunnable, 200);
    }

    /**
     * Updates the safety bar and warning card.
     * Prevents premature dismissal: Warning card will NOT close automatically if the flagged
     * word is still present in the input text!
     */
    private void setSafetyStatus(boolean isOffensive, String flaggedWord) {
        if (tvSafetyStatus == null || safetyBar == null) return;

        if (isOffensive && flaggedWord != null && !flaggedWord.isEmpty()) {
            currentFlaggedWord = flaggedWord;
            safetyBar.setBackgroundColor(Color.parseColor("#3D1B1E"));
            tvSafetyStatus.setText("⚠️ Flagged: \"" + flaggedWord + "\" (Tap to review)");
            tvSafetyStatus.setTextColor(Color.parseColor("#FF8A80"));

            // Pop up the ReThink Warning Card immediately and keep it open
            if (cardReThinkWarning != null) {
                cardReThinkWarning.setVisibility(View.VISIBLE);
                if (tvFlaggedWordCard != null) {
                    tvFlaggedWordCard.setText(isUrduMode 
                        ? "شناخت شدہ لفظ: \"" + flaggedWord + "\"" 
                        : "Detected Word: \"" + flaggedWord + "\"");
                }
            }
        } else {
            // Check if currently flagged word is still present in input text
            InputConnection ic = getCurrentInputConnection();
            boolean stillPresent = false;
            if (ic != null && currentFlaggedWord != null && !currentFlaggedWord.isEmpty()) {
                CharSequence textBefore = ic.getTextBeforeCursor(150, 0);
                if (textBefore != null && textBefore.toString().toLowerCase(Locale.ROOT).contains(currentFlaggedWord.toLowerCase(Locale.ROOT))) {
                    stillPresent = true;
                }
            }

            // Only clear warning if the offensive word was deleted or dismissed
            if (!stillPresent) {
                currentFlaggedWord = "";
                safetyBar.setBackgroundColor(Color.parseColor("#18281E"));
                tvSafetyStatus.setText(isUrduMode ? "🛡️ Offense Guard • صاف و محفوظ" : "🛡️ Offense Guard Active • Clean");
                tvSafetyStatus.setTextColor(Color.parseColor("#81C784"));

                if (cardReThinkWarning != null) {
                    cardReThinkWarning.setVisibility(View.GONE);
                }
            }
        }
    }

    private void handleClearWord() {
        if (currentFlaggedWord == null || currentFlaggedWord.isEmpty()) return;

        InputConnection ic = getCurrentInputConnection();
        if (ic == null) return;

        CharSequence textBefore = ic.getTextBeforeCursor(100, 0);
        if (textBefore == null) return;

        String textStr = textBefore.toString();
        int lastIndex = textStr.toLowerCase(Locale.ROOT).lastIndexOf(currentFlaggedWord.toLowerCase(Locale.ROOT));

        if (lastIndex != -1) {
            int charsFromEnd = textStr.length() - lastIndex;
            ic.deleteSurroundingText(charsFromEnd, 0);
        }

        if (cardReThinkWarning != null) {
            cardReThinkWarning.setVisibility(View.GONE);
        }
        currentFlaggedWord = "";
        setSafetyStatus(false, "");
    }

    /**
     * ⚡ Use Anyway — allows the word for THIS session/message only.
     * Next time the same word is typed, the warning will fire again.
     */
    private void handleUseAnyway() {
        if (currentFlaggedWord != null && !currentFlaggedWord.isEmpty()) {
            detector.allowWordOnce(currentFlaggedWord);
        }
        if (cardReThinkWarning != null) {
            cardReThinkWarning.setVisibility(View.GONE);
        }
        currentFlaggedWord = "";
        setSafetyStatus(false, "");
    }

    /**
     * ⭐ Always Allow — permanently whitelists the word across all sessions.
     * Saves to SharedPreferences and updates the detector's permanent list.
     */
    private void handleAlwaysAllow() {
        if (currentFlaggedWord != null && !currentFlaggedWord.isEmpty()) {
            detector.addPermanentAllowedWord(currentFlaggedWord);
            savePermanentAllowedWord(currentFlaggedWord);
        }
        if (cardReThinkWarning != null) {
            cardReThinkWarning.setVisibility(View.GONE);
        }
        currentFlaggedWord = "";
        setSafetyStatus(false, "");
    }

    /** Load persistent always-allowed words from SharedPreferences into the detector. */
    private void loadPermanentAllowedWords() {
        try {
            SharedPreferences prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE);
            String json = prefs.getString(PREFS_KEY_ALLOWED, "[]");
            org.json.JSONArray arr = new org.json.JSONArray(json);
            Set<String> words = new HashSet<>();
            for (int i = 0; i < arr.length(); i++) {
                String w = arr.optString(i, "").trim();
                if (!w.isEmpty()) words.add(w.toLowerCase(Locale.ROOT));
            }
            detector.setPermanentAllowedWords(words);
        } catch (Exception e) {
            // Ignore parse errors — start fresh
        }
    }

    /** Persist a newly allowed word into SharedPreferences. */
    private void savePermanentAllowedWord(String word) {
        try {
            SharedPreferences prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE);
            String json = prefs.getString(PREFS_KEY_ALLOWED, "[]");
            org.json.JSONArray arr = new org.json.JSONArray(json);
            String lowerWord = word.toLowerCase(Locale.ROOT).trim();
            // Avoid duplicates
            for (int i = 0; i < arr.length(); i++) {
                if (lowerWord.equals(arr.optString(i, "").toLowerCase(Locale.ROOT).trim())) return;
            }
            arr.put(lowerWord);
            prefs.edit().putString(PREFS_KEY_ALLOWED, arr.toString()).apply();
        } catch (Exception e) {
            // Ignore save errors
        }
    }
}
