package com.offenseguard.keyboard;

import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.provider.Settings;
import android.view.inputmethod.InputMethodInfo;
import android.view.inputmethod.InputMethodManager;
import android.widget.Button;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import java.util.List;

public class MainActivity extends AppCompatActivity {

    private TextView tvStatus;
    private Button btnEnable;
    private Button btnSelect;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        tvStatus = findViewById(R.id.tvCurrentStatus);
        btnEnable = findViewById(R.id.btnEnableKeyboard);
        btnSelect = findViewById(R.id.btnSelectKeyboard);

        // Step 1: Open Android Language & Input settings
        btnEnable.setOnClickListener(v -> {
            Intent intent = new Intent(Settings.ACTION_INPUT_METHOD_SETTINGS);
            intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
            startActivity(intent);
        });

        // Step 2: Show input method switcher dialog
        btnSelect.setOnClickListener(v -> {
            InputMethodManager imm = (InputMethodManager) getSystemService(Context.INPUT_METHOD_SERVICE);
            if (imm != null) {
                imm.showInputMethodPicker();
            }
        });
    }

    @Override
    protected void onResume() {
        super.onResume();
        updateKeyboardStatus();
    }

    private void updateKeyboardStatus() {
        InputMethodManager imm = (InputMethodManager) getSystemService(Context.INPUT_METHOD_SERVICE);
        if (imm == null) return;

        List<InputMethodInfo> enabledMethods = imm.getEnabledInputMethodList();
        boolean isEnabled = false;
        String packageName = getPackageName();

        for (InputMethodInfo info : enabledMethods) {
            if (info.getPackageName().equals(packageName)) {
                isEnabled = true;
                break;
            }
        }

        if (isEnabled) {
            tvStatus.setText("✓ Offense Guard Keyboard is Enabled in Android Settings!");
            tvStatus.setTextColor(getColor(R.color.safety_clean_text));
            btnEnable.setText("✓ Enabled in Settings");
            btnEnable.setEnabled(false);
        } else {
            tvStatus.setText("⚠️ Offense Guard is not yet enabled in System Settings.");
            tvStatus.setTextColor(getColor(R.color.accent));
            btnEnable.setEnabled(true);
        }
    }
}
