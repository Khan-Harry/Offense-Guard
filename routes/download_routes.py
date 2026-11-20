"""
APK Download & Landing Page Routes
Provides mobile-friendly landing page and APK streaming endpoints.
"""

import os
from flask import Blueprint, send_file, send_from_directory, render_template_string, redirect

download_bp = Blueprint('download', __name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOWNLOADS_DIR = os.path.join(BASE_DIR, 'static', 'downloads')


@download_bp.route('/download')
def download_page():
    """
    Mobile-friendly landing page to download both OffenseGuard APKs:
    - Main Dashboard App (React Native)
    - AI Keyboard APK (System Keyboard)
    """
    keyboard_apk = os.path.join(DOWNLOADS_DIR, 'OffenseGuard_v1.8.apk')
    app_apk = os.path.join(DOWNLOADS_DIR, 'OffenseGuard_App_v1.0.apk')

    keyboard_size = round(os.path.getsize(keyboard_apk) / (1024 * 1024), 1) if os.path.exists(keyboard_apk) else 5.8
    app_size = round(os.path.getsize(app_apk) / (1024 * 1024), 1) if os.path.exists(app_apk) else 56.2

    github_app_url = "https://github.com/Khan-Harry/Offense-Guard/releases/download/v1.0/OffenseGuard_App_v1.0.apk"
    github_kbd_url = "https://github.com/Khan-Harry/Offense-Guard/releases/download/v1.0/OffenseGuard_v1.8.apk"

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Offense Guard — Download</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; font-family: 'Inter', -apple-system, sans-serif; }}
            body {{ background: #0B0E14; color: #E5E7EB; display: flex; justify-content: center; align-items: center; min-height: 100vh; padding: 20px; }}
            .card {{ background: #161B26; border: 1px solid #232B3E; border-radius: 24px; max-width: 500px; width: 100%; padding: 36px 28px; text-align: center; box-shadow: 0 20px 40px rgba(0,0,0,0.4); }}
            .badge {{ display: inline-block; background: rgba(99, 102, 241, 0.15); color: #818CF8; border: 1px solid rgba(99, 102, 241, 0.3); padding: 6px 14px; border-radius: 999px; font-size: 12px; font-weight: 700; letter-spacing: 0.5px; text-transform: uppercase; margin-bottom: 20px; }}
            .icon-box {{ width: 84px; height: 84px; margin: 0 auto 20px; background: linear-gradient(135deg, #6366F1, #8B5CF6); border-radius: 22px; display: flex; align-items: center; justify-content: center; font-size: 42px; box-shadow: 0 10px 25px rgba(99, 102, 241, 0.35); }}
            h1 {{ font-size: 26px; font-weight: 800; color: #FFFFFF; margin-bottom: 10px; }}
            p.desc {{ font-size: 14px; color: #9CA3AF; line-height: 1.6; margin-bottom: 28px; }}
            .apk-section {{ margin-bottom: 18px; }}
            .apk-label {{ font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; color: #6B7280; margin-bottom: 8px; text-align: left; }}
            .btn-download {{ display: block; width: 100%; background: linear-gradient(135deg, #6366F1, #4F46E5); color: #FFFFFF; text-decoration: none; padding: 16px 24px; border-radius: 14px; font-weight: 700; font-size: 15px; box-shadow: 0 8px 20px rgba(99, 102, 241, 0.4); transition: transform 0.2s, box-shadow 0.2s; }}
            .btn-download.secondary {{ background: linear-gradient(135deg, #374151, #1F2937); box-shadow: 0 8px 20px rgba(0,0,0,0.3); }}
            .btn-download:hover {{ transform: translateY(-2px); }}
            .meta {{ font-size: 12px; color: #6B7280; margin-top: 8px; margin-bottom: 6px; text-align: left; }}
            .divider {{ border: none; border-top: 1px solid #232B3E; margin: 20px 0; }}
            .steps {{ text-align: left; background: #11151F; border-radius: 16px; padding: 20px; border: 1px solid #1E2536; margin-top: 24px; }}
            .steps h3 {{ font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px; color: #818CF8; margin-bottom: 14px; }}
            .step-item {{ display: flex; align-items: flex-start; margin-bottom: 12px; }}
            .step-item:last-child {{ margin-bottom: 0; }}
            .step-num {{ width: 24px; height: 24px; background: #232B3E; color: #E5E7EB; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 700; margin-right: 12px; flex-shrink: 0; }}
            .step-text {{ font-size: 13px; color: #D1D5DB; line-height: 1.4; }}
            .step-text strong {{ color: #FFFFFF; }}
        </style>
    </head>
    <body>
        <div class="card">
            <span class="badge">FYP Release — v1.0 + v1.8</span>
            <div class="icon-box">🛡️</div>
            <h1>Offense Guard</h1>
            <p class="desc">AI-powered Urdu/Roman Urdu offensive language detection. Install both APKs for the full experience.</p>

            <div class="apk-section">
                <div class="apk-label">📱 Step 1 — Main Dashboard App</div>
                <a href="{github_app_url}" class="btn-download">
                    ⬇️ Download Dashboard App ({app_size} MB)
                </a>
                <div class="meta">Analytics, real-time detection & keyboard setup guide</div>
            </div>

            <div class="apk-section">
                <div class="apk-label">⌨️ Step 2 — AI Keyboard</div>
                <a href="{github_kbd_url}" class="btn-download secondary">
                    ⬇️ Download AI Keyboard ({keyboard_size} MB)
                </a>
                <div class="meta">System-wide offensive language blocker keyboard</div>
            </div>

            <div class="steps">
                <h3>Setup Guide</h3>
                <div class="step-item">
                    <div class="step-num">1</div>
                    <div class="step-text">Install <strong>Dashboard App</strong> first. Open it and go to Settings → Enable Keyboard.</div>
                </div>
                <div class="step-item">
                    <div class="step-num">2</div>
                    <div class="step-text">Install the <strong>AI Keyboard APK</strong> and enable it in Android Settings.</div>
                </div>
                <div class="step-item">
                    <div class="step-num">3</div>
                    <div class="step-text">Select <strong>OffenseGuard Keyboard</strong> as your active keyboard. Done! 🎉</div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return render_template_string(html)


@download_bp.route('/download/app')
def download_app_file():
    """Direct endpoint to stream the main OffenseGuard Dashboard APK."""
    apk_filename = 'OffenseGuard_App_v1.0.apk'
    github_release_app = "https://github.com/Khan-Harry/Offense-Guard/releases/download/v1.0/OffenseGuard_App_v1.0.apk"
    target_path = os.path.join(DOWNLOADS_DIR, apk_filename)
    if not os.path.exists(target_path):
        return redirect(github_release_app)
    return send_from_directory(DOWNLOADS_DIR, apk_filename, as_attachment=True, mimetype='application/vnd.android.package-archive')


@download_bp.route('/download/apk')
def download_apk_file():
    """Direct endpoint to stream the OffenseGuard AI Keyboard APK."""
    apk_filename = 'OffenseGuard_v1.8.apk'
    github_release_keyboard = "https://github.com/Khan-Harry/Offense-Guard/releases/download/v1.0/OffenseGuard_v1.8.apk"
    target_path = os.path.join(DOWNLOADS_DIR, apk_filename)
    if not os.path.exists(target_path):
        alt_apk = os.path.join(BASE_DIR, 'offense_guard_keyboard', 'app', 'build', 'outputs', 'apk', 'debug', 'app-debug.apk')
        if os.path.exists(alt_apk):
            return send_file(alt_apk, as_attachment=True, download_name=apk_filename, mimetype='application/vnd.android.package-archive')
        return redirect(github_release_keyboard)
    return send_from_directory(DOWNLOADS_DIR, apk_filename, as_attachment=True, mimetype='application/vnd.android.package-archive')
