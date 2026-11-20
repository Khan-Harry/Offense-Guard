"""
Email Notification Service
Handles OTP verification email dispatching via SMTP or dev logger fallback.
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def send_otp_email(recipient_email, otp_code):
    """
    Send a 6-digit OTP code to user's email via SMTP.
    Falls back gracefully to server console logs if SMTP is not configured.
    """
    smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
    smtp_port = int(os.getenv('SMTP_PORT', 587))
    smtp_user = os.getenv('SMTP_EMAIL', '')
    smtp_pass = os.getenv('SMTP_PASSWORD', '')

    print(f"\n=======================================================")
    print(f"🔐 [OFFENSE GUARD] PASSWORD RESET OTP DISPATCH")
    print(f"📧 Destination Email: {recipient_email}")
    print(f"🔑 6-DIGIT OTP CODE : {otp_code}")
    print(f"⏳ Expiration Time  : 10 Minutes")
    print(f"=======================================================\n")

    if not smtp_user or not smtp_pass:
        print("ℹ️ Note: SMTP_EMAIL / SMTP_PASSWORD environment variables are not set.")
        print("👉 The OTP has been printed to the server console above for development testing.\n")
        return True

    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = 'Offense Guard - Password Reset Verification Code'
        msg['From'] = f"Offense Guard Security <{smtp_user}>"
        msg['To'] = recipient_email

        html_content = f"""
        <html>
          <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #f4f5fa; padding: 24px;">
            <div style="max-width: 480px; margin: 0 auto; background: #ffffff; border-radius: 16px; padding: 32px; box-shadow: 0 4px 20px rgba(0,0,0,0.06); border: 1px solid #e5e7eb;">
              <div style="text-align: center; margin-bottom: 20px;">
                <h2 style="color: #6750A4; margin: 0; font-size: 24px;">🛡️ Offense Guard</h2>
                <p style="color: #6b7280; font-size: 14px; margin-top: 4px;">Security & Account Recovery</p>
              </div>
              <p style="color: #1f2937; font-size: 15px; line-height: 1.5;">Hello,</p>
              <p style="color: #4b5563; font-size: 15px; line-height: 1.5;">We received a request to reset your password. Enter the 6-digit verification code below to verify your identity:</p>
              
              <div style="text-align: center; margin: 28px 0;">
                <span style="display: inline-block; font-size: 34px; font-weight: 800; letter-spacing: 10px; color: #6750A4; background: #EDE9F8; padding: 14px 28px; border-radius: 12px; border: 1px dashed #6750A4;">
                  {otp_code}
                </span>
              </div>
              
              <p style="color: #6b7280; font-size: 13px; line-height: 1.4;">
                • This verification code is valid for <strong>10 minutes</strong>.<br/>
                • Never share this code with anyone.<br/>
                • If you did not request this, you can safely ignore this email.
              </p>
              <hr style="border: none; border-top: 1px solid #f3f4f6; margin: 24px 0;" />
              <p style="color: #9ca3af; font-size: 12px; text-align: center; margin: 0;">
                Offense Guard AI Platform • Automated Security Notification
              </p>
            </div>
          </body>
        </html>
        """
        msg.attach(MIMEText(html_content, 'html'))

        server = smtplib.SMTP(smtp_server, smtp_port, timeout=10)
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.sendmail(smtp_user, [recipient_email], msg.as_string())
        server.quit()
        print(f"✅ OTP email successfully dispatched to {recipient_email}")
        return True
    except Exception as e:
        print(f"❌ Failed to dispatch email via SMTP: {e}")
        return False
