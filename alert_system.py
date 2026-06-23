import datetime
import json
import os
import random
import smtplib
import ssl
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


GMAIL_ADDRESS = os.getenv("GMAIL_ADDRESS", "")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "")

CONTACTS_FILE = Path(__file__).parent / "contacts.json"
_pending_otps = {}


def load_contact() -> dict | None:
    if CONTACTS_FILE.exists():
        try:
            return json.loads(CONTACTS_FILE.read_text())
        except Exception:
            return None
    return None


def save_contact(email: str, phone: str = "") -> None:
    CONTACTS_FILE.write_text(json.dumps({
        "email": email,
        "phone": phone,
        "verified_at": datetime.datetime.now().isoformat(),
    }, indent=2))


def delete_contact() -> None:
    if CONTACTS_FILE.exists():
        CONTACTS_FILE.unlink()


def generate_otp() -> str:
    return str(random.randint(100000, 999999))


def send_otp_email(email: str, phone: str = "") -> dict:
    if not GMAIL_ADDRESS or not GMAIL_APP_PASSWORD:
        return {"success": False, "error": "Gmail credentials not set in .env file"}

    otp = generate_otp()
    expires_at = datetime.datetime.now() + datetime.timedelta(minutes=10)
    _pending_otps[email] = {
        "otp": otp,
        "expires_at": expires_at,
        "phone": phone,
    }

    msg = MIMEMultipart("alternative")
    msg["Subject"] = "CrisisSense OTP Verification Code"
    msg["From"] = GMAIL_ADDRESS
    msg["To"] = email

    html = f"""
    <html>
      <body style="font-family:Arial,sans-serif;background:#0d0d1a;color:#fff;padding:40px;">
        <div style="max-width:480px;margin:auto;background:#1a1a2e;border-radius:12px;
                    padding:32px;border:1px solid #2E74B5;">
          <h2 style="color:#2E74B5;margin-top:0;">CrisisSense Alert Setup</h2>
          <p>You are registering as the authority contact for CrisisSense alerts.</p>
          <p style="margin-top:24px;">Your OTP verification code:</p>
          <div style="background:#0d0d1a;border-radius:8px;padding:20px;text-align:center;
                      font-size:36px;font-weight:bold;letter-spacing:12px;color:#ff4444;
                      margin:16px 0;">{otp}</div>
          <p style="color:#888;font-size:13px;">
            Expires in 10 minutes.<br>
            If you did not request this, ignore this email.
          </p>
        </div>
      </body>
    </html>
    """

    msg.attach(MIMEText(html, "html"))

    try:
        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx) as server:
            server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
            server.sendmail(GMAIL_ADDRESS, email, msg.as_string())
        return {"success": True}
    except smtplib.SMTPAuthenticationError:
        return {
            "success": False,
            "error": "Gmail authentication failed. Check your App Password in .env",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def verify_otp(email: str, otp_input: str) -> dict:
    entry = _pending_otps.get(email)

    if not entry:
        return {"success": False, "error": "No OTP sent to this email. Request a new one."}

    if datetime.datetime.now() > entry["expires_at"]:
        del _pending_otps[email]
        return {"success": False, "error": "OTP expired. Please request a new one."}

    if otp_input.strip() != entry["otp"]:
        return {"success": False, "error": "Incorrect OTP. Please try again."}

    save_contact(email, entry.get("phone", ""))
    del _pending_otps[email]

    return {
        "success": True,
        "message": f"Verified! Email alerts will be sent to {email}",
    }


def send_email_alert(
    crowd_count: int,
    risk_label: str,
    risk_score: int,
    location: str = "Camera Feed",
    frame_jpeg_bytes: bytes | None = None,
) -> dict:
    if not GMAIL_ADDRESS or not GMAIL_APP_PASSWORD:
        return {"success": False, "error": "Gmail credentials not configured"}

    contact = load_contact()
    if not contact:
        return {"success": False, "error": "No verified contact found"}

    to_email = contact["email"]
    now = datetime.datetime.now().strftime("%d %b %Y, %I:%M:%S %p")
    color = "#ff4444" if risk_label == "DANGER" else "#ffaa00"

    msg = MIMEMultipart("mixed")
    msg["Subject"] = f"CrisisSense STAMPEDE ALERT - {location}"
    msg["From"] = GMAIL_ADDRESS
    msg["To"] = to_email

    html = f"""
    <html>
      <body style="font-family:Arial,sans-serif;background:#f4f4f4;padding:32px;">
        <div style="max-width:520px;margin:auto;background:#fff;border-radius:12px;
                    padding:32px;border-top:6px solid {color};">
          <h1 style="color:{color};margin-top:0;">STAMPEDE ALERT</h1>

          <table style="width:100%;border-collapse:collapse;margin:20px 0;">
            <tr style="background:#f9f9f9;">
              <td style="padding:12px;font-weight:bold;color:#555;">Location</td>
              <td style="padding:12px;">{location}</td>
            </tr>
            <tr>
              <td style="padding:12px;font-weight:bold;color:#555;">Time</td>
              <td style="padding:12px;">{now}</td>
            </tr>
            <tr style="background:#f9f9f9;">
              <td style="padding:12px;font-weight:bold;color:#555;">Crowd Count</td>
              <td style="padding:12px;font-size:20px;font-weight:bold;">{crowd_count} people</td>
            </tr>
            <tr>
              <td style="padding:12px;font-weight:bold;color:#555;">Risk Level</td>
              <td style="padding:12px;font-size:20px;font-weight:bold;color:{color};">
                {risk_label} ({risk_score}/100)
              </td>
            </tr>
          </table>

          <p style="background:#fff3cd;padding:16px;border-radius:8px;color:#856404;">
            Immediate attention required. Please dispatch crowd control personnel to {location}.
          </p>

          {"<p style='color:#888;font-size:12px;'>Frame snapshot attached.</p>" if frame_jpeg_bytes else ""}

          <p style="color:#aaa;font-size:11px;margin-top:24px;">
            Sent automatically by CrisisSense AI Crowd Monitoring System
          </p>
        </div>
      </body>
    </html>
    """

    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(html, "html"))
    msg.attach(alt)

    if frame_jpeg_bytes:
        img = MIMEImage(frame_jpeg_bytes, _subtype="jpeg")
        img.add_header(
            "Content-Disposition",
            "attachment",
            filename=f"stampede_frame_{datetime.datetime.now().strftime('%H%M%S')}.jpg",
        )
        msg.attach(img)

    try:
        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx) as server:
            server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
            server.sendmail(GMAIL_ADDRESS, to_email, msg.as_string())
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


def send_alert(
    crowd_count: int,
    risk_label: str,
    risk_score: int,
    location: str = "Camera Feed",
    frame_jpeg_bytes: bytes | None = None,
) -> dict:
    email_result = send_email_alert(
        crowd_count,
        risk_label,
        risk_score,
        location,
        frame_jpeg_bytes,
    )

    return {
        "success": email_result["success"],
        "email": email_result,
        "error": email_result.get("error"),
    }


_last_alert_time: datetime.datetime | None = None
ALERT_COOLDOWN_SECONDS = 120


def should_send_alert() -> bool:
    global _last_alert_time

    if _last_alert_time is None:
        return True

    return (datetime.datetime.now() - _last_alert_time).total_seconds() >= ALERT_COOLDOWN_SECONDS


def mark_alert_sent() -> None:
    global _last_alert_time
    _last_alert_time = datetime.datetime.now()


def send_whatsapp_alert(
    crowd_count,
    risk_label,
    risk_score,
    location="Camera Feed",
    frame_jpeg_bytes=None,
):
    return send_alert(
        crowd_count,
        risk_label,
        risk_score,
        location,
        frame_jpeg_bytes,
    )
