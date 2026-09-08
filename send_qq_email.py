#!/usr/bin/env python3
"""Send the desktop configuration guide to your own QQ mailbox via SMTP.

Credentials are read from a config file (never printed). Create the file
  /Users/travis.zhao/ultrahdr/.qq_smtp.env
with lines:
  QQ_EMAIL=you@qq.com          (sender, must have SMTP enabled)
  QQ_SMTP_AUTH=xxxxxxxxxxxxxx  (the 16-digit QQ 授权码, NOT your password)
  QQ_TO=you@qq.com             (recipient; defaults to QQ_EMAIL)

Then run:
  python send_qq_email.py
"""
import os
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parent / ".qq_smtp.env"
GUIDE = Path(__file__).resolve().parent / "desktop_setup_email.txt"

SMTP_HOST = "smtp.qq.com"
SMTP_PORT = 465  # SSL
FROM_NAME = "ultrahdr 配置"  # 显示名，可改


def load_env_file(path) -> dict:
    path = Path(path)
    if not path.is_file():
        return {}
    data = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        data[key.strip()] = value.strip()
    return data


def parse_guide(path) -> tuple[str, str]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    subject = next((l for l in lines if l.lower().startswith("subject:")), "台式机配置指南")
    subject = subject.split(":", 1)[1].strip()
    body = "\n".join(lines[1:]).strip()  # drop the Subject line
    return subject, body


def main():
    env = load_env_file(ENV_FILE)
    sender = env.get("QQ_EMAIL", os.environ.get("QQ_EMAIL", ""))
    auth = env.get("QQ_SMTP_AUTH", os.environ.get("QQ_SMTP_AUTH", ""))
    to = env.get("QQ_TO", os.environ.get("QQ_TO", sender))

    if not sender or not auth:
        print("缺配置：请在 %s 里填 QQ_EMAIL 和 QQ_SMTP_AUTH" % ENV_FILE)
        return
    if not to:
        print("缺收件人：QQ_TO 未设置")
        return

    subject, body = parse_guide(GUIDE)
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"{FROM_NAME} <{sender}>"
    msg["To"] = to
    # Plain text for broad compatibility; readers render headings as-is.
    msg.attach(MIMEText(body, "plain", "utf-8"))

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=context, timeout=30) as server:
            server.login(sender, auth)
            server.sendmail(sender, [to], msg.as_string())
        print("✅ 已发送到 %s（主题：%s）" % (to, subject))
    except Exception as error:  # noqa: BLE001 - report friendly message
        print(f"❌ 发送失败：{type(error).__name__}: {error}")
        print("   请确认：1) 已在 QQ 邮箱开启 SMTP 服务；2) 用的是 16 位『授权码』而非密码；3) 网络可达 smtp.qq.com:465")


if __name__ == "__main__":
    main()
