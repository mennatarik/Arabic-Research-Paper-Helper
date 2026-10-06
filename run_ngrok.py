"""Run Streamlit and open a public ngrok URL.  Needs NGROK_AUTHTOKEN in the environment."""
import os
import subprocess
import sys
import time

from pyngrok import ngrok

token = os.getenv("NGROK_AUTHTOKEN")
if not token:
    sys.exit("Set NGROK_AUTHTOKEN first (free at dashboard.ngrok.com).")

ngrok.set_auth_token(token)
proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py",
                         "--server.port", "8501", "--server.headless", "true"])
time.sleep(6)
print("Public URL:", ngrok.connect(8501).public_url, flush=True)

try:
    proc.wait()
except KeyboardInterrupt:
    pass
finally:
    ngrok.kill()
    proc.terminate()