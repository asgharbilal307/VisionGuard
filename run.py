# run.py  (in project root)
import os
from dotenv import load_dotenv

load_dotenv()

from flask_app.app import app

if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=5000, debug=debug)  # False in production per .env
