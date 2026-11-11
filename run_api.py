import os

from dotenv import load_dotenv

load_dotenv()

from flask_app.api import app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
