from flask import Flask, jsonify
from flask_cors import CORS
import subprocess

app = Flask(__name__)
CORS(app)

@app.route('/run_script', methods=['POST'])
def run_script():
    try:
        subprocess.Popen(['python3', 'facetracker.py'])
        return jsonify({"status": "success", "message": "Script started successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
