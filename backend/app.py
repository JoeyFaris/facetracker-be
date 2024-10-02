from flask import Flask, render_template, jsonify
import subprocess
import sys
import os
import signal
import psutil

app = Flask(__name__)

script_process = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_script', methods=['POST'])
def run_script():
    global script_process
    try:
        script_process = subprocess.Popen(['python3', 'facetracker.py'], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE,
                                   universal_newlines=True)
        
        # Wait for a short time to catch immediate errors
        try:
            stdout, stderr = script_process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            # Script is running without immediate errors
            return jsonify({
                "status": "success", 
                "message": "Face tracker started successfully. Two windows should open: 'Colorful Face Mesh and Hands' and 'Original Frame'. To close the application, press 'q' while one of these windows is active."
            })
        
        # If we get here, the script exited quickly (probably due to an error)
        if script_process.returncode != 0:
            error_message = stderr.strip() if stderr else stdout.strip()
            return jsonify({"status": "error", "message": f"Script failed to start: {error_message}"})
        
        return jsonify({"status": "success", "message": "Script completed successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/stop_script', methods=['POST'])
def stop_script():
    global script_process
    if script_process:
        try:
            parent = psutil.Process(script_process.pid)
            for child in parent.children(recursive=True):
                child.terminate()
            parent.terminate()
            
            # Wait for the process to fully terminate
            script_process.wait(timeout=5)
            
            # Force kill if it's still running
            if script_process.poll() is None:
                parent.kill()
                for child in parent.children(recursive=True):
                    child.kill()
            
            script_process = None
            return jsonify({"status": "success", "message": "Script stopped successfully"})
        except Exception as e:
            return jsonify({"status": "error", "message": f"Failed to stop script: {str(e)}"})
    else:
        return jsonify({"status": "error", "message": "No script is currently running"})

if __name__ == '__main__':
    app.run(debug=True)
