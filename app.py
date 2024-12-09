from flask import Flask, request, render_template_string
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kalman Filter</title>
    <style>
        :root {
            --bg-primary: #121212;
            --bg-secondary: #1E1E1E;
            --text-primary: #E0E0E0;
            --text-secondary: #A0A0A0;
            --accent-color: #4CAF50;
            --border-color: #333;
        }

        body {
            font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            margin: 0;
            padding: 20px;
        }

        .container {
            display: flex;
            gap: 30px;
            justify-content: center;
            align-items: flex-start;
            margin: 0 auto;
        }

        .sidebar {
            flex: 1;
            max-width: 500px;
            background-color: var(--bg-secondary);
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        .results {
            flex: 1.5;
            background-color: var(--bg-secondary);
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            min-height: 100px; 
            box-sizing: border-box;
        }

        h1 {
            text-align: center;
            color: var(--accent-color);
            border-bottom: 2px solid var(--accent-color);
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        
        h2 {
            color: var(--accent-color);
            padding-bottom: 10px;
            margin-bottom: 20px;
            margin-top: 0px;
        }

        form {
            display: grid;
            gap: 15px;
        }

        label {
            display: block;
            color: var(--text-secondary);
            font-size: 0.9em;
        }

        input {
            width: 100%;
            padding: 8px;
            margin-top: 5px;
            background-color: var(--bg-primary);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            border-radius: 4px;
            transition: border-color 0.3s ease;
        }

        input:focus {
            outline: none;
            border-color: var(--accent-color);
        }

        button {
            width: 100%;
            padding: 12px;
            background-color: transparent;
            color: var(--accent-color);
            border: 2px solid var(--accent-color);
            border-radius: 6px;
            cursor: pointer;
            font-weight: medium;
            transition: all 0.3s ease;
            margin-top: 15px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        button:hover {
            background-color: var(--accent-color);
            color: var(--bg-primary);
        }

        .results img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 20px auto;
            border-radius: 8px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        }

        .results p {
            background-color: var(--bg-primary);
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
        }

        @media (max-width: 768px) {
            .container {
                flex-direction: column;
            }
            
            .sidebar, .results {
                width: 100%;
            }
        }
    </style>
</head>
<body>
    <h1>Kalman Filter</h1>
    <div class="container">
        <div class="sidebar">
            <h2>Parameters</h2>
            <form method="post">
                <label>Transition Matrix (F): <input type="number" name="F" step="0.1" value="{{ F }}"></label>
                <label>Measurement Matrix (H): <input type="number" name="H" step="0.1" value="{{ H }}"></label>
                <label>Process Noise Covariance (Q): <input type="number" name="Q" step="0.1" value="{{ Q }}"></label>
                <label>Measurement Noise Covariance (R): <input type="number" name="R" step="0.1" value="{{ R }}"></label>
                <label>Initial Estimate Error Covariance (P): <input type="number" name="P" step="0.1" value="{{ P }}"></label>
                <label>Initial State (x): <input type="number" name="x" step="0.1" value="{{ x }}"></label>
                <label>Frequency: <input type="number" name="frequency" step="0.1" value="{{ frequency }}"></label>
                <label>Amplitude: <input type="number" name="amplitude" step="0.1" value="{{ amplitude }}"></label>
                <label>Offset: <input type="number" name="offset" step="0.1" value="{{ offset }}"></label>
                <label>Sampling Interval: <input type="number" name="sampling_interval" step="0.001" value="{{ sampling_interval }}"></label>
                <label>Total Time: <input type="number" name="total_time" step="0.1" value="{{ total_time }}"></label>
                <label>Noise Variance: <input type="number" name="noise_variance" step="0.1" value="{{ noise_variance }}"></label>
                <button type="submit">Run Kalman Filter</button>
            </form>
        </div>
        
        {% if plot_url %}
        <div class="results">
            <h2>Kalman Filter Visualization</h2>
            <img src="data:image/png;base64,{{ plot_url }}" alt="Kalman Filter Results">
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

class KalmanFilter:
    def __init__(self, F, H, Q, R, P, x):
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.P = P
        self.x = x

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        return self.x

    def update(self, z):
        K = np.dot(self.P, self.H.T) / (np.dot(self.H, np.dot(self.P, self.H.T)) + self.R)
        self.x = self.x + K * (z - np.dot(self.H, self.x))
        self.P = (np.eye(len(self.P)) - K * self.H) @ self.P
        return self.x


@app.route('/', methods=['GET', 'POST'])
def index():
    # Default parameters
    params = {
        "F": 1.0,
        "H": 1.0,
        "Q": 1.0,
        "R": 10.0,
        "P": 1.0,
        "x": 0.0,
        "frequency": 1.0,
        "amplitude": 5.0,
        "offset": 10.0,
        "sampling_interval": 0.001,
        "total_time": 1.0,
        "noise_variance": 16.0,
    }

    plot_url = None

    if request.method == 'POST':
        # Update parameters from user input
        for key in params.keys():
            params[key] = float(request.form.get(key, params[key]))

        # Extract parameters
        F = np.array([[params["F"]]])
        H = np.array([[params["H"]]])
        Q = np.array([[params["Q"]]])
        R = np.array([[params["R"]]])
        P = np.array([[params["P"]]])
        x = np.array([[params["x"]]])
        frequency = params["frequency"]
        amplitude = params["amplitude"]
        offset = params["offset"]
        sampling_interval = params["sampling_interval"]
        total_time = params["total_time"]
        noise_variance = params["noise_variance"]

        # Kalman filter setup
        kf = KalmanFilter(F, H, Q, R, P, x)

        # Signal generation
        time_steps = np.arange(0, total_time, sampling_interval)
        true_signal = offset + amplitude * np.sin(2 * np.pi * frequency * time_steps)
        noise_std_dev = np.sqrt(noise_variance)
        noisy_signal = [val + np.random.normal(0, noise_std_dev) for val in true_signal]

        # Apply Kalman Filter
        kalman_estimates = []
        for measurement in noisy_signal:
            kf.predict()
            estimate = kf.update(measurement)
            kalman_estimates.append(estimate[0][0])

        # Generate plot
        plt.figure(figsize=(10, 6))
        plt.plot(time_steps, noisy_signal, label='Noisy Signal', color='orange', alpha=0.6)
        plt.plot(time_steps, true_signal, label='True Signal', linestyle='--', color='blue')
        plt.plot(time_steps, kalman_estimates, label='Kalman Filter Estimate', color='green')
        plt.xlabel('Time (s)')
        plt.ylabel('Value')
        plt.title('Kalman Filter')
        plt.legend()
        plt.grid()

        # Save plot to a bytes buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_url = base64.b64encode(buffer.read()).decode('utf-8')
        buffer.close()

    return render_template_string(
        HTML_TEMPLATE, **params,
        plot_url=plot_url,
    )


if __name__ == "__main__":
    app.run(debug=True)
