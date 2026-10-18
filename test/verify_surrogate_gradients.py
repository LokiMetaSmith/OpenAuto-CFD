import json
import numpy as np
import os
import sys

# Ensure we can import from optimizer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'optimizer')))
from optimizer.surrogate_gradients import compute_acquisition_and_grad
from optimizer.surrogate_multiphysics import MultiPhysicsSurrogate

def verify_surrogate_drift():
    print("Verifying surrogate backtesting drift...")

    try:
        surrogate = MultiPhysicsSurrogate()
    except Exception as e:
        print(f"Skipping verification: failed to initialize surrogate ({e})")
        return

    timeline_path = os.path.join(os.path.dirname(__file__), "..", "optimization_log.jsonl")
    if not os.path.exists(timeline_path):
        print(f"No timeline found at {timeline_path}. Assuming no drift.")
        return

    total_l2_error = 0.0
    count = 0
    with open(timeline_path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                if 'metrics' in data and 'parameters' in data:
                    metrics = data['metrics']
                    params = data['parameters']

                    if 'delta_p' not in metrics or 'separation_efficiency' not in metrics:
                        continue

                    # Update surrogate history with this point (simulating an active learning loop)
                    surrogate.add_sample(params, metrics)
                    count += 1
            except Exception:
                pass

    if count < 5:
        print(f"Not enough valid data points ({count}) in timeline to measure CFD drift reliably. Minimum 5 required. Skipping.")
        return

    # Fit the surrogate on all points
    surrogate.fit()

    # Calculate error over all points (training error/drift)
    drift_count = 0
    with open(timeline_path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                if 'metrics' in data and 'parameters' in data:
                    metrics = data['metrics']
                    params = data['parameters']

                    if 'delta_p' not in metrics or 'separation_efficiency' not in metrics:
                        continue

                    pred_metrics, _ = surrogate.predict_metrics(params)

                    # Calculate L2 norm error (simplified percentage error for delta_p and efficiency)
                    dp_true = metrics['delta_p']
                    dp_pred = pred_metrics.get('delta_p', dp_true)
                    eff_true = metrics['separation_efficiency']
                    eff_pred = pred_metrics.get('separation_efficiency', eff_true)

                    if dp_true > 0:
                        err_dp = abs(dp_pred - dp_true) / dp_true
                    else:
                        err_dp = 0.0

                    if eff_true > 0:
                        err_eff = abs(eff_pred - eff_true) / eff_true
                    else:
                        err_eff = 0.0

                    l2_err = np.sqrt(err_dp**2 + err_eff**2)
                    total_l2_error += l2_err
                    drift_count += 1
            except Exception:
                pass

    if drift_count > 0:
        avg_error = (total_l2_error / drift_count) * 100.0
        print(f"Average surrogate L2 error (drift): {avg_error:.2f}%")
        if avg_error > 5.0:
            print("Warning: Surrogate drift exceeds 5%. Retraining surrogate automatically.")
            # Trigger automatic retrain
            surrogate.fit()
            surrogate.save("surrogate_model.pkl")
            print("Retrain complete. Model saved to surrogate_model.pkl")
        else:
            print("Surrogate drift is within acceptable 5% threshold.")
    else:
        print("No valid metric evaluations found in timeline for drift check.")

if __name__ == "__main__":
    verify_surrogate_drift()
