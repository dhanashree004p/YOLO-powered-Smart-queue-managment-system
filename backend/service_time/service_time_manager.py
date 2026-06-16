from sklearn.linear_model import LinearRegression
import time
import numpy as np
from collections import deque

class ServiceTimeManager:
    """
    Manages the learning of dynamic service (billing) times for a queue.
    
    EXPLANATION: Prediction Model (Linear Regression)
    -------------------------------------------------
    We use a simple Linear Regression model to predict the service time.
    Features used:
    - queue_length: Number of people in the queue.
    - current_avg_service_time: The rolling average of the last 50 service times.
    - queue_density: (Calculated internally if needed).
    
    The model is retrained incrementally every 20 samples to adapt to changing patterns.
    """
    def __init__(self, history_size=50, default_time=5.0):
        self.history_size = history_size
        self.default_time = default_time
        
        # Store completed billing times (in seconds)
        self.billing_times = deque(maxlen=history_size)
        
        # Store features for regression training: (queue_length, duration)
        self.training_data = [] # List of (queue_len, duration)
        
        # Track currently detected IDs in cashier area: {tracker_id: (entry_timestamp, queue_len_at_entry)}
        self.active_service = {}
        
        # Regression model
        self.model = LinearRegression()
        self.model_trained = False
        self.samples_since_train = 0

    def update(self, current_ids_in_cashier_roi, current_queue_len=0):
        """
        Update tracking of who is in the cashier area.
        - New IDs -> Record entry time and current queue length.
        - Missing IDs -> Record exit time, compute duration, and save for training.
        """
        now = time.time()
        current_ids_set = set(current_ids_in_cashier_roi)
        
        # Check for exits
        exited_ids = [pid for pid in self.active_service if pid not in current_ids_set]
        
        for pid in exited_ids:
            entry_time, q_len = self.active_service.pop(pid)
            duration = now - entry_time
            
            if duration > 1.0: 
                self.billing_times.append(duration)
                self.training_data.append((q_len, duration))
                self.samples_since_train += 1
                
                # Keep training data manageable
                if len(self.training_data) > 200:
                    self.training_data = self.training_data[-200:]
                
                # Retrain model periodically (every 20 samples)
                if self.samples_since_train >= 20 and len(self.training_data) >= 10:
                    self._train_model()
                    self.samples_since_train = 0
                
        # Check for entries
        for pid in current_ids_set:
            if pid not in self.active_service:
                self.active_service[pid] = (now, current_queue_len)

    def _train_model(self):
        """Trains the Linear Regression model on collected data."""
        try:
            X = np.array([d[0] for d in self.training_data]).reshape(-1, 1)
            y = np.array([d[1] for d in self.training_data])
            self.model.fit(X, y)
            self.model_trained = True
        except Exception as e:
            print(f"Model Training Error: {e}")

    def predict_service_time(self, queue_len):
        """
        Returns a robust estimate of service time using median of last N.
        """
        return self.get_median_billing_time()

    def get_avg_billing_time(self):
        """Returns the rolling average of billing times."""
        if not self.billing_times:
            return self.default_time
        return float(np.mean(self.billing_times))

    def get_median_billing_time(self):
        """Returns the rolling median of billing times."""
        if not self.billing_times:
            return self.default_time
        return float(np.median(self.billing_times))

    def get_current_remaining_time(self, avg_time: float | None = None):
        """Estimates the remaining time for the current person being served."""
        if not self.active_service:
            return 0.0
        
        # We use the predicted or avg time as the 'expected' total duration
        now = time.time()
        remaining = []
        for pid, (entry_ts, q_len) in self.active_service.items():
            # Use prediction if possible for the specific queue state they entered at
            expected = (self.get_median_billing_time() if avg_time is None else float(avg_time))
            elapsed = now - entry_ts
            remaining.append(max(expected - elapsed, 0.0))
        return min(remaining) if remaining else 0.0
