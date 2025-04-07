import time
import numpy as np

class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint=0):
        self.Kp = Kp  # Proportional gain
        self.Ki = Ki  # Integral gain
        self.Kd = Kd  # Derivative gain
        
        self.setpoint = setpoint  # Desired target value
        
        self.previous_error = 0  # To store previous error
        self.integral = 0  # Integral term
        self.last_time = time.time()  # Last time the PID was updated

    def compute(self, measured_value):
        """Compute the control output based on the measured value."""
        current_time = time.time()
        dt = current_time - self.last_time  # Time difference
        if dt == 0:
            return 0  # Prevent division by zero
        
        error = self.setpoint - measured_value  # Error calculation
        
        self.integral += error * dt  # Integral term
        derivative = (error - self.previous_error) / dt  # Derivative term
        
        output = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)
        print(f"{(self.Kp * error)}, {(self.Ki * self.integral)}, {(self.Kd * derivative)}")
        
        self.previous_error = error  # Store error for next iteration
        self.last_time = current_time  # Update last time
        
        return output  # Control output

# Example usage
pid = PIDController(Kp=1e-5, Ki=1e-7, Kd=6 * 1e-5, setpoint=np.array([100., 100.]))

# Simulated system response
measured_value = np.array([10., 10.])  # Initial value
for i in range(1000):
    control_signal = pid.compute(measured_value)  # Get control output
    measured_value += control_signal * 0.1  # Apply control (simulating system response)
    print(f"Iteration {i+1}: Control Output = {control_signal}, Measured Value = {measured_value}")
    time.sleep(0.1)  # Simulate delay
