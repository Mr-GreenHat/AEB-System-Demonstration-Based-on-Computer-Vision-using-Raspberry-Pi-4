import time

class EgoVehicle:
    def __init__(self, initial_z=0.0):
        self.z = initial_z   # forward position (meters)
        self.velocity = 0.0  # m/s
        self.last_time = time.time()

    def set_speed(self, speed):
        # you control this externally
        self.velocity = speed

    def update(self):
        now = time.time()
        dt = now - self.last_time
        self.last_time = now

        # simple forward motion
        self.z += self.velocity * dt

        return self.z, dt