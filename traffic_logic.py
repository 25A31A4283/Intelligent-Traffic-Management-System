import time

MIN_GREEN = 10
MAX_GREEN = 40
PER_VEHICLE = 2

class SignalController:
    def __init__(self, num_lanes):
        if num_lanes < 1:
            raise ValueError("num_lanes must be at least 1")
        self.num_lanes = num_lanes
        self.green_lane = 0
        self.green_until = time.time() + MIN_GREEN

    def update(self, counts):
        if len(counts) != self.num_lanes:
            raise ValueError(f"Expected {self.num_lanes} lane counts.")
        now = time.time()
        if now >= self.green_until:
            nxt = (self.green_lane + 1) % self.num_lanes
            self.green_lane = nxt
            duration = min(MAX_GREEN, MIN_GREEN + PER_VEHICLE * counts[nxt])
            self.green_until = now + duration
        return self.green_lane, max(0, int(self.green_until - now))

    def emergency(self, lane, hold=15):
        if not 0 <= lane < self.num_lanes:
            raise ValueError(f"lane must be between 0 and {self.num_lanes - 1}")
        if hold <= 0:
            raise ValueError("hold must be greater than 0")
        self.green_lane = lane
        self.green_until = time.time() + hold