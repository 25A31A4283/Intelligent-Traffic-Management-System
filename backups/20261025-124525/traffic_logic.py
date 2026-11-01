import time

MIN_GREEN = 10
MAX_GREEN = 40
PER_VEHICLE = 2

class SignalController:
    def __init__(self, num_lanes):
        self.num_lanes = num_lanes
        self.green_lane = 0
        self.green_until = time.time() + MIN_GREEN

    def update(self, counts):
        now = time.time()
        if now >= self.green_until:
            order = sorted(range(self.num_lanes), key=lambda i: counts[i], reverse=True)
            nxt = order[0] if order[0] != self.green_lane else order[1]
            self.green_lane = nxt
            duration = min(MAX_GREEN, MIN_GREEN + PER_VEHICLE * counts[nxt])
            self.green_until = now + duration
        return self.green_lane, max(0, int(self.green_until - now))

    def emergency(self, lane, hold=15):
        self.green_lane = lane
        self.green_until = time.time() + hold