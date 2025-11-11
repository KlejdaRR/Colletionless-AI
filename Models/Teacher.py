class Teacher:

    def __init__(self):
        self.calls_remaining = 1000

    def teach(self, x, y_true):
        if self.calls_remaining > 0:
            self.calls_remaining -= 1
            return y_true
        return None