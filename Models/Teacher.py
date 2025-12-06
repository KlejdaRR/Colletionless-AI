class Teacher:

    def __init__(self):
        self.calls_remaining = 1000 # The teacher will only provide correct answers 1000 times
        # Forces the system to become self-sufficient rather than relying endlessly on external help

    # When a student asks for help:
    def teach(self, x, y_true):
        # Step 1: Checking if teacher still has "energy"
        if self.calls_remaining > 0:
            # Step 2: Using up one teaching opportunity
            self.calls_remaining -= 1
            # Step 3: Providing the correct answer
            return y_true
        # Step 4: If exhausted, provide no help
        return None

    # Teacher's Lifecycle:
    # Steps 0-1000: Teacher actively provides y_true labels
    # After 1000 calls: Teacher returns None (no more help)
    # Result: Students must rely on each other