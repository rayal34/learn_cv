class ZeroOneScale:
    def __init__(self, min_val: float, max_val: float):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, img):
        return (img - self.min_val) / (self.max_val - self.min_val)
