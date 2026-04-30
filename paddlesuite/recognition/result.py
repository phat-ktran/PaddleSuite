class RecognitionResult:
    def __init__(self, labels=None, scores=None):
        self.labels = labels if labels is not None else []
        self.scores = scores if scores is not None else []

    def __repr__(self):
        return f"RecognitionResult(labels: {self.labels}, scores: {self.scores})"
