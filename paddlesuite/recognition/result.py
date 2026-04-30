class RecognitionResult:
    def __init__(self, labels=None, scores=None, file_name=None):
        self.labels = labels if labels is not None else []
        self.scores = scores if scores is not None else []
        self.file_name = file_name

    def __str__(self):
        res = ""
        if self.file_name:
            res += f"{self.file_name}:\t"

        scores_str = "[{}]".format(", ".join("{:.5f}".format(r) for r in self.scores))
        res += f"label(s): {self.labels}, score(s): {scores_str}"
        return res

    def __repr__(self):
        return f"RecognitionResult(file_name={self.file_name}, labels={self.labels}, scores={self.scores})"
