"""Expected numerical failures, distinct from infrastructure or CUDA OOM errors."""

class NumericalDivergence(RuntimeError):
    """An update produced nonfinite loss/gradients; retain its last valid checkpoint."""

    def __init__(self, update: int, reason: str):
        self.update = update
        super().__init__(f'Numerical divergence at update {update}: {reason}')
