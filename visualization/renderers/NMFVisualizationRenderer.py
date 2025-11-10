from .FunkSVDVisualizationRenderer import FunkSVDVisualizationRenderer

class NMFVisualizationRenderer(FunkSVDVisualizationRenderer):
    def __init__(self, run_dir, explanations):
        super().__init__(run_dir, explanations)
        self.algorithm_name = "NMF"

