from fedpost.evaluation.base import EvalResult

__all__ = ["EvalResult", "PaperBenchmarkEvaluator"]


def __getattr__(name):
    if name == "PaperBenchmarkEvaluator":
        from fedpost.evaluation.paper import PaperBenchmarkEvaluator

        return PaperBenchmarkEvaluator
    raise AttributeError(name)
