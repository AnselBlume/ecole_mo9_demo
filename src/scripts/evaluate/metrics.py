from typing import Any, Union
from model.concept import ConceptKB
from kb_ops.predict import PredictOutput

class Metric:
    def __call__(prediction_dict: dict[str, Any]):
        pass

    @staticmethod
    def extract_predicted_label(prediction: Union[PredictOutput, list[PredictOutput]]):
        if isinstance(prediction, list): # List of PredictOutput
            final_predict_output = prediction[-1]
            predicted_label = final_predict_output.predicted_label
        else: # Instance of PredictOutput
            predicted_label = prediction.predicted_label

        return predicted_label

class MetricDict:
    def __init__(self, metrics: dict[str, Metric]):
        self.metrics = metrics

    def __call__(self, prediction_dict: dict[str, Any]):
        for metric in self.metrics.values():
            metric(prediction_dict)

    def compute(self):
        return {name : metric.compute() for name, metric in sorted(self.metrics.items(), key=lambda x: x[0])}

class FlatAccuracy(Metric):
    def __init__(self):
        self.n_correct = 0
        self.total = 0

    def __call__(self, prediction_dict: dict[str, Any]):
        label = prediction_dict['label']
        predicted_label = self.extract_predicted_label(prediction_dict['prediction'])

        is_correct = label == predicted_label
        self.n_correct += is_correct
        self.total += 1

    def compute(self):
        return self.n_correct / self.total

class HierarchicalAccuracy(Metric):
    def __init__(self, conceptKB: ConceptKB):
        self.conceptKB = conceptKB

    def __call__(self, prediction_dict: dict[str, Any]):
        raise NotImplementedError('Hierarchical accuracy not yet implemented')