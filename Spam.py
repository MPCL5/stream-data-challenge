from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.lazy import SAMKNNClassifier
from skmultiflow.meta import AdaptiveRandomForestClassifier, StreamingRandomPatchesClassifier, \
    DynamicWeightedMajorityClassifier
import matplotlib

from Classifiers.MLP import MLPClassifier
from Data.Spam.getter import get_spam_stream

matplotlib.use("TkAgg")  # just for getting rid of SciView

# get stream data
print('Getting data stream...')
spam_stream = get_spam_stream()

# model
print('Initializing models...')
arf_cls = AdaptiveRandomForestClassifier()
sam_knn_cls = SAMKNNClassifier()
srp_cls = StreamingRandomPatchesClassifier()
dwn_cls = DynamicWeightedMajorityClassifier()
mlp_cls = MLPClassifier(hidden_layer_sizes=(16, 16))

# evaluator
# The prequential evaluation method or interleaved test-then-train method.
evaluator = EvaluatePrequential(show_plot=True,
                                max_samples=20000,
                                metrics=['accuracy'],
                                output_file='./Results/Spam.csv')

evaluator.evaluate(stream=spam_stream,
                   model=[arf_cls, mlp_cls, sam_knn_cls, dwn_cls],
                   model_names=['ARF', 'MLP', 'SAM_KNN', 'DWN_CLS'])
