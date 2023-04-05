import math

from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.lazy import SAMKNNClassifier
from skmultiflow.meta import AdaptiveRandomForestClassifier, StreamingRandomPatchesClassifier, \
    DynamicWeightedMajorityClassifier
import matplotlib

from Classifiers.MLP import MLPClassifier
from Data.Hyperplane.getter import get_hyperplane_stream
from Data.Rialto.getter import get_rialto_stream
from Data.Sea.getter import get_sea_stream
from Data.Spam.getter import get_spam_stream

matplotlib.use("TkAgg")  # just for getting rid of SciView


def evaluate_stram(stream, output_file, samples=10000):
    # model
    print('Initializing models...')
    arf_cls = AdaptiveRandomForestClassifier()
    sam_knn_cls = SAMKNNClassifier()
    srp_cls = StreamingRandomPatchesClassifier()
    dwn_cls = DynamicWeightedMajorityClassifier()
    mlp_cls = MLPClassifier(hidden_layer_sizes=(16, 16))

    n_wait = math.ceil((samples - 200) / 20)

    # evaluator
    # The prequential evaluation method or interleaved test-then-train method.
    evaluator = EvaluatePrequential(show_plot=False,  # if u want to see the plot in real time just switch this to True
                                    max_samples=samples,
                                    metrics=['accuracy'],
                                    output_file=output_file,
                                    n_wait=n_wait)

    evaluator.evaluate(stream=stream,
                       model=[arf_cls, mlp_cls, sam_knn_cls, dwn_cls, srp_cls],
                       model_names=['ARF', 'MLP', 'SAM_KNN', 'DWN_CLS', 'SRP'])


evaluate_stram(get_spam_stream(), './Results/Spam.csv', 6214)
evaluate_stram(get_rialto_stream(), './Results/Rialto.csv', 82250)
evaluate_stram(get_sea_stream(), './Results/Sea.csv')
evaluate_stram(get_hyperplane_stream(), './Results/Hyperplane.csv')
