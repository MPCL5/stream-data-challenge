from skmultiflow.evaluation import EvaluatePrequential
from Data.Spam.getter import spam_stream_getter
from skmultiflow.meta import AdaptiveRandomForestClassifier
import matplotlib

matplotlib.use("TkAgg")
# stream data
spam_stream = spam_stream_getter()

# model
arf = AdaptiveRandomForestClassifier()

# evaluator
evaluator = EvaluatePrequential(show_plot=True,
                                pretrain_size=200,
                                max_samples=20000)

# run!
evaluator.evaluate(stream=spam_stream, model=arf)
