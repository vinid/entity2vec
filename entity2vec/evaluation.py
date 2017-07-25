from __future__ import division
import pandas as pd
import gensim
import sys
reload(sys)

sys.setdefaultencoding('utf8')

from gensim.models import word2vec


class EvaluateModel:
    def __init__(self, _model, _gold_standard, _type):
        """
        Class used to do the evaluation of the model
        :param _model: the model to be  evaluated
        :param _gold_standard: the gold standard with the analogies
        :param _type: entity or entity+type model
        """
        self.model = _model
        self.gold_standard = _gold_standard
        self.type = _type

    def evaluate(self):
        """
        Evaluation method
        :return:
        """
        df = pd.read_csv(self.gold_standard, header=None, delimiter=r"\s+", names=["First", "Second", "Third", "Fourth"])
        correct = 0
        uncorrect = 0
        total_number = 0
        for index, row in df.iterrows():
            # print row['First'], row['Second']
            try:
                second = row['Second']
                first = row['First']
                third = row['Third']
                fourth = row['Fourth']

                #there's a problem with loading non-utf8 from text
                if self.type == "type":
                    second = second.decode('utf-8')
                    first = first.decode('utf-8')
                    third = third.decode('utf-8')
                    fourth = fourth.decode('utf-8')
                predicted = self.model.wv.most_similar_cosmul(positive=[second, third], negative=[first])[0][0]
                if fourth == predicted:
                    # print "correct"
                    correct = correct + 1
                else:
                    uncorrect = uncorrect + 1
                    # print "uncorrect", first, second, third, fourth, predicted
                total_number = total_number + 1
            except Exception as e:
                #print(e)
		continue

        return correct/total_number, total_number




