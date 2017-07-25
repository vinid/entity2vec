from gensim.models import word2vec
# encoding=utf8
import sys
reload(sys)
sys.setdefaultencoding('utf8')


class GenerateModel:
    def __init__(self, file_name):
        """
        Initialize the class with the filename from which data must be extracted from
        :param file_name:
        """
        self.file_name = file_name

    def fit(self, save_file_name, _size=100, _window=5, _min_count=5, _workers=7, _sg=0, _iter=15):
        """
        Reads the file line by line and then feeds a word2vec model with given parameters. Produced models are saved
        in a local directory
        :param save_file_name:
        :param _size: size of the resulting space
        :param _window: window span considered for each target word
        :param _min_count:
        :param _workers:
        :param _sg: skip-gram or cbow, default is cbow
        :return:
        """
        if not save_file_name:
            save_file_name = self.file_name + ".model"

        with open(self.file_name) as f:
            content = f.readlines()

        sentences = [x.strip() for x in content]

        model = word2vec.Word2Vec([s.encode('utf-8').split() for s in sentences], size=_size,
                                  window=_window, min_count=_min_count,
                                  workers=_workers, sg=_sg, iter=_iter)

        model.save("../data/produced_models/" + save_file_name)

        return model
