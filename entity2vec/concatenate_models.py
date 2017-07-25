import numpy as np
import gensim
# encoding=utf8
import sys

reload(sys)
sys.setdefaultencoding('utf8')

import pandas as pd


class ConcatenateEntityToType():
    def __init__(self, _types_file, _entity_model, _type_model, _total_size):
        """
        Initialize the class used to concatenate the entity model and the type model
        :param _types_file: file of dbpedia types used to match each entity with it is own type
        :param _entity_model: model of the entities
        :param _type_model: model of the types
        :param _total_size: total number of features of the model
        """
        self.type_files = _types_file
        self.entity_model = _entity_model
        self.type_model = _type_model
        self.total_size = _total_size

    def concatenate(self):
        """
        Method that concatenate the two models
        :return:
        """
        load_file_with_types = self.type_files #"~/Scaricati/file.stdout"

        df = pd.read_csv(load_file_with_types, header=None, delimiter=r"\s+", names = ["Subject", "Property", "Object", "Point"])
        df = df.drop('Property', 1)
        df = df.drop('Point', 1)

        df = df.set_index('Subject')

        dictionary = df['Object'].to_dict()

        with open("../data/produced_models/complete_model.txt", "w") as text_file:
    	    text_file.write(str(len(self.entity_model.wv.vocab)) + " " + str(self.total_size) + " " + "\n")
            for word, obj in self.entity_model.wv.vocab.items():
                try:
                    type = dictionary['<http://dbpedia.org/resource/' + word + '>']
                    type = type.replace(">", "")
                    type = type.replace("<http://dbpedia.org/ontology/", "")
                    type = type.replace("<http://www.w3.org/2002/07/", "")
                    type_array = self.type_model[type]
                except:
                    type_array = self.type_model["owl#Thing"]
                entity_array = self.entity_model[word]
                concatenated_array = np.concatenate((type_array, entity_array))
                concatenated_list = concatenated_array.tolist()
                string_to_save = ' '.join(map(str, concatenated_list))
                text_file.write(word +  " " + string_to_save + "\n")

        return gensim.models.KeyedVectors.load_word2vec_format("../data/produced_models/complete_model.txt")


