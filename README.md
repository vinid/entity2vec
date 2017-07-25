#Code file for the entity and type representation

File Listing:

###generate_model.py
Wrapper class around the gensim word2vec class for the creation of models.

###concatenate_models.py

Class used to concatenate the model that represent types witht the one that represent entities. **WARNING**: Since the content is stored in a textual file, it takes long to load in the word2vec object.

###evaluation.py

Class used to define the evaluation context and that takes in input the gold standard on which the models are evaluated

###play.ipynb

Sample notebook that can be used to test the models

###validate.py

Class that evaluates the model with different configuration and saves the output in the data/evaluation_result/folder