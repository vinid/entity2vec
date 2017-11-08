## Code for the paper "Joint Learning of Entity and Type Embeddings for Analogical Reasoning with Entities" Federico Bianchi, Matteo Palmonari presented at Proceedings of the 1st Workshop on Natural Language for Artificial Intelligence co-located with 16th International Conference of the Italian Association for Artificial Intelligence (AI*IA 2017)

http://ceur-ws.org/Vol-1983/

# Code file for the entity and type representation

The folder with data, gold standard and a few already trained models, can be found [here](http://inside.disco.unimib.it/download/federico/entity2vec.tar.gz):


File Listing:

### generate_model.py
Wrapper class around the gensim word2vec class for the creation of models.

### concatenate_models.py

Class used to concatenate the model that represent types witht the one that represent entities. **WARNING**: Since the content is stored in a textual file, it takes long to load in the word2vec object.

### evaluation.py

Class used to define the evaluation context and that takes in input the gold standard on which the models are evaluated

### play.ipynb

Sample notebook that can be used to test the models

### validate.py

Class that evaluates the model with different configuration and saves the output in the data/evaluation_result/folder
