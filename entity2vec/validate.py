from __future__ import division

import gc

import evaluation
import generate_model
from concatenate_models import *

size = [25, 50, 100, 200]
window = [2, 3, 5]
size_e = [100, 200]
window_e = [2, 3]
combinations_e = [(x, y) for x in window_e for y in size_e]
combinations = [(x, y) for x in window for y in size]

num_combinations = len(combinations)

for i in range(1, 10):
    with open("../data/evaluation_results/results", "a") as f:
        for window_e, size_e in combinations_e:
            print "Creating Entity Model", window_e, size_e
            model_e = generate_model.GenerateModel("../data/starting_data/entity_sentences.txt")
            model_e = model_e.fit("entity_vectors", _window=window_e, _size=size_e, _min_count=5, _sg=0)

            print "Evaluating"
            evaluator = evaluation.EvaluateModel(model_e,
                                                 _gold_standard="../data/starting_data/gold_standard", _type="entity")
            accuracy, total_number = evaluator.evaluate()
            print "Done"

            f.write("[ENTITY - MODEL]," + str(window_e) + "," + str(size_e) + ",none" + ",none" + "," + str(
                accuracy) + "," + str(total_number) + "\n")
            f.flush()
            for window_t, size_t in combinations:
                print "Creating Type Model", window_t, size_t
                # Generate Model for Types
                model_t = generate_model.GenerateModel("../data/starting_data/type_sentences.txt")
                model_t = model_t.fit("type_vectors", _window=window_t, _size=size_t, _min_count=5, _sg=0)
                # Load Models for Entities
                total_size = size_e + size_t
                print "Concatenating Models"
                # Concatenate Both Models
                ct = ConcatenateEntityToType(_entity_model=model_e,
                                             _type_model=model_t,
                                             _types_file="../data/starting_data/entity_to_type",
                                             _total_size=total_size)
                model = ct.concatenate()
                print "Evaluating"
                evaluator = evaluation.EvaluateModel(model,
                                                     _gold_standard="../data/starting_data/gold_standard", _type="type")
                accuracy, total_number = evaluator.evaluate()
                print "Done"
                f.write("[TYPE - MODEL]," + str(window_e) + "," + str(size_e) + "," + str(window_t) + "," + str(size_t)
                        + "," + str(accuracy) + "," + str(total_number) + "\n")
                f.flush()
        gc.collect()
