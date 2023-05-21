'''
Date: 2022-12-09 16:05:28
LastEditors: Qingcheng Zeng
LastEditTime: 2022-12-12 14:14:20
FilePath: /comp_psycholing/utils.py
'''

from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging


predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")

def generate_content_object_questions(sentence,predictor,sentence_type):
    sentences = list(filter(None, sentence.split(". ")))
    bg_sentence = sentences[0]
    rf_sentence = sentences[1]
    
    # retrieve two people
    bg_sentence_words = bg_sentence.split(" ")
    subject = bg_sentence_words[0]
    object = bg_sentence_words[-1]

    
    predictor_output = predictor.predict(sentence=sentence)
    main_verb_tags = predictor_output["verbs"][0]["tags"]
    words = predictor_output["words"]

    # retrieve ARG0
    start_0 = main_verb_tags.index("B-ARG0")
    if "I-ARG0" in main_verb_tags:
        end_0 = main_verb_tags.index("I-ARG0")
        ARG0 = " ".join(words[start_0:end_0+1])
    else:
        ARG0 = words[start_0]
    
    # retrieve verb
    start_1 = main_verb_tags.index("B-V")
    if "I-V" in main_verb_tags:
        end_1 = main_verb_tags.index("I-V")
        Verb = " ".join(words[start_1:end_1+1])
    else:
        Verb = words[start_1]

    # retrieve ARG1
    start_2 = main_verb_tags.index("B-ARG1")
    if "I-ARG1" in main_verb_tags:
        end_2 = main_verb_tags.index("I-ARG1")
        ARG1 = " ".join(words[start_2:end_2+1])
    else:
        ARG1 = words[start_2]

def generate_content_action_questions():
    question = "What did they do?"
    return question

if __name__ == "__main__":
    generate_content_action_questions("Matt played the piano with Liz.",predictor)