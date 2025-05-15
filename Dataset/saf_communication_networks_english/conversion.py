import os
import string
import math
import random
import xml.etree.ElementTree as et
import jsonlines
import uuid

# set random seed for shuffling
random.seed(1)

def convert_xml_to_jsonl(path_to_dataset, dir, filename, train_split=None):
    """
    Utility function used for conversion of XML files from the dataset into JSON lines
    
    Params:
        path_to_dataset (string): path to the folder containing the dataset (in XML format)
        dir (string): name of the directory where the JSON lines file will be stored
        filename (string): name of the JSON lines file that will store the dataset
        train_split (float or None): if not None, defines which percentage of the dataset to use for the train and validation splits  
    
    Returns:
        None: the file is saved in JSON lines format in the specified location
    """
    data = []

    # loop through all files in directory
    for f in os.listdir(path_to_dataset):
        if f.endswith('.xml'):
            root = et.parse(os.path.join(path_to_dataset, f)).getroot()
            # get question
            question = root.find('questionText').text.replace('\n', ' ')
            # get reference and student answers
            ref_answers = [x for x in root.find('referenceAnswers')]
            student_answers = [x for x in root.find('studentAnswers')]

            if len(ref_answers) == 1:
                # get reference answer and clear all spaces
                ref_answer = ref_answers[0].text.strip()

                # loop through all student answers and store the appropriate fields in a list
                for answer in student_answers:
                    response = answer.find('response').text.strip()
                    score = float(answer.find('score').text)
                    feedback = answer.find('response_feedback').text.strip()
                    verification_feedback = answer.find('verification_feedback').text.strip()

                    # create dictionary with the appropriate fields
                    data.append({
                        'id': uuid.uuid4().hex, # generate unique id in HEX format
                        'question': question,
                        'reference_answer': ref_answer,
                        'provided_answer': response,
                        'answer_feedback': feedback,
                        'verification_feedback': verification_feedback,
                        'score': score
                    })

    if not os.path.exists(dir):
        print('Creating directory where JSON file will be stored\n')
        os.makedirs(dir)
    
    if train_split is None:
        with jsonlines.open(f'{os.path.join(dir, filename)}.jsonl', 'w') as writer:
            writer.write_all(data)
    else:
        # shuffle data and divide it into train and validation splits
        random.shuffle(data)
        train_data = data[: int(train_split * (len(data) - 1))]
        val_data = data[int(train_split * (len(data) - 1)) :]

        # write JSON lines file with train data
        with jsonlines.open(f'{os.path.join(dir, filename)}-train.jsonl', 'w') as writer:
            writer.write_all(train_data)
        
        # write JSON lines file with validation data
        with jsonlines.open(f'{os.path.join(dir, filename)}-validation.jsonl', 'w') as writer:
            writer.write_all(val_data)

if __name__ == '__main__':
    # convert communication networks dataset (english) to JSON lines
    convert_xml_to_jsonl(
        'data/training/english',
        'data/json',
        'saf-communication-networks-english',
        train_split=0.8)

    convert_xml_to_jsonl(
        'data/unseen_answers/english',
        'data/json',
        'saf-communication-networks-english-unseen-answers')

    convert_xml_to_jsonl(
        'data/unseen_questions/english',
        'data/json',
        'saf-communication-networks-english-unseen-questions')