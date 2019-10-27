import numpy as np
from flask import Flask, request, render_template
import pickle
from lib.sentence2vec import Sentence2Vec
import warnings
import nltk
nltk.download('stopwords')
nltk.download('punkt')

warnings.filterwarnings("ignore")

app = Flask(__name__)
model = pickle.load(open('data/job_titles.model', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    model = Sentence2Vec('./data/job_titles.model')
    # Pickle the Sentences from document
    axis_array = pickle.load(open('AxisWellCleanlist.data', 'rb'))
    query_similarity = [x for x in request.form.values()]
    query = query_similarity[0]
    percent = int(query_similarity[1])
    similarity_axis_array_keywords = np.zeros((len([1]), len(axis_array)))
    for ind, ele in enumerate(axis_array):
        similarity_axis_array_keywords[0, ind] = model.similarity(query, ele)
    string_result2 = ""
    print(query,percent)
    print(similarity_axis_array_keywords)
    stop_at = 0
    for i, val in enumerate([index for index, value in enumerate(similarity_axis_array_keywords[0,:]) if value > (percent/100)]):
        string_result2 += axis_array[val] + '\n'
        stop_at += 1
        if(stop_at==3):
            break
    print(string_result2)
    list_string_result2 = string_result2.split('\n')
    if string_result2 == "":
        string_result2 = "Monkeys are working hard to get your request..."
    return render_template('index.html', prediction_text=list_string_result2)


if __name__ == "__main__":
    app.run(debug=True)
