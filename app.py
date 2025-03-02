import flask
import string
from flask import Flask,request ,jsonify
import nltk
import pickle
import numpy as np
import sklearn
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

print("numpy:", np.__version__)
print("sklearn:", sklearn.__version__)
print("flask:",flask.__version__)
print("nltk:",nltk.__version__)
 

model =pickle.load(open('model.pkl','rb'))
vectorizer =pickle.load(open('vectorizer.pkl','rb'))
 
def transform_text (text):
     
    text = text.lower() 
    text = nltk.word_tokenize(text) 
    removedSC = list()
    for i in text:
        if i.isalnum():
            removedSC.append(i)
    text = removedSC[:]
    removedSWPC = list()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            removedSWPC.append(i)
    text = removedSWPC[:]
    ps = PorterStemmer()
    stemmed = list()
    for i in text:
        stemmed.append(ps.stem(i))
    
    text = stemmed[:]
    
    return " ".join(text)




app =Flask(__name__)

@app.route('/')
def home():
  return "Hello World"

@app.route('/predict',methods=['POST'])
def predict():
   msg=request.form.get('msg')
   transformed_msg=transform_text(msg)
   vector_input=vectorizer.transform([transformed_msg])
   print(vector_input)
   res=model.predict(vector_input)[0]
   ans='Not Spam'
   if(res==1):
      ans='Spam'
   result={'msg':msg,'result':ans}

   return jsonify(result)

if __name__ == '__main__':
  app.run(debug=True)