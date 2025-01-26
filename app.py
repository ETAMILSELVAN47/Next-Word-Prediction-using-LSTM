import streamlit as st
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import pad_sequences
import numpy as np
import pickle

# load the model
model=load_model('model.h5')

# load the tokenizer
with open(file='tokenizer.pkl',mode='rb') as file:
    tokenizer=pickle.load(file=file)

def predict_next_word(model,tokenizer,text,max_sequence_len):
    # 1.convert text --> sequences
    token_list=tokenizer.texts_to_sequences([text])[0]
    if len(token_list)>=max_sequence_len:
        token_list=token_list[-(max_sequence_len-1):]        
    # 2.pad sequences
    token_list=pad_sequences(sequneces=token_list,maxlen=max_sequence_len-1,padding='pre')        
    # 3.prediction
    predicted=model.predict(token_list,verbose=1)
    # 4.find the maximum value index
    predicted_word_index=np.argmax(predicted,axis=1)
    # 5.Return the word
    reverse_word_index={v:k for k,v in tokenizer.word_index.items()}
    return reverse_word_index.get(predicted_word_index)



st.title('Next Word Prediction')
input_text=st.text_area('Enter the sequence of words')
if st.button('Predict Next Word'):
    next_word=predict_next_word(model=model,tokenizer=tokenizer,text=input_text,max_sequence_len=14)
    st.write(f'Next Word Prediction:{next_word}')


