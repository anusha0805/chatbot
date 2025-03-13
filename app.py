from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pickle

app = Flask(__name__)

# Load the model and tokenizer
model = load_model('chatbot_model.h5')

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Define the maximum sequence length (same as training)
max_seq_length = 20  # Replace with your actual max length

# Decode the response from the model
def decode_sequence(input_seq):
    states_value = model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer.word_index['<start>']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens = model.predict([target_seq, states_value])
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = tokenizer.index_word[sampled_token_index]
        decoded_sentence += ' ' + sampled_word

        if sampled_word == '<end>' or len(decoded_sentence) > max_seq_length:
            stop_condition = True

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

    return decoded_sentence.replace('<end>', '').strip()

# API endpoint for chatbot
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    input_seq = tokenizer.texts_to_sequences([f"<start> {user_message} <end>"])
    input_seq = tf.keras.preprocessing.sequence.pad_sequences(input_seq, maxlen=max_seq_length, padding='post')
    response = decode_sequence(input_seq)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
