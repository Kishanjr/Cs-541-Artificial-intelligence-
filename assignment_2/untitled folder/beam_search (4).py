# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
import string
import random
from data_utils import *
from rnn import *
import torch
import codecs
from tqdm import tqdm
import string

#Set GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Load vocabulary files
input_lang = torch.load('data-bin/fra.data')
output_lang = torch.load('data-bin/eng.data')

#Create and empty RNN model
encoder = EncoderRNN(input_size=input_lang.n_words, device=device)
attn_decoder = AttnDecoderRNN(output_size=output_lang.n_words, device=device)

#Load the saved model weights into the RNN model
encoder.load_state_dict(torch.load('model/encoder'))
attn_decoder.load_state_dict(torch.load('model/decoder'))

#Return the decoder output given input sentence 
#Additionally, the previous predicted word and previous decoder state can also be given as input
def translate_single_word(encoder, decoder, sentence, decoder_input=None, decoder_hidden=None, max_length=MAX_LENGTH, device=device):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence, device)
        input_length = input_tensor.size()[0]
        
        encoder = encoder.to(device)
        decoder = decoder.to(device)
        
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        if decoder_input==None:
            decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        else:
            decoder_input = torch.tensor([[output_lang.word2index[decoder_input]]], device=device) 
        
        if decoder_hidden == None:        
            decoder_hidden = encoder_hidden
        
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
        return decoder_output.data, decoder_hidden

#########################################################################################
#####Modify this function to use beam search to predict instead of greedy prediction#####
#########################################################################################
# def beam_search(encoder,decoder,input_sentence,beam_size=1,max_length=MAX_LENGTH):
#     decoded_output = []
    
#     #Predicted the first word
#     decoder_output, decoder_hidden = translate_single_word(encoder, decoder, input_sentence, decoder_input=None, decoder_hidden=None)
    
#     #Get the probability of all output words
#     decoder_output_probs = decoder_output.data
    
#     #Select the id of the word with maximum probability
#     idx = torch.argmax(decoder_output_probs)
	
#     #Convert the predicted id to the word
#     first_word = output_lang.index2word[idx.item()]
    
#     #Add the predicted word to the output list and also set it as the previous prediction
#     decoded_output.append(first_word)
#     previous_decoded_output = first_word
    
#     #Loop until the maximum length
#     for i in range(max_length):
    
#         #Predict the next word given the previous prediction and the previous decoder hidden state
#         decoder_output, decoder_hidden = translate_single_word(encoder, decoder, input_sentence, previous_decoded_output, decoder_hidden)
        
#         #Get the probability of all output words
#         decoder_output_probs = decoder_output.data
        
#         #Select the id of the word with maximum probability
#         idx = torch.argmax(decoder_output_probs)
        
#         #Break if end of sentence is predicted
#         if idx.item() == EOS_token:
#             break 
            
#         #Else add the predicted word to the list
#         else:
#             #Convert the predicted id to the word
#             selected_word = output_lang.index2word[idx.item()]
            
#             #Add the predicted word to the output list and update the previous prediction
#             decoded_output.append(selected_word)    
#             previous_decoded_output = selected_word
            
#     #Convert list of predicted words to a sentence and detokenize 
#     output_translation = " ".join(i for i in decoded_output)
    
#     return output_translation

import numpy as np
from math import log

def get_top_k(all_list, k):
    all_list = sorted(all_list, key = lambda x: x[-1], reverse = True)
    answer = all_list[:k]
    return answer

def get_sentence(temp_list):
  answers = []
  for h in temp_list:
    sentence_indexes = h[0]
    sentence = ' '.join([output_lang.index2word[index] for index in sentence_indexes])
    answers.append(sentence)
  return answers

def beam_search(encoder,decoder,input_sentence,beam_size , max_length=20):
    temp_list = [[[], [None, None], 1.0]]
    for i in range(max_length):
        all_list = []
        for temp in temp_list:
            prev_sentence, (previous_decoded_output, previous_decoded_hidden), prev_prob = temp
            last_word_index = -1 if len(prev_sentence) == 0 else prev_sentence[-1]
            if last_word_index == EOS_token:
              print('encountered end of token')
              return temp_list
            last_word = None if last_word_index == -1 else output_lang.index2word[last_word_index]
            if last_word == 'SOS':
              print('found start of token')
              continue
            decoder_output, decoder_hidden = translate_single_word(encoder, decoder, input_sentence, last_word, previous_decoded_hidden)
            decoder_output_probs = decoder_output.data
            values, decoder_output_probs_indices = decoder_output_probs.topk(beam_size)
            #print(decoder_output_probs)
            #print('shape', decoder_output_probs.shape)
            for i, prob in zip(decoder_output_probs_indices[0], values[0]):
                i = i.item()
                prob = prob.item()
                all_list.append([prev_sentence + [i], (decoder_output, decoder_hidden), prev_prob*prob])
        temp_list = get_top_k(all_list, beam_size)
    sentences = get_sentence(temp_list)
    return sentences

import pandas as pd

df1 = pd.read_csv('/content/drive/MyDrive/neural_machine_translation/data/test.eng', header = None)
df2 = pd.read_csv('/content/drive/MyDrive/neural_machine_translation/data/test.fra', header = None)


target_sentences = df1[0].tolist()
source_sentences = df2[0].tolist()


# target_sentences = ["i can speak a bit of french .",
#         "i ve bought some cheese and milk .",
#         "boy where is your older brother ?",
#         "i ve just started reading this book .",
#         "she loves writing poems ."]

# source_sentences = ["je parle un peu francais .",
#             "j ai achete du fromage et du lait .",
#             "garcon ou est ton grand frere ?",
#             "je viens justement de commencer ce livre .",
#             "elle adore ecrire des poemes ."]

target = codecs.open('example.txt','w',encoding='utf-8')

f = open("/content/drive/MyDrive/neural_machine_translation/data/test_beam_2.out", mode = "w")

beam_size = 7
for i,source_sentence in enumerate(source_sentences):
    
    target_sentence = normalizeString(target_sentences[i])
    input_sentence = normalizeString(source_sentence)
  
    hypothesis = beam_search(encoder, attn_decoder, input_sentence, beam_size=beam_size)
    # print(hypothesis)

    answers = []
    for h in hypothesis:
      sentence_indexes = h[0]
      sentence = ' '.join([output_lang.index2word[index] for index in sentence_indexes])
      answers.append(sentence)
    
    print(answers)
    # print("S-"+str(i)+": "+input_sentence)
    # print("T-"+str(i)+": "+target_sentence)
    # print("H-"+str(i)+": "+hypothesis)
    # print()

    f.write(answers+'\n')
    break
    # target.write(hypothesis+'\n')


  f.close()
  target.close()    
