#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random 
import pandas as pd
import re

from transformers import AutoModelWithLMHead, AutoTokenizer, AutoModelForCausalLM
import torch

def talking_gpt2_personas(test_dict, model, name1, name2, num_lines, t):
  """
  This function takes are arguments the below inputs and processes a dialogue
  between two characters from Sienfield of a choosing.

  Inputs:
    (DataFrame) test_dict: lines from Sienfield that the gpt2s were not trained on 
    (gpt2)      model: gpt2 model with personas
    (str)       name1: name of Character A (capitalize for distinction from model)
    (str)       name2: name of Character B (capitalize for distinction from model)
    (int)       num_lines: number of lines of dialouge to generate
    (tokenizer) t
  Outputs: 
    Dialogue iteraction between Character A and Character B using the 2 gpt2 
    models.
    (list)      references: lines that actually follow the initial input
    (list)      candidates: lines that are predicted to follow the initial input
  """

  test_personas = {"JERRY": ["your persona: i make a living telling jokes. \nyour persona: i like to watch sports. \nyour persona: i am a fan of cartoons. \nyour persona: women find me quirky and charming."],
                  "GEORGE": ["your persona: i am vulnerable and slightly neurotic. \nyour persona: i am short, stocky, and slow-witted. \nyour persona: i am dishonest. \nyour persona: i have eccentric behavior."],
                  "ELAINE": ["your persona: i am intelligent. \nyour persona: i am funny. \nyour persona: i am assertive and confident. \nyour persona: i am edgy and superficial."],
                  "KRAMER": ["your persona: i have eccentric behavior. \nyour persona: i am unemployed. \nyour persona: i always tell the truth. \nyour persona: i have interesting ideas."]
                  }

  tokenizer = t

  # Initalize lists to keep track of references and candidates
  references = []
  candidates = []
  
  # Select only lines from Character A from the test DataFrame where Character B 
  # responds
  test_dict["next_char"] = test_dict["Character"].shift(-1)
  charAB_df = test_dict.loc[(test_dict['Character']==name1) & (test_dict["next_char"]==name2) ]
    
  # Randomly select an input statement for the dialogue from Character A's lines
  # This is supplemented with Character B's persona 
  rand_idx = random.choice(list(charAB_df.index))
  input = test_personas[name2][0] + '\n ' + charAB_df['Dialogue'][rand_idx]
  #print(input)
  print(f"{name1}: {charAB_df['Dialogue'][rand_idx]}")

  # Append to references list the next 'step' number of lines 
  i = rand_idx
  for step in range(num_lines):
    i += 1
    references.append(test_dict['Dialogue'][i].split(' '))

  for step in range(num_lines):
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    #new_user_input_ids = tokenizer.encode(input + tokenizer.eos_token, return_tensors='pt')

    # append the new user input tokens to the chat history
    bot_input_ids = tokenizer.encode(input + tokenizer.eos_token, return_tensors='pt')

    # generate a response while limiting the total chat history to 1000 tokens, 
    chat_history_ids = model.generate(bot_input_ids, max_length=1000,pad_token_id=tokenizer.eos_token_id,  
      no_repeat_ngram_size=3, do_sample=True, top_k=100, top_p=0.7, temperature = 0.8)
    
    # At first step & even steps, use the input from the test dictionary to generate a new line
    # for Character B
    if step % 2 == 0:
      print("{}: {}".format(name2, tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True))) 
      candidates.append(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True).split(' '))
      input = test_personas[name1][0] + '\n ' + tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    else:
      print("{}: {}".format(name1, tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True))) 
      candidates.append(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True).split(' '))
      input = test_personas[name2][0] + '\n ' + tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
  
  return references, candidates

def talking_gpt2_2char(test_dict, model1, model2, name1, name2, num_lines, tokenizer1, tokenizer2):
  """
  This function takes are arguments the below inputs and processes a dialogue
  between two characters from Sienfield of a choosing.

  Inputs:
    (DataFrame)  test_dict: lines from Sienfield that the gpt2s were not trained on 
    (gpt2)       model1: gpt2 model for Character A
    (gpt2)       model2: gpt2 model for Character B
    (str)        name1: name of Character A (capitalize for distinction from model)
    (str)        name2: name of Character B (capitalize for distinction from model)
    (int)        num_lines: number of lines of dialouge to generate
    (tokenizer)  tokenizer1: gpt2 tokenizer for Character A
    (tokenizer)  tokenizer2: gpt2 tokenizer for Character B
  Outputs: 
    Dialogue iteraction between Character A and Character B using the 2 gpt2 
    models.
    (list)      references: lines that actually follow the initial input
    (list)      candidates: lines that are predicted to follow the initial input
  """

  # Initalize lists to keep track of references and candidates
  references = []
  candidates = []
  
  # Select only lines from Character A from the test DataFrame where Character B 
  # responds
  test_dict["next_char"] = test_dict["Character"].shift(-1)
  charAB_df = test_dict.loc[(test_dict['Character']==name1) & (test_dict["next_char"]==name2) ]
    
  # Randomly select an input statement for the dialogue from Character A's lines
  rand_idx = random.choice(list(charAB_df.index))
  input = charAB_df['Dialogue'][rand_idx]
  print(f"{name1}: {input}")

  # Append to references list the next 'step' number of lines 
  i = rand_idx
  for step in range(num_lines):
    i += 1
    references.append(test_dict['Dialogue'][i].split(' '))

  for step in range(num_lines):
    # At first step, use the input from the test dictionary to generate a new line
    # for Character B
    if step == 0:
        # encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input_ids = tokenizer2.encode(input + tokenizer2.eos_token, return_tensors='pt')

        # append the new user input tokens to the chat history
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

        # generate a response while limiting the total chat history to 1000 tokens, 
        chat_history_ids = model2.generate(bot_input_ids, max_length=1000,pad_token_id=tokenizer2.eos_token_id,  
          no_repeat_ngram_size=3, do_sample=True, top_k=100, top_p=0.7, temperature = 0.8)
        print("{}: {}".format(name2, tokenizer2.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))  # Catchall so character doesnt' say their own name, they'll say "Tal" instead
      
        # Save the output are input for the next step
        last = tokenizer2.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        # Apppend to candidate list the output 
        candidates.append(last.split(' '))

    # At an odd number step, Character A is responding
    elif step% 2 != 0: 
        # encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input_ids = tokenizer1.encode(last + tokenizer1.eos_token, return_tensors='pt')

        # append the new user input tokens to the chat history
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

        # generate a response while limiting the total chat history to 1000 tokens, 
        chat_history_ids = model1.generate(bot_input_ids, max_length=1000,pad_token_id=tokenizer1.eos_token_id,  
            no_repeat_ngram_size=3, do_sample=True, top_k=100, top_p=0.7, temperature = 0.8)
        print("{}: {}".format(name1, tokenizer1.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))
        
        last = tokenizer1.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        # Apppend to candidate list the output 
        candidates.append(last.split(' '))

    # At an even number step, Character B is responding 
    else:
        # encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input_ids = tokenizer2.encode(last + tokenizer2.eos_token, return_tensors='pt')

        # append the new user input tokens to the chat history
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

        # generate a response while limiting the total chat history to 1000 tokens, 
        chat_history_ids = model2.generate(bot_input_ids, max_length=1000,pad_token_id=tokenizer2.eos_token_id,  
            no_repeat_ngram_size=3, do_sample=True, top_k=100, top_p=0.7, temperature = 0.8)
        print("{}: {}".format(name2, tokenizer2.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))
        
        last = tokenizer2.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        # Apppend to candidate list the output 
        candidates.append(last.split(' '))

  return references, candidates

def clean_df(df):
  #Data cleaning

  # No empty series
  df = df[df['Dialogue'].notnull()]

  # Remove annotations describing actions
  df['Dialogue'] = df['Dialogue'].apply(lambda x: re.sub(r'\([^)]*\)', '', str(x)))

  # Remove annotations describing actions
  df['Dialogue'] = df['Dialogue'].apply(lambda x: re.sub(r'\[[^)]*\]', '', str(x)))

  # Remove ` mark
  df['Dialogue'] = df['Dialogue'].apply(lambda x: re.sub(r'\`', '', str(x)))

  # Remove * mark
  df['Dialogue'] = df['Dialogue'].apply(lambda x: re.sub(r'\*', '', str(x)))

  # Fix i'm to I'm
  df['Dialogue'] = df['Dialogue'].apply(lambda x: re.sub(r'([i]\'[m]+)', 'I\'m', str(x)))

  # Convert anything with more than two or more '.' to ';'
  df['Dialogue'] = df['Dialogue'].apply(lambda x: re.sub(r'\.{2,}', ' ; ', str(x)))

  # "For word based model which is commonly used in many systems, we treat punctuation as it's own token."
  #    https://ai.stackexchange.com/questions/17326/how-to-use-lstm-to-generate-a-paragraph
  import string
  def _normalize_text(text):
    text = text.lower()
    text = re.sub("[%s]" % re.escape(string.punctuation), r" \g<0> ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text 
  #Isolate single punctuation marks to become tokens 
  df['Dialogue'] = df['Dialogue'].apply(lambda x: _normalize_text(x))
  
  return df
