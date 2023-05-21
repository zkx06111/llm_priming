'''
Date: 2022-12-24 17:15:12
LastEditors: Qingcheng Zeng
LastEditTime: 2023-01-13 17:02:20
FilePath: /comp_psycholing/GPT_incontext.py
'''

import json
import requests
API_TOKEN = ''
API_URL = "https://api-inference.huggingface.co/models/google/flan-ul2"
headers = {"Authorization": "Bearer {API_TOKEN}"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

import pandas as pd
import random
from sklearn.utils import shuffle

names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller']
countries = ['England', 'the United States', 'Australia']
titles = ['Mr.', 'Ms.', 'Dr.']
genders = ['He', 'She']

def gen_prompt(seed):
    name = names[seed % len(names)]
    country = countries[seed % len(countries)]
    title = titles[seed % len(titles)]
    
    random.seed(seed * 19260817)
    if title == 'Dr.':
        gender = genders[random.randint(0, 1)]
    else:
        gender = genders[seed % len(titles)]
    
    prefix = f'{title} {name} is a native English speaker living in {country}. {gender} is asked in a psycholinguistic experiment to answer the following questions.\n'
    
    suffix = f"{title} {name}'s answer is "
    
    return prefix, suffix
    

def GPT_incontext(pref, suf,exposure_data_path,target_data_path,exposure_type,target_data_list,reverse,engine,temperature,response_path):
    random.seed(5)

    # define prompt
    prompt = pref

    # read exposure data
    exposure_data = pd.read_csv(exposure_data_path)
    exposure_data_by_type = exposure_data[exposure_data["Type filler"] == exposure_type]

    # read target data
    target_data = pd.read_csv(target_data_path)
    target_data_by_list = target_data[target_data["List"] == target_data_list]

    # filter 20 items from exposure data
    # first filter item by name
    ana_exposure = exposure_data_by_type[exposure_data_by_type["Stimulus"].str.startswith("Ana")]
    matt_exposure = exposure_data_by_type[exposure_data_by_type["Stimulus"].str.startswith("Matt")]
    liz_exposure = exposure_data_by_type[exposure_data_by_type["Stimulus"].str.startswith("Liz")]
    will_exposure = exposure_data_by_type[exposure_data_by_type["Stimulus"].str.startswith("Will")]

    # then random sample 5 from each as exposure
    ana_select = ana_exposure.sample(n=5,random_state=3407)
    ana_remaining = ana_exposure.drop(ana_select.index)
    matt_select = matt_exposure.sample(n=5,random_state=3407)
    matt_remaining = matt_exposure.drop(matt_select.index)
    liz_select = liz_exposure.sample(n=5,random_state=3407)
    liz_remaining = liz_exposure.drop(liz_select.index)
    will_select = will_exposure.sample(n=5,random_state=3407)
    will_remaining = will_exposure.drop(will_select.index)

    # concatenate 20 together as the exposure
    previous_exposure = pd.concat([ana_select,matt_select,liz_select,will_select])
    previous_exposure = shuffle(previous_exposure,random_state=3407)
    previous_exposure.reset_index(drop=True,inplace=True)
    other_exposure = pd.concat([ana_remaining,matt_remaining,liz_remaining,will_remaining])
    other_exposure = shuffle(other_exposure,random_state=3407)
    other_exposure.reset_index(drop=True,inplace=True)

    # try to randomize other exposure and target question
    other_exposure_stories = other_exposure["Stimulus"].tolist()
    target_data_exposure_stories = target_data_by_list["Story"].tolist()
    follwing_32_stories = other_exposure_stories + target_data_exposure_stories
    random.shuffle(follwing_32_stories)
    reverse_32_stories = list(reversed(follwing_32_stories))

    if reverse == True:
        follwing_32_stories = reverse_32_stories
    
    # self in context inference
    for _,info in previous_exposure.iterrows():
        story = info["Stimulus"]
        prompt += story + "\n"
        Q1 = info["Q1"] + "?"
        option1 = info["answer1"]
        option2 = info["answer2"]
        Q2 = info["Q2"] + "?"
        option3 = info["answer3"]
        option4 = info["answer4"]

        prompt += "Question1: " + Q1 + "\n(A)" + option1 + "\n(B)" + option2 + "\nAnswer:"
        output = query({
            "inputs": prompt,
            "temperature": temperature,
            "max_new_tokens": 5,
        })
        
        text = output[0]['generated_text']
        prompt += text + "\n"

        prompt += "Question2: " + Q2 + "\n(A)" + option3 + "\n(B)" + option4 + "\nAnswer:"
        output = output = query({
            "inputs": prompt,
            "temperature": temperature,
            "max_new_tokens": 5,
        })
        text = output[0]['generated_text']
        prompt += text + "\n"

        prompt += "\n"
    
    for story in follwing_32_stories:
        if story in other_exposure_stories:
            stim = other_exposure.loc[other_exposure["Stimulus"] == story]["Stimulus"].to_list()[0]
            prompt += stim + "\n"
            Q1 = other_exposure.loc[other_exposure["Stimulus"] == story]["Q1"].to_list()[0] + "?"
            option1 = other_exposure.loc[other_exposure["Stimulus"] == story]["answer1"].to_list()[0]
            option2 = other_exposure.loc[other_exposure["Stimulus"] == story]["answer2"].to_list()[0]
            Q2 = other_exposure.loc[other_exposure["Stimulus"] == story]["Q2"].to_list()[0] + "?"
            option3 = other_exposure.loc[other_exposure["Stimulus"] == story]["answer3"].to_list()[0]
            option4 = other_exposure.loc[other_exposure["Stimulus"] == story]["answer4"].to_list()[0]

            prompt += "Question1: " + Q1 + "\n(A)" + option1 + "\n(B)" + option2 + f"\n{suf}"
            output = query({
                "inputs": prompt,
                "temperature": temperature,
                "max_new_tokens": 5,
            })
            text = output[0]['generated_text']
            prompt += text + "\n"

            prompt += "Question2: " + Q2 + "\n(A)" + option3 + "\n(B)" + option4 + f"\n{suf}"
            output = output = query({
                "inputs": prompt,
                "temperature": temperature,
                "max_new_tokens": 5,
            })
            text = output[0]['generated_text']
            prompt += text + "\n"

            prompt += "\n"
        else:
            stim = target_data_by_list.loc[target_data_by_list["Story"] == story]["Story"].to_list()[0]
            prompt += stim + "\n"
            Q1 = target_data_by_list.loc[target_data_by_list["Story"] == story]["Critical Question"].to_list()[0] + "?"
            option1 = "Yes"
            option2 = "No"
            Q2 = target_data_by_list.loc[target_data_by_list["Story"] == story]["Comprehension"].to_list()[0] + "?"
            option3 = target_data_by_list.loc[target_data_by_list["Story"] == story]["answer1"].to_list()[0]
            option4 = target_data_by_list.loc[target_data_by_list["Story"] == story]["answer2"].to_list()[0]




            prompt += "Question1: " + Q1 + "\n(A)" + option1 + "\n(B)" + option2 + f"\n{suf}"
            output = query({
                "inputs": prompt,
                "temperature": temperature,
                "max_new_tokens": 5,
            })
            text = output[0]['generated_text']
            prompt += text + "\n"

            prompt += "Question2: " + Q2 + "\n(A)" + option3 + "\n(B)" + option4 + f"\n{suf}"
            output = query({
                "inputs": prompt,
                "temperature": temperature,
                "max_new_tokens": 5,
            })
            text = output[0]['generated_text']
            prompt += text + "\n"

            prompt += "\n"
    
    with open(response_path,"w") as f:
        f.write("List: " + str(target_data_list) + "\n" + "Reverse: " + str(reverse) + "\n" + "Engine: " + engine + "\n" + "Temperature " + str(temperature) + "\n")
        f.write(prompt)


if __name__ == "__main__":
    import os
    list_number = int(os.environ['LN'])
    reverse = bool(os.environ['REV'])
    seed = int(os.environ['SEED'])
    #temperature = float(os.environ['TEMP'])
    pref, suf = gen_prompt(seed)
    GPT_incontext(pref, suf,"priming_data/c1_exposure.csv","priming_data/c1_target.csv",
    "Subject_exposure",list_number,reverse,"text-davinci-003",1,"response/Prompt_Experiment_1/1a_temp/Subject/incontext_{}_{}_{}_{}.txt".format(list_number,reverse,1, seed))