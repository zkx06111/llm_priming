'''
Date: 2022-12-23 16:19:51
LastEditors: Qingcheng Zeng
LastEditTime: 2022-12-27 20:24:42
FilePath: /comp_psycholing/GPT_zeroinference.py
'''

import openai
import pandas as pd
openai.api_key = "api_key_here"

def GPT_zeroinference(data_path,list_number,engine,response_path):
    # Filter data and list
    data = pd.read_csv(data_path)
    assert list_number == 1 or list_number == 2
    selected_data = data[data["List"]==list_number]

    # Merge story and critical question
    stories = selected_data["Story"].tolist()
    critical_questions = selected_data["Critical Question"].tolist()
    full_list = [' '.join(pair) for pair in zip(stories,critical_questions)]

    # generate response
    response_list = []
    for item in full_list:
        item += "? Reply in yes or no."
        completion = openai.Completion.create(engine=engine, prompt=item)
        text = completion.choices[0].text
        response_list.append(text.strip())
    
    with open(response_path,"w",encoding="UTF-8") as f:
        for i in response_list:
            f.write(f"{i}\n")

if __name__ == "__main__":
    GPT_zeroinference("priming_data/c1_target.csv",1,"text-davinci-003","response/zero_1_yesno.txt")