import requests
import json

def labelling_test_set():

    #Create dict with labels. 

    labels = {
        "politics": [],
        "health": [],
        "finance": [],
        "travel": [],
        "food": [],
        "education": [],
        "environment": [],
        "fashion": [],
        "science": [],
        "sports": [],
        "technology": [],
        "entertainment": []
    }

    #Introduction prompt

    intro_prompt = """You're an assistant that can perfectly label sentences you're given, with one the following labels :
    [politics;health;finance;travel;food;education;environment;fashion;science;sports;technology;entertainment]
    Your answers MUST always be a SINGLE WORD corresponding to one label of the previous list that best describes the following sentence :"""


    #create a list of prompts from a .txt file 
    with open('data/test_shuffle.txt', 'r') as file:
        prompts = file.readlines()


    url = "http://51.178.94.112:10464/api/generate"
    auth = ("angry_faraday", "mZQNHN1X7U")
    headers = {"Content-Type": "application/json"}

    for prompt in prompts:
        print(prompt)
        data = {
            "model": "mixtral:latest",
            "prompt": intro_prompt + prompt,
            "stream": False,
        }
        

        response = requests.post(url, json=data, headers=headers, auth=auth)
        
        # Check if the request was successful
        if response.status_code == 200:
            response_text = response.json()["response"]
            response_text = response_text.replace(" ", "").lower()
            
            print(response_text)
            
            if response_text in labels.keys():
                labels[response_text].append(prompt)
                print(labels[response_text])
            

        else:
            print("Error:", response.text)

        # Write labels dictionary to a JSON file
        with open('data/test_with_labels.json', 'w') as file:
            json.dump(labels, file)

    

if __name__ == "__main__":
    labelling_test_set()
