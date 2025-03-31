# making request
import argparse
import pandas as pd
import os
import json
import re
import ast



def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        default = './datasets',
        type=str,
        help="the path of data"
    )

    parser.add_argument(
        "--data_filename",
        default = None,
        type=str,
        help="If specified, only process the file with the given name"
    )
    
    parser.add_argument(
        "--raw_output",
        default = './GPT4_gen/raw_generation.jsonl',
        type=str,
        help="raw output path"
    )
    parser.add_argument(
        "--filtered_output",
        default = './GPT4_gen/final_generation.jsonl',
        type=str,
        help="filtered output"
    )

    parser.add_argument(
        "--model",
        default = 'gpt-4o',
        choices=['gpt-4o','gpt-4-turbo-2024-04-09', 'gpt-4-turbo', 'gpt-4-0125-preview', 'gpt-4-turbo-preview', 'gpt-4-1106-preview', "gpt-3.5-turbo-0125", "gpt-3.5-turbo"],
        type=str,
        help="the model used for generation"
    )

    parser.add_argument(
        "--max_tokens",
        default = 512,
        type=int,
        help="number of max tokens for generation"
    )

    parser.add_argument(
        "--temperature",
        default = .99,
        type=float,
        help="temperature to adjust diversity"
    )

    parser.add_argument(
        "--num_choices",
        default = 1,
        type=int,
        help="number of generations for each prompt"
    )

    parser.add_argument(
        "--task_type",
        default = "request",
        choices=['request', 'filter'],
        type=str,
        help="the purpose of the call"
    )

    parser.add_argument(
        "--request_form",
        default = './requests/request.jsonl',
        type=str,
        help="the path of request form"
    )
    return parser





def extract_qa(text):
    # Regex to find the question and answer
    question_match = re.search(r"Question\s*:\s*(.+?)\s*\n", text, re.DOTALL)
    answer_match = re.search(r"Answer\s*:\s*(\[[^\]]*\])", text, re.DOTALL)
    
    if question_match and answer_match:
        try:
            question = question_match.group(1).strip()
            answer = eval(answer_match.group(1).strip())  # Safe assuming controlled input
        except:
            question = None
            answer = None
        return question, answer
    else:
        return None, None



def request_API(args):
    """
        request the API to generate answers for the given questions
    """
    data_path = os.path.join(args.data_path, f'{args.data_filename}.json')
    with open(data_path, 'r') as file, open(args.request_form, "w") as r_file:
        data = json.load(file)
        for example in data:
            ################ TODO : MAKE YOUR QUERY ##############
            # Set the instruction

            query = f"""Instruction : Given a question that requires one single answer, answer the question following the instructions below:

            1. Only provide the answer without any explanation.
            2. Use the format ||ANSWER|| where ANSWER is the answer to the question. 
                Example: ||Korea||, ||David||, ||1784||


            Input:

            Question:  

            {example['question']}

            Answer: 

            """
            
            
            # set the api request
            job = {"model": args.model, "n": args.num_choices, "temperature": args.temperature, "max_tokens": args.max_tokens, "messages": [{"role": "user", "content": f'{query}'}]}

            r_file.write(json.dumps(job) + '\n')


def filter_data(args):
    # Open your jsonl file and create a new file for the cleaned data
    ids = []
    with open(args.raw_output, 'r') as infile, open(args.filtered_output, 'w') as outfile:
        # Read each line from the input file
        for index, json_data in enumerate(infile):
            # json_data = line.split(" ", 1)
            json_data = json.loads(json_data)
            message = json_data[0]
            response = json_data[1]

            data_id = int(index)

            if data_id in ids:
                continue
            else:
                ids.append(data_id)

            
            question, answer = extract_qa(message['messages'][0]['content'])
            
            cleaned_data = {
                        "id": data_id,
                        "model": message["model"],
                        "question": question,
                        "answer": answer,
                    }

            # Write the cleaned data to the output file in jsonl format
            outfile.write(f"{json.dumps(cleaned_data)}\n")
                


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    if args.task_type == 'request':
        request_API(args)
    elif args.task_type == 'filter':
        filter_data(args)
            