import generate_text
from evaluation import Evaluator
import os
from datasets import load_dataset, concatenate_datasets
import random
from string import ascii_uppercase
from uncertainty.white_box import WhiteBox
from uncertainty.black_box import BlackBox
import json
import re
import torch
import copy
from tqdm import tqdm

import warnings
from transformers import logging
from utils import make_prompt, determine_llm_answer


class Test(object):
    def __init__(self, args):
        bounds = args['question_range'].split('-')
        (self.start_q, self.end_q) = (int(bounds[0]), int(bounds[1]))
        self.model = generate_text.Generator(args)
        self.args = args

        dset_name = args['dataset'] #.lower()

        with open(f'datasets/{dset_name}.json') as f:
            dset = json.load(f)

        if dset is None:
            raise Exception(f"Unsupported dataset name: {dset_name}")

        self.data = list(dset)

        if self.end_q == 0:
            self.end_q = len(self.data)
        else:
            self.end_q = min(self.end_q, len(self.data))

        self.get_q = (lambda x:
                      x['question'] )
        self.get_a = (lambda x:
                      x['answer'] )

        self.evaluator = Evaluator(args)
        # get the confidence levels
        self.white_box_confidence_model = WhiteBox(self.model.tokenizer, args)
        self.black_box_confidence_model = BlackBox(args)

    def run_test(self, start_q, end_q):
        assert(start_q < end_q)
        # First assemble all of the prompts
        print("Forming prompts...\n")
        num_prompts = end_q - start_q
        result = [{} for _ in range(num_prompts)]
        prompts = [None] * num_prompts

        if self.args['output'] is None:
            for i in range(start_q, end_q):
                qa_data = self.data[i]
                question = self.get_q(qa_data)

                gt_answers = self.get_a(qa_data)

                prompt = make_prompt(self.args, question)
                prompts[i - start_q]   = prompt
                result[i - start_q]['prompt'] = prompt
                result[i - start_q]['question'] = question
                result[i - start_q]['gt_answers'] = gt_answers

            # Batch inference
            print("Running inference...\n")
            (text_outputs, scores) = self.model.generate(prompts)
            predicted_answers = []

            print("Evaluating results...\n")

        else:
            print(len(self.args['output']))
            for i in range(start_q, end_q):
                result[i-start_q] = copy.deepcopy(self.args['output'][i])

        # grade the result
        for i in range(len(result)):
            if self.args['ue_mode'] == 'white':
                if self.args['output'] is None:

                    result[i]['llm_output'] = text_outputs[i]
                    print(text_outputs[i])

                    # in case for the verbalized confidencez
                    if abs(self.args['prompt_phrasing']) >= 5 and abs(self.args['prompt_phrasing']) <= 10:
                        predicted_answer, confidence = determine_llm_answer(text_outputs[i], self.args['prompt_phrasing'])
                    else:
                        predicted_answer = determine_llm_answer(text_outputs[i], self.args['prompt_phrasing'])

                    predicted_answers.append(predicted_answer)
                    result[i]['predicted_answer'] = predicted_answer
                    llm_output = text_outputs[i]

                    # get the confidence score
                    result[i]["confidence_measures"] = {}

                    if abs(self.args['prompt_phrasing']) >= 5 and abs(self.args['prompt_phrasing']) <= 10:
                        result[i]["confidence_measures"]['verbalize'] = confidence

                    else:
                        top_vocab_logit_maps = self.white_box_confidence_model.get_top_logits(text_outputs[i], predicted_answers[i], scores[:,i,:], top_k=self.args['num_top_tokens'], normalize=True)


                        result[i]["confidence_measures"]['top_vocab_logit_maps'] = top_vocab_logit_maps

                        confidence_score = self.white_box_confidence_model.mean_top_logits(top_vocab_logit_maps)
                        max_logits = self.white_box_confidence_model.max_list_top_logits(top_vocab_logit_maps)
                        result[i]["confidence_measures"]['max_logits'] = max_logits
                        result[i]["confidence_measures"]['margins'] = self.white_box_confidence_model.margin_list_top_logits(top_vocab_logit_maps)
                        result[i]["confidence_measures"]['entropy'] = self.white_box_confidence_model.entropy_list_top_logits(top_vocab_logit_maps)


                        result[i]["confidence_measures"]['mean_seq_logit'] = confidence_score
                        entropy = self.white_box_confidence_model.mean_greedy_entropy(top_vocab_logit_maps)
                        result[i]["confidence_measures"]['mean_greedy_entropy'] = entropy


                else:

                    # in case for the verbalized confidencez
                    if abs(self.args['prompt_phrasing']) >= 5 and abs(self.args['prompt_phrasing']) <= 10:
                        print("error check: ", result[i]['llm_output'])
                        predicted_answer, confidence = determine_llm_answer(result[i]['llm_output'], self.args['prompt_phrasing'])
                    else:
                        predicted_answer = determine_llm_answer(result[i]['llm_output'], self.args['prompt_phrasing'])

                    result[i]['predicted_answer'] = predicted_answer

                    if abs(self.args['prompt_phrasing']) >= 5 and abs(self.args['prompt_phrasing']) <= 10:
                        result[i]["confidence_measures"] = {}
                        result[i]["confidence_measures"]['verbalize'] = confidence

                    else:

                        # calculate the confidence
                        top_vocab_logit_maps = result[i]["confidence_measures"]['top_vocab_logit_maps']
                        confidence_score = self.white_box_confidence_model.mean_top_logits(top_vocab_logit_maps)
                        max_logits = self.white_box_confidence_model.max_list_top_logits(top_vocab_logit_maps)
                        result[i]["confidence_measures"]['max_logits'] = max_logits
                        result[i]["confidence_measures"]['margins'] = self.white_box_confidence_model.margin_list_top_logits(top_vocab_logit_maps)
                        result[i]["confidence_measures"]['entropy'] = self.white_box_confidence_model.entropy_list_top_logits(top_vocab_logit_maps)
                        result[i]["confidence_measures"]['mean_seq_logit'] = confidence_score
                        entropy = self.white_box_confidence_model.mean_greedy_entropy(top_vocab_logit_maps)
                        result[i]["confidence_measures"]['mean_greedy_entropy'] = entropy

                # get the evalution score
                result[i]['evaluation_results'] = self.evaluator.eval_all(result[i]['gt_answers'], predicted_answer)


            elif self.args['ue_mode'] == 'black':
                if self.args['output'] is None:
                    result[i]['llm_output'] = text_outputs[i]
                    #predicted_answer = [determine_llm_answer(text, self.args['prompt_phrasing']) for text in text_outputs[i]]
                    #predicted_answer, = determine_llm_answer(text_outputs[i], self.args['prompt_phrasing'])
                    if abs(self.args['prompt_phrasing']) >= 5 and abs(self.args['prompt_phrasing']) <= 10:
                        print("error check: ", result[i]['llm_output'])
                        predicted_answer, confidence = determine_llm_answer(result[i]['llm_output'], self.args['prompt_phrasing'])
                        result[i]["confidence_measures"] = {}
                        result[i]["confidence_measures"]['verbalize'] = confidence
                    else:
                        predicted_answer = determine_llm_answer(result[i]['llm_output'], self.args['prompt_phrasing'])
                    predicted_answers.append(predicted_answer)


                    result[i]['predicted_answer'] = predicted_answer
                    llm_output = text_outputs[i]

                    # get the confidence

                    result[i]["confidence_measures"] = {}

                    mean_exact_score = self.black_box_confidence_model.mean_exact_score(text_outputs[i])
                    result[i]["confidence_measures"]['mean_exact_score'] = mean_exact_score

                else:
                    # predicted_answer = [determine_llm_answer(text, self.args['prompt_phrasing']) for text in result[i]['llm_output']]
                    # result[i]['predicted_answer'] = predicted_answer
                    result[i]["confidence_measures"] = {}
                    if abs(self.args['prompt_phrasing']) >= 5 and abs(self.args['prompt_phrasing']) <= 10:
                        print("error check: ", result[i]['llm_output'])
                        predicted_answer, confidence = determine_llm_answer(result[i]['llm_output'], self.args['prompt_phrasing'])

                        result[i]["confidence_measures"]['verbalize'] = confidence
                    else:
                        predicted_answer = determine_llm_answer(result[i]['llm_output'], self.args['prompt_phrasing'])
                    result[i]['predicted_answer'] = predicted_answer
                    predicted_answers.append(predicted_answer)

                    mean_exact_score = self.black_box_confidence_model.mean_exact_score(predicted_answer)
                    result[i]["confidence_measures"]['mean_exact_score'] = mean_exact_score


                # get the evalution score
                eval_results = []
                for ans in predicted_answer:

                    cur_eval_result = self.evaluator.eval_all(result[i]['gt_answers'], ans)
                    eval_results.append(cur_eval_result)

                # averge over the evaluation results
                eval_result = {}
                for key in eval_results[0].keys():
                    eval_result[key] = sum([res[key] for res in eval_results]) / len(eval_results)


                #result[i]['evaluation_results'] = eval_result
                result[i]['evaluation_results'] = self.evaluator.eval_all(result[i]['gt_answers'], predicted_answer)


            else:
                raise Exception(f"Unknown uncertainty estimation mode: {self.args['ue_mode']}")

        return result

def main():
    random.seed(2549900867) # We'll randomize the order of questions and of answer choices, but we want every run to have the same randomization
    args = generate_text.parse_args()
    qrange = ""
    if args['question_range'] != "0-0":
        qrange = f"_{args['question_range']}"
    args['ue_mode'] = 'black'
    args['prompt_phrasing'] = 5
    args['output_path'] = f"result_debug/{args['model']}_{args['ue_mode']}_{args['prompt_phrasing']}_{args['dataset']}{qrange}.json"
    args['output'] = None
    args['single'] = False
    if args['prompt_phrasing'] <= -1 or "single" in args['dataset']:
        args['single'] = True
    args['eval_methods'] = args['eval_methods'].split(',')
    # args.output_path already exists
    if os.path.exists(args['output_path']) and not args['overwrite']:
        with open(args['output_path'], 'r') as f:
            existing_results = json.load(f)
            args['output'] = existing_results

    test = Test(args)
    results = []


    for start_q in tqdm(range(test.start_q, test.end_q, args['batch_size'])):
        end_q = min(start_q + args['batch_size'], test.end_q)
        if args['batch_size'] > 1:
            print(f"\nSTARTING NEW BATCH: questions {start_q} to {end_q}\n")
        result = test.run_test(start_q, end_q)
        results.extend(result)

        if args['output'] is None:
            with open(f"{args['output_path']}", 'w') as f:
                json.dump(results, f, indent=4)

        torch.cuda.empty_cache()

    with open(f"{args['output_path']}", 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    main()
