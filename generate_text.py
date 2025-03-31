from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer, BitsAndBytesConfig
import argparse
import torch as t
from utils import str_to_bool

bnb_config = BitsAndBytesConfig(
load_in_8bit=True,
llm_int8_enable_fp32_cpu_offload=False,
bnb_4bit_compute_dtype=t.bfloat16)


# text generator class
class Generator(object):
    def __init__(self, args):
        model_name_map = {'Mistral-raw':'mistralai/Mistral-7B-v0.2',
                          'Mistral':'mistralai/Mistral-7B-Instruct-v0.2',
                          'Mixtral-raw':'mistralai/Mixtral-8x7B-v0.1',
                          'Mixtral':'mistralai/Mixtral-8x7B-Instruct-v0.1',
                          'Zephyr':'HuggingFaceH4/zephyr-7b-beta',
                          'gpt2':'gpt2',
                          'Llama-7b-raw':'meta-llama/Llama-2-7b-hf',
                          'Llama-7b':'meta-llama/Llama-2-7b-chat-hf',
                          'Llama-13b-raw':'meta-llama/Llama-2-13b-hf',
                          'Llama-13b':'meta-llama/Llama-2-13b-chat-hf',
                          'Llama-70b-raw':'meta-llama/Llama-2-70b-hf',
                          'Llama-70b':'meta-llama/Llama-2-70b-chat-hf',
                          'Llama-3-8b': 'meta-llama/Meta-Llama-3-8B-Instruct',
                          'Llama-3-70b': 'meta-llama/Meta-Llama-3-70B-Instruct',
                          'Gemma-2b': "google/gemma-2b-it",
                          'Gemma-2b-raw': "google/gemma-2b",
                          'Gemma-7b': "google/gemma-1.1-7b-it",
                          'Gemma-7b-raw': "google/gemma-7b",
                          "Qwen-7b": "Qwen/Qwen1.5-7B-Chat",
                          "Qwen-32b": "Qwen/Qwen1.5-32B-Chat",
                          "Qwen-72b": "Qwen/Qwen1.5-72B-Chat",
                          'Falcon-7b-raw':'tiiuae/falcon-7b',
                          'Falcon-7b':'tiiuae/falcon-7b-instruct',
                          'Falcon-40b-raw':'tiiuae/falcon-40b',
                          'Falcon-40b':'tiiuae/falcon-40b-instruct',
                          'Solar-10.7B-raw':'upstage/SOLAR-10.7B-v1.0',
                          'Solar':'upstage/SOLAR-10.7B-Instruct-v1.0',
                          'Yi-34b':'01-ai/Yi-34B-Chat',
                          'Yi-6b':'01-ai/Yi-6B-Chat',
                          'Yi-34b-raw':'01-ai/Yi-34B',
                          'Yi-6b-raw':'01-ai/Yi-6B',
                          "ChatGPT-3.5": "openai/chatgpt-3.5",
        }
        if args['model'] not in model_name_map:
            raise Exception("Unrecognized model name. Check model_name_map")
        else:
            model_name = model_name_map[args['model']]
        if 'raw' in args['model']:
            args['completion_mode'] = True

            
        if args['output'] is None:
            print("Loading model", model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, 
            device_map="auto",
            torch_dtype=t.float16)
            #, quantization_config=bnb_config
        
        if 'ChatGPT' not in args['model']:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
            if 'Qwen' not in args['model'] and "Gemma" not in args["model"]:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        else:
            self.tokenizer = None

        self.args = args

        self.num_responses = 1 if not self.args['do_sample'] else self.args['num_responses']
            
    
    def prepare_for_chat(self, prompts):
        if 'Falcon' in self.args['model']:
            return prompts # These models doesn't use chat templates
        else:
            chats = [[{"role": "user", "content": p}] for p in prompts]
            return [self.tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True, return_tensors="pt") for c in chats]

    
            
    def generate(self, prompts):
        prompts = self.prepare_for_chat(prompts) if not self.args['completion_mode'] else prompts
        
        model_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")



        if 'Llama' in self.args['model']:

            terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

        else:
            terminators = self.tokenizer.eos_token_id

        print("tokenized finished")

        renormalize_logits=True
        
        
        output = self.model.generate(**model_inputs, max_new_tokens=self.args['max_new_tokens'], do_sample=self.args['do_sample'], output_scores=(not self.args['do_sample']), num_return_sequences=self.num_responses, return_dict_in_generate=True, renormalize_logits=renormalize_logits, temperature=self.args['temperature'], top_p=self.args['top_p'], top_k=self.args['top_k'], num_beams=1, eos_token_id=terminators)

        print("generation finished")
        
        token_inputs = model_inputs['input_ids'] if 'Yi' in self.args['model'] else model_inputs
        token_outputs = [output.sequences[i][len(token_inputs[i//self.num_responses]):] for i in range(len(output.sequences))] #
 
        text_outputs = self.tokenizer.batch_decode(token_outputs, skip_special_tokens=True)


        del token_outputs
        del model_inputs
        del token_inputs

        num_respones = self.args["num_responses"]

        if num_respones > 1:
            # for the text outputs, make sure to group them by prompt
            text_outputs = [text_outputs[i*num_respones:(i+1)*num_respones] for i in range(len(prompts))]
            scores = None

        else:
            scores = t.stack(list(output.scores), dim=0).to('cpu') # initially it's a tuple of tensors


            if scores.dtype != t.float32:
                print("Casting scores to float32")
                scores = scores.to(t.float32)

        del output

        return (text_outputs, scores)

def parse_args():
    parser = argparse.ArgumentParser(description='Perform text generation and Q&A tasks via Hugging Face models.')
    parser.add_argument('-m', '--model', type=str, help='Which LLM to use. Check this file for currently supported options and/or add your own.',required=True)
    parser.add_argument('-p', '--prompts', type=str, help='List of prompts, separated by |. For example "Hello my name is Ben|What a time to be alive". If not provided, you will be asked for a prompt by command line.', default=None)
    parser.add_argument('-n', '--max_new_tokens', type=int, help='Number of new tokens to generate on top of the prompt', default=10)
    parser.add_argument('-k', '--num_top_tokens', type=int, help='For each token, print out the top candidates considered by the model and their probabilities', default=3)
    parser.add_argument('-c', '--completion_mode', action="store_true", help='Use traditional auto-complete mode, rather than user-assistant chat', default=False)
    parser.add_argument('-s', '--do_sample', action="store_true", help='Should we sample from the probability distribution, or greedily pick the most likely token?', default=False)
    parser.add_argument('-r', '--num_responses', type=int, help='Number of responses to generate per prompt. This argument is ignored for greedy decoding, since that only generates one answer.', default=1)
    parser.add_argument('-d', '--dataset', type=str, default=None, help='The name of the Hugging Face dataset (needed for experiments and such)')
    parser.add_argument('-q', '--question_range', type=str, help='When running a Q&A test, what range of questions should we test? Format is "-q startq-endq", 0 indexed. For example, "-q 0-100".', default=None)
    parser.add_argument('-b', '--batch_size', type=int, help='Maximum number of prompts to batch together. Only used for experiments', default=1)
    parser.add_argument('-g', '--prompt_phrasing', type=int, help='When running a Q&A test, which of the two prompt phrasings should we use? 0 or 1', default=0)
    parser.add_argument('-t', '--temperature', type=float, help="Temperature for sampling. Only used when do_sample is set to True", default=1.0)
    parser.add_argument('--top_p', type=float, help="Top-p sampling cutoff. Only used when do_sample is set to True", default=1.0)
    parser.add_argument('--top_k', type=int, help="Top-k sampling cutoff. Only used when do_sample is set to True", default=1)
    parser.add_argument('-u', '--ue_mode', type=str, choices=['black', 'white'], help="Uncertainty estimation mode", default='white')
    parser.add_argument('-e', '--eval_methods', type=str, help="Evaluation methods to use, separated by ,", default="rouge")
    parser.add_argument('-o', '--overwrite', type=str_to_bool, help='when to overwrite the output?', default=False)
    return dict(vars(parser.parse_args())) # dictionaries are easier to manipulate sometimes

    
def main():
    args = parse_args() 
    generator = Generator(args)

    if args['prompts'] == None:
        prompts = [input("\nEnter an initial prompt:\n")]
        print('\n')
    else:
        prompts = args['prompts'].split('|')
    
    generator.generate(prompts)
        
if __name__ == '__main__':
    main()
