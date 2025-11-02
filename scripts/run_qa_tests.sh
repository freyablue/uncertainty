#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 6 ]; then
    echo "Error: Incorrect number of arguments."
    echo "Usage format: ./run_qa_tests.sh model1,model2 dataset1,dataset2,dataset3 question_range1,question_range2 prompt_phrasing overwrite eval_methods"
    echo "Example usage: bash ./scripts/run_qa_tests.sh Mistral,Qwen-7b,Llama-3-8b,Zephyr,Gemma-7b MAQA_world_knowledge_nq 0-0 4 True rouge,exact_match"
    exit 1
fi

# Read the argument options from the command line input and split them into arrays
IFS=',' read -r -a model_options <<< "$1"
IFS=',' read -r -a dataset_options <<< "$2"
IFS=',' read -r -a question_ranges <<< "$3"
prompt_phrasing="$4"
overwrite="$5"
eval_methods="$6"

# Function to determine batch_size based on model and dataset names
get_batch_size() {
    local model_name="$1"
    local dataset_name="$2"
    local batch_size

    # remove -raw from the model name if it exists
    model_name="${model_name/-raw/}"

    case "$model_name" in
        "Llama-70b" | "Llama-3-70b" | "Qwen-32b" | "Mixtral")
            batch_size=32
            ;;
        "Qwen-72b")
            batch_size=8
            ;;
        "Llama-7b" | "Llama-3-8b" | "Qwen-7b")
            batch_size=32
            ;;
        "Gemma-7b")
            batch_size=8
            ;;
        "Llama-13b")
            batch_size=22
            ;;
        "Falcon-40b")
            batch_size=10
            ;;
        "Falcon-7b")
            batch_size=168
            ;;
        "Yi-6b" | "Yi-34b") # For some reason, this crashes for batch_size > 1
            batch_size=1
            ;;
        "Mistral" | "Zephyr")
            batch_size=8
            ;;
        "Solar")
            batch_size=4
            ;;
        *)
            batch_size=63 # Default value
            ;;
    esac

    # If "nq" is in the dataset_name, make batch size 4 times larger
    if [[ "$dataset_name" == *"nq"* ]]; then
        batch_size=$(( batch_size * 8 ))
    fi

    echo "$batch_size"
}


# Create logs directory if it doesn't exist
mkdir -p logs


# Loop through each combination of model, dataset, and question_range
for question_range in "${question_ranges[@]}"
do
    for dataset in "${dataset_options[@]}"
    do
        for model in "${model_options[@]}"
        do
            # Determine batch_size based on the model and dataset
            #batch_size=$(get_batch_size "$model" "$dataset")
            batch_size=2

            # Define log file name
            log_file="logs/${model}_${dataset}_${question_range}_prompt-phrasing-${prompt_phrasing}_log.txt"

            # Running the command with the arguments
            echo -e "\nRunning take_qa_test.py with arguments: --model=$model --dataset=$dataset --question_range=$question_range --batch_size=$batch_size --prompt_phrasing=$prompt_phrasing --max_new_tokens=64 --num_top_tokens=10 --ue_mode white --eval_methods $eval_methods --overwrite $overwrite\n"
            
            python3 take_qa_test.py --model="$model" --dataset="$dataset" --question_range="$question_range" --batch_size="$batch_size" --prompt_phrasing="$prompt_phrasing" --max_new_tokens=64 --num_top_tokens=10 --ue_mode "white" --eval_methods "$eval_methods" --overwrite "$overwrite" #&> "$log_file"
        done
    done
done

echo "Script completed."
