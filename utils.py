import re

def make_prompt(args, question_string):
    """
        function to make the instruction prompt for the given question

        - number > 0 : prompts for multiple answers
        - number < 0 : prompts for single answer

        - number >= 5: prompts for multiple answers with verbalized confidence
        - number <= -5: prompts for single answer with verbalized confidence
    """

    ###### -------------------multiple answers ---- ##########
    if args['prompt_phrasing'] == 1:
        # Zero-shot for world knowledge

        prompt = f"""Instruction : Given a question that has multiple answers, answer the question following the instructions below:

        1. Keep your response as brief as possible without any explanation.
        2. Mark each answer with a number followed by a period.
        3. Separate each answer with a number, a comma and a space.

        The format of the answer should be given as follows:

        1.YourAnswer1, 2.YourAnswer2, 3.YourAnswer3

        Input : 

        Question: 

        {question_string}

        Now, please answer this question.

        Answer :\n

        """
        # For some reason the final newline makes Falcon-7b act really weird


    elif args['prompt_phrasing'] == 2:
        # zero-shot COT for mathemtical reasoning
        
        prompt = f"""Instruction : Given a question that has multiple answers, answer the question following the instructions below:

        1. Explain step-by-step, and then provide your answer. 
        2. When providing an answer, use the format ||ANSWERS|| where ANSWERS are the answers to the given question.
        3. Separate each answer of ANSWERS with a comma and a space.

        The format of the final answer should be given as follows:

        ||ANSWER1, ANSWER2, ANSWER3||

        Input : 

        Question: 

        {question_string}

        Now, please answer this question.

        Answer :\n

        """

    elif args['prompt_phrasing'] == 3:
        # zero-shot COT for commonsense reasoning
        
        prompt = f"""Instruction : Given a set of questions, find the questions that have true answers following the instructions below:

        1. Explain step-by-step, and then provide your answer. 
        2. When providing an answer, use the format ||ANSWERS|| where ANSWERS are the answers to the given question.
        3. Separate each answer of ANSWERS with a comma and a space.
        4. Each answer should be a single alphabet letter that corresponds to the question number.

        Use the following format for the final answer:

        ||ANSWER1, ANSWER2, ANSWER3||

        Ex: ||a, c, d||

        Here is the given set of questions:

        Questions: 

        {question_string}

        Now provide the indexes of the questions that have true answers.

        Answer: \n

        """


    elif args['prompt_phrasing'] == 5:
        # verbalized confidence for world knowledge
        
        prompt = f"""Instruction : Given a question that has multiple answers, answer the question and then provide the confidence in this answer, which indicates how likely you think your answer is true, following the instructions below:

        1. Keep your response as brief as possible without any explanation, and then provide your answer and confidence. 
        2. When providing an answer, use the format ||ANSWERS|| where ANSWERS are the answers to the given question.
        3. Separate each answer of ANSWERS with a comma and a space.
        4. The confidence should be a numerical number in the range of 0-100.

        Use the following format for the final answer and confidence:

        ||ANSWER1, ANSWER2, ANSWER3||, CONFIDENCE


        Now, please answer this question.

        Question: 

        {question_string}

        Answer: \n

        """


    elif args['prompt_phrasing'] == 6:
        # verbalized confidence for mathemtical reasoning
        
        prompt = f"""Instruction : Given a question that has multiple answers, answer the question and then provide the confidence in this answer, which indicates how likely you think your answer is true, following the instructions below:

        1. Explain step-by-step, and then provide your answer and confidence. 
        2. When providing an answer, use the format ||ANSWERS|| where ANSWERS are the answers to the given question.
        3. Separate each answer of ANSWERS with a comma and a space.
        4. The confidence should be a numerical number in the range of 0-100.

        Use the following format for the final answer and confidence:

        ||ANSWER1, ANSWER2, ANSWER3||, CONFIDENCE


        Now, please answer this question.

        Question: 

        {question_string}

        Answer: \n

        """


    elif args['prompt_phrasing'] == 7:
        # verbalized confidence for commonsense reasoning
        
        prompt = f"""Instruction : Given a set of questions, find the questions that have true answers and then provide the confidence in your answer, which indicates how likely you think your answer is true, following the instructions below:

        1. Explain step-by-step, and then provide your answer and confidence. 
        2. When providing an answer, use the format ||ANSWERS|| where ANSWERS are the answers to the given question.
        3. Separate each answer of ANSWERS with a comma and a space.
        4. Each answer should be a single alphabet letter that corresponds to the question number.
        5. The confidence should be a numerical number in the range of 0-100.

        Use the following format for the final answer and confidence:

        ||ANSWER1, ANSWER2, ANSWER3||, CONFIDENCE


        Here is the given set of questions:

        Question: 

        {question_string}

        Now provide the indexes of the questions that have true answers.

        Answer: \n

        """
    


    ################ -------------------single answer ---- ##########     

    elif args['prompt_phrasing'] == -1:
        # zero-shot for world knowledge
        prompt = f"""Instruction : Given a question that requires one single answer, answer the question following the instructions below:

        1. Only provide the answer without any explanation.
        2. Use the format ||ANSWER|| where ANSWER is the answer to the question. 
            Example: ||Korea||, ||David||, ||1784||

        Input:

        Question: 

        {question_string}

        Answer:\n
        """


    elif args['prompt_phrasing'] == -2:
        # zero-shot for mathemtical reasoning
        prompt = f"""Instruction : Given a question that requires one single answer, answer the question following the instructions below:

        1. Explain step-by-step, and then provide your answer. 
        2. Remark: The answer should only be a number. So do not use any other units or symbols.
        
        The format of the final answer, that comes after the explanation, should be given as follows:

        ||YOUR_ANSWER||

        where YOUR_ANSWER is an answer to the given question, which is a number (ex, ||5||, ||3.2||).

        Input:

        Question: {question_string}

        Answer:\n
        """

    elif args['prompt_phrasing'] == -3:
        # zero-shot for commonsense reasoning
        
        prompt = f"""Instruction : Given a question, provide your thought whether the answer to the question is true or false, following the instructions below:

        1. Explain step-by-step, and then provide your answer. 
        2. When providing an answer, use the format ||ANSWER|| where ANSWER is the answer to the given question.
        3. Answer should be either ||true|| or ||false||.

        Questions: 

        {question_string}

        Now answer the question. 

        Answer: \n

        """
    
    elif args['prompt_phrasing'] == -5:
        # verbalized confidence for world knowledge
        prompt = f"""Instruction : Given a question that requires one single answer, answer the question and then provide the confidence in this answer, which indicates how likely you think your answer is true, following the instructions below:

        1. Only provide the answer without any explanation.
        2. Use the format ||ANSWER|| where ANSWER is the answer to the question. 
        3. The confidence should be a numerical number in the range of 0-100.

        Use the following format for the final answer and confidence:

        ||YOUR_ANSWER||, CONFIDENCE

        where YOUR_ANSWER is an answer to the given question (ex, ||Korea||, ||1784||).

        Now, please answer this question.


        Input:

        Question: 

        {question_string}

        Answer:\n
        """


    elif args['prompt_phrasing'] == -6:
        # verbalized confidence for mathemtical reasoning
        prompt = f"""Instruction : Given a question that requires one single answer, answer the question and then provide the confidence in this answer, which indicates how likely you think your answer is true, following the instructions below:

        1. Explain step-by-step, and then provide your answer.
        2. Use the format ||ANSWER|| where ANSWER is the answer to the question. 
        3. Remark: The answer should only be a number. So do not use any other units or symbols.
        4. The confidence should be a numerical number in the range of 0-100.

        Use the following format for the final answer and confidence:

        ||YOUR_ANSWER||, CONFIDENCE

        where YOUR_ANSWER is an answer to the given question, which is a number (ex, ||5||, ||3.2||).


        Input:

        Question: 

        {question_string}

        Answer:\n
        """

    
    elif args['prompt_phrasing'] == -7:
        # verbalized confidence for commonsense reasoning
        prompt = f"""Instruction : Given a question, provide your thought whether the answer to the question is true or false and then provide the confidence in this answer, which indicates how likely you think your answer is true, following the instructions below:

        1. Explain step-by-step, and then provide your answer. 
        2. When providing an answer, use the format ||ANSWER|| where ANSWER is the answer to the given question.
        3. Answer should be either ||true|| or ||false||.
        4. The confidence should be a numerical number in the range of 0-100.

        Use the following format for the final answer and confidence:

        ||YOUR_ANSWER||, CONFIDENCE

        where YOUR_ANSWER is a true or false (ex, ||true||, ||false||).

        Now, please answer this question.

        Input:

        Question: 

        {question_string}

        Answer:\n
        """

    
    else:
        raise Exception(f"Unknown phrasing option: {args['prompt_phrasing']}. Must be 0 or 1.")
            # For some reason the final newline makes Falcon-7b act really weird
    return prompt if args['model'] != 'Falcon-7b' else prompt[:-1]


def determine_llm_answer(llm_output, prompt_phrasing=0):
    """
        Function to parse the answer from the LLM output
        Please refer to make_prompt function to match the format of the answer with the prompt_phrasing
    """

    if prompt_phrasing==2:
        try:
            # pattern = re.compile(r'\d+\.\s(.*?)(,|\n|$)')
            pattern = re.compile(r'\d+\.\s*([^,]+)')
            
            # Find all matches in the answer string
            matches = pattern.findall(llm_output)
            
            # Extract just the answer part from each match, ignoring trailing commas or new lines
            # Define the new regex pattern to capture answers with or without numbers
            


            # Extract just the answer part from each match, ignoring leading or trailing spaces
            answers = [match.strip() for match in matches if match.strip()]

            # If you need to handle a single comma-separated string without numbers in front
            if len(answers) == 1 and ',' in answers[0]:
                answers = [answer.strip() for answer in answers[0].split(',')]

            # Remove unwanted characters like 1., 2., , or .
            answers = [re.sub(r'^\d+\.\s*|[,.]$', '', answer).strip() for answer in answers]
            # answers = [match[0].strip() for match in matches if match[0].strip()]
            
            return answers
        
        except Exception as e:
            return f"Could not parse answer: {str(e)}"

    elif prompt_phrasing >= 5 and prompt_phrasing < 10:
        try:
            # Define the regular expression pattern to extract answers and confidence
            pattern = re.compile(r'.*?\|\|?(.*?)(?:\|\||\|).*?,\s*(\d{1,3})\s*.*')

            # Search for the pattern in the input string
            match = pattern.search(llm_output)

            if match:
                # Extract answers and confidence from the matched groups
                answers_part = match.group(1)
                answers = [answer.strip() for answer in answers_part.split(',')]
                confidence = int(match.group(2).strip())
                
                return answers, confidence
            else:
                return [], None

        except ValueError:
            # Handle case where confidence part is not a valid integer
            return [], None


    elif prompt_phrasing >= 1:
        try:
            
            # Split the input string by '||'
            parts = llm_output.split('||')
            
            # Check if the input string has the correct format
            if len(parts) >= 3:
                # Extract the answers from the second part
                # print("parts")
                # print(parts)
                answers = parts[-2].split(',')
                answers = [answer.strip() for answer in answers]
                return answers

            elif len(parts) == 2:
                if "," in parts[-1] and "|" in parts[-1]:
                    answers = parts[-1].split("|")[-2].split(',')
                    answers = [answer.strip() for answer in answers]
                    return answers

                elif "," in parts[-2] and "|" in parts[-2]:
                    answers = parts[-2].split("|")[-1].split(',')
                    answers = [answer.strip() for answer in answers]
                    return answers
                else:
                    return []

            else:
                return []
        
        except Exception as e:
            return f"Could not parse answer: {str(e)}"


    elif prompt_phrasing <= -5 and prompt_phrasing > -10:
        try:
            # Define the regular expression pattern to extract the answer and confidence
            pattern = re.compile(r'.*?\|\|?(.*?)(?:\|\||\|).*?,\s*(\d{1,3})\s*.*')

            # Search for the pattern in the input string
            match = pattern.search(llm_output)

            if match:
                # Extract answer and confidence from the matched groups
                answer = match.group(1).strip()
                confidence = int(match.group(2).strip())
                
                return answer, confidence
            else:
                return "No answer found", None

        except Exception as e:
            return f"Could not parse answer: {str(e)}", None

    elif prompt_phrasing <= -1:
        try:
            # Use regex to find all matches within the || tags
            matches = re.findall(r'\|\|(.*?)\|\|', llm_output)
            if matches:
                return matches[-1]  # Return the last match found
            else:
                return "No answer found"
        except Exception as e:
            return f"Could not parse answer: {str(e)}"
