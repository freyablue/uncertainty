from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score
import nltk
from nltk.metrics.distance import edit_distance
from nltk.translate.meteor_score import meteor_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import copy

nltk.download('wordnet')  # Required by METEOR
nltk.download('omw-1.4')  # Also often needed by METEOR
nltk.download('punkt')

class Evaluator:
    def __init__(self, args):
        self.args = args

    def single_eval(self, gt_answers, pred_answer):

        pred_answer = pred_answer.lower()

        # Evaluate the prediction
        if isinstance(gt_answers, list):
            # Check if any ground truth answer is included in the predicted answer or vice versa

            #match_found = any((gt.lower() in pred_answer or pred_answer in gt.lower()) for gt in gt_answers)
            match_found = any((gt.lower() in pred_answer) for gt in gt_answers)

            eval_score = int(match_found)
        elif isinstance(gt_answers, str):
            eval_score = 1 if gt_answers.lower() == pred_answer.lower() else 0
        elif isinstance(gt_answers, (int, float)):
            try:
                eval_score = 1 if float(gt_answers) == float(pred_answer) else 0
            except ValueError:
                eval_score = 0
        else:
            eval_score = 0

        return eval_score

    def single_rouge_eval(self,gt_answers, pred_answer):
        # Create a ROUGE scorer with the desired metrics
        rouge = Rouge()

        # Join multiple ground truth answers into a single string if they are in a list
        if isinstance(gt_answers, list):
            gt_answers = ' '.join(gt_answers)
        try:
            # Calculate the ROUGE scores
            scores = rouge.get_scores(gt_answers, pred_answer)[0]['rouge-l']['f']
        except:
            scores=0

        return scores


    def exact_match(self, gt_answers, pred_answers):
        def compare_values(gt, pred):
            try:
                if isinstance(gt, (int, float)):
                    return gt == float(pred)
                elif isinstance(gt, str):
                    gt_answers = str(gt).lower() if not isinstance(gt, list) else [str(item).lower() for item in gt]
                    pred_answers = str(pred).lower() if not isinstance(pred, list) else [str(item).lower() for item in pred]
                    #return pred.lower() == gt.lower()
                    return gt_answers == pred_answers
                    # return (pred.lower() in gt.lower()) or (gt.lower() in pred.lower())
                elif isinstance(gt, list):
                    if isinstance(pred, list):
                        pred = " ".join(pred)
                    if pred==None:
                        return False
                    return any((pred.lower() == str(item).lower()) for item in gt)
                    # return any(((pred.lower() in str(item).lower()) or (str(item).lower() in pred.lower())) for item in gt)
                    #return any((pred.lower() == str(item).lower()) for item in gt if isinstance(gt, list))
            except ValueError:
                return False
            return False

        def calculate_match_scores(answers1, answers2, is_gt_first=True):
            match_scores = []
            for answer1 in answers1:
                match_found = any(compare_values(answer1, answer2) if is_gt_first else compare_values(answer2, answer1) for answer2 in answers2)
                match_scores.append(int(match_found))
            return match_scores

        # Calculate precision
        match_scores_precision = calculate_match_scores(pred_answers, gt_answers, is_gt_first=False)
        mean_precision = sum(match_scores_precision) / len(pred_answers) if pred_answers else 0

        # Calculate recall
        match_scores_recall = calculate_match_scores(gt_answers, pred_answers)
        mean_recall = sum(match_scores_recall) / len(gt_answers) if gt_answers else 0

        # Calculate F1 score
        mean_f1_score = 2 * mean_precision * mean_recall / (mean_precision + mean_recall) if (mean_precision + mean_recall) != 0 else 0

        return mean_precision, mean_recall, mean_f1_score, match_scores_precision

    def rouge_eval(self, gt_answers, pred_answers):
        def compare_rouge(gt, pred):
            rouge = Rouge()
            try:
                if isinstance(gt, (int, float)):
                    gt = str(gt)
                elif isinstance(gt, list):
                    gt = ' '.join(map(str, gt))
                return rouge.get_scores(pred, gt)[0]['rouge-l']['f']
            except ValueError:
                return 0.0

        def calculate_rouge_scores(answers1, answers2, is_gt_first=True):
            rouge_scores = []
            for answer1 in answers1:
                scores = [compare_rouge(answer1, answer2) if is_gt_first else compare_rouge(answer2, answer1) for answer2 in answers2]
                rouge_scores.append(max(scores) if scores else 0.0)
            return rouge_scores

        # Normalize and lower case answers
        gt_answers = [str(gt).lower() if not isinstance(gt, list) else [str(item).lower() for item in gt] for gt in gt_answers]
        pred_answers = [pred.lower() for pred in pred_answers]

        # Calculate precision
        rouge_scores_precision = calculate_rouge_scores(pred_answers, gt_answers, is_gt_first=False)
        rouge_precision = sum(rouge_scores_precision) / len(rouge_scores_precision) if rouge_scores_precision else 0

        # Calculate recall
        rouge_scores_recall = calculate_rouge_scores(gt_answers, pred_answers)
        rouge_recall = sum(rouge_scores_recall) / len(gt_answers) if rouge_scores_recall else 0

        # Calculate F1 score
        rouge_f1_score = 2 * rouge_precision * rouge_recall / (rouge_precision + rouge_recall) if (rouge_precision + rouge_recall) != 0 else 0

        return rouge_precision, rouge_recall, rouge_f1_score, rouge_scores_precision

    def bleu_eval(self, gt_answers, pred_answers):
        #gt_answers = [[gt.lower().split()] for gt in gt_answers]  # List of lists of tokens for each ground truth answer
        gt_answers = [str(gt).lower() if not isinstance(gt, list) else [str(item).lower() for item in gt] for gt in gt_answers]
        pred_answers = [str(pred).lower() if not isinstance(pred, list) else [str(item).lower() for item in pred] for pred in pred_answers]

        #pred_answers = [pred.lower().split() for pred in pred_answers]  # List of tokens for each predicted answer

        max_bleu_scores = []
        for pred in pred_answers:
            scores = [sentence_bleu(gt, pred, smoothing_function=SmoothingFunction().method1) for gt in gt_answers]  # Compute BLEU for each ground truth
            max_bleu_scores.append(max(scores))

        bleu_precision = sum(max_bleu_scores) / len(max_bleu_scores) if max_bleu_scores else 0
        bleu_recall = sum(max_bleu_scores) / len(gt_answers) if max_bleu_scores else 0
        bleu_f1_score = 2 * bleu_precision * bleu_recall / (bleu_precision + bleu_recall) if (bleu_precision + bleu_recall) != 0 else 0

        return bleu_precision, bleu_recall, bleu_f1_score

    def bertscore_eval(self, gt_answers, pred_answers):
        #gt_answers = [gt.lower() for gt in gt_answers]
        gt_answers = [str(gt).lower() if not isinstance(gt, list) else [str(item).lower() for item in gt] for gt in gt_answers]
        pred_answers = [str(pred).lower() if not isinstance(pred, list) else [str(item).lower() for item in pred] for pred in pred_answers]

        max_bert_scores = []
        for pred in pred_answers:
            P, R, F1 = score([pred]*len(gt_answers), gt_answers, lang='en', device='cuda')
            max_bert_scores.append(F1.max().item())

        bert_precision = sum(max_bert_scores) / len(max_bert_scores) if max_bert_scores else 0
        bert_recall = sum(max_bert_scores) / len(gt_answers) if max_bert_scores else 0
        bert_f1_score = 2 * bert_precision * bert_recall / (bert_precision + bert_recall) if (bert_precision + bert_recall) != 0 else 0

        return bert_precision, bert_recall, bert_f1_score

    def edit_distance_eval(self, gt_answers, pred_answers):
        #gt_answers = [gt.lower() for gt in gt_answers]
        gt_answers = [str(gt).lower() if not isinstance(gt, list) else [str(item).lower() for item in gt] for gt in gt_answers]


        #pred_answers = [pred.lower() for pred in pred_answers]
        pred_answers = [str(pred).lower() if not isinstance(pred, list) else [str(item).lower() for item in pred] for pred in pred_answers]

        min_edit_distances = []
        for pred in pred_answers:
            scores = [edit_distance(pred, gt) for gt in gt_answers]  # Compute edit distance for each ground truth
            min_edit_distances.append(min(scores))  # Minimum edit distance for this prediction

        # Normalize edit distances to a scale of 0 to 1 (0 being no difference and 1 being maximum difference)
        max_length = max(max(len(gt) for gt in gt_answers), max(len(pred) for pred in pred_answers))
        normalized_scores = [1 - (score / max_length) for score in min_edit_distances]

        edit_precision = sum(normalized_scores) / len(normalized_scores) if normalized_scores else 0
        edit_recall = sum(normalized_scores) / len(gt_answers) if normalized_scores else 0
        edit_f1_score = 2 * edit_precision * edit_recall / (edit_precision + edit_recall) if (edit_precision + edit_recall) != 0 else 0

        return edit_precision, edit_recall, edit_f1_score

    def cosine_similarity_eval(self, gt_answers, pred_answers):
        #gt_answers = [gt.lower() for gt in gt_answers]
        gt_answers = [str(gt).lower() if not isinstance(gt, list) else [str(item).lower() for item in gt] for gt in gt_answers]


        pred_answers = [pred.lower() for pred in pred_answers]

        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(gt_answers + pred_answers)
        gt_vectors = vectors[:len(gt_answers)]
        pred_vectors = vectors[len(gt_answers):]

        max_cosine_scores = []
        for pred_vector in pred_vectors:
            scores = cosine_similarity(pred_vector, gt_vectors).flatten()
            max_cosine_scores.append(max(scores))

        cosine_precision = sum(max_cosine_scores) / len(max_cosine_scores) if max_cosine_scores else 0
        cosine_recall = sum(max_cosine_scores) / len(gt_answers) if max_cosine_scores else 0
        cosine_f1_score = 2 * cosine_precision * cosine_recall / (cosine_precision + cosine_recall) if (cosine_precision + cosine_recall) != 0 else 0

        return cosine_precision, cosine_recall, cosine_f1_score

    def meteor_eval(self, gt_answers, pred_answers):
        # Tokenize both ground truth and predicted answers
        gt_answers = [str(gt).lower() if not isinstance(gt, list) else [str(item).lower() for item in gt] for gt in gt_answers]

        pred_answers = [nltk.word_tokenize(pred.lower()) for pred in pred_answers]

        max_meteor_scores = []
        for pred in pred_answers:
            # Calculate METEOR score for each ground truth against the prediction
            scores = [meteor_score([gt], pred) for gt in gt_answers]
            max_meteor_scores.append(max(scores))

        meteor_precision = sum(max_meteor_scores) / len(max_meteor_scores) if max_meteor_scores else 0
        meteor_recall = sum(max_meteor_scores) / len(gt_answers) if max_meteor_scores else 0
        meteor_f1_score = 2 * meteor_precision * meteor_recall / (meteor_precision + meteor_recall) if (meteor_precision + meteor_recall) != 0 else 0

        return meteor_precision, meteor_recall, meteor_f1_score


    def eval_all(self, gt_answer, predicted_answer):

        result_el = {}

        if self.args['single']:
            eval_score = self.single_eval(gt_answer, predicted_answer)
            result_el['single_score'] = eval_score
            rouge_score = self.single_rouge_eval(gt_answer, predicted_answer)
            result_el['single_rouge_score'] = rouge_score

        else:

            for eval_method in self.args['eval_methods']:

                if 'rouge' in eval_method:
                    rouge_precision, rouge_recall, rouge_f1_score, rouge_match_scores = self.rouge_eval(gt_answer, predicted_answer)
                    result_el['rouge_precision'] = rouge_precision
                    result_el['rouge_recall'] = rouge_recall
                    result_el['rouge_f1_score'] = rouge_f1_score
                    if self.args['ue_mode'] == 'white':
                        result_el['rouge_match_scores'] = rouge_match_scores

                elif 'exact' in eval_method:
                    mean_precision, mean_recall, mean_f1_score, exact_match_scores = self.exact_match(gt_answer, predicted_answer)
                    result_el['exact_precision'] = mean_precision
                    result_el['exact_recall'] = mean_recall
                    result_el['exact_f1_score'] = mean_f1_score
                    if self.args['ue_mode'] == 'white':
                        result_el['exact_match_scores'] = exact_match_scores


                elif 'bleu' in eval_method:
                    bleu_precision, bleu_recall, bleu_f1_score = self.bleu_eval(gt_answer, predicted_answer)
                    result_el['bleu_precision'] = bleu_precision
                    result_el['bleu_recall'] = bleu_recall
                    result_el['bleu_f1_score'] = bleu_f1_score

                elif 'bertscore' in eval_method:

                    bert_precision, bert_recall, bert_f1_score = self.bertscore_eval(gt_answer, predicted_answer)
                    result_el['bert_precision'] = bert_precision
                    result_el['bert_recall'] = bert_recall
                    result_el['bert_f1_score'] = bert_f1_score

                elif 'edit' in eval_method:
                    edit_precision, edit_recall, edit_f1_score = self.edit_distance_eval(gt_answer, predicted_answer)
                    result_el['edit_precision'] = edit_precision
                    result_el['edit_recall'] = edit_recall
                    result_el['edit_f1_score'] = edit_f1_score

                elif "cosine" in eval_method:
                    cosine_precision, cosine_recall, cosine_f1_score = self.cosine_similarity_eval(gt_answer, predicted_answer)
                    result_el['cosine_precision'] = cosine_precision
                    result_el['cosine_recall'] = cosine_recall
                    result_el['cosine_f1_score'] = cosine_f1_score


                elif "meteor" in eval_method:
                    meteor_precision, meteor_recall, meteor_f1_score = self.meteor_eval(gt_answer, predicted_answer)
                    result_el['meteor_precision'] = meteor_precision
                    result_el['meteor_recall'] = meteor_recall
                    result_el['meteor_f1_score'] = meteor_f1_score

                else:
                    raise Exception(f"Unknown evaluation method: {self.args['eval_methods']}")

        return result_el

    
    

    