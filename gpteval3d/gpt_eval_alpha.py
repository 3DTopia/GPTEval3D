import os
import time
import json
import argparse
import os.path as osp
from gpteval3d.utils import t23d_tournament


def __parse_args():
    parser = argparse.ArgumentParser(
        prog='python gpt_eval_alph.py', 
        description='Evaluation metrics based on GPT-4V.',
        epilog='')
    
    parser.add_argument(
        '-e', '--eval', default="new_method",
        help="Evaluation options. Choices: new_method.")
    parser.add_argument(
        '-k', '--apikey', default=None,
        help="API key. If None, will use environment variable OPENAI_API_KEY.")
    parser.add_argument(
        '-u', '--baseurl', default=None,
        help="API Base URL. Default is None.")
    parser.add_argument(
        '-m', '--method', default=None, 
        help="Folder storing the information of a method.")
    # TODO: Explain question_methods
    parser.add_argument(
        '-q', '--question_methods', default=None,
        help="TODO: Explain question_methods")
    parser.add_argument(
        '-t', '--tournament', default=None, 
        help="Folder storing the information of a tournament.")
    parser.add_argument(
        '-c', '--comparisons', default=None, 
        help="Folder storing the information of a list of comparisons.")
    parser.add_argument(
        '-b', '--budget', default=1000, type=int,
        help="Number of requests budgeted for the evaluation task.")
    parser.add_argument(
        '-o', '--output', default=None, 
        help="Output folder.")
    parser.add_argument(
        '--repeats', default=1,
        help="Number of times we perturbs the input prompt to GPT judge.")
    parser.add_argument(
        '--gpt-results', default=None,
        help="Existing GPT results in json format.")
    args = parser.parse_args()
    return args

def score_with_new_tournament(tournament_folder, apikey=None, baseurl=None, comparisons=None, budget=1000, repeats=1, output=None):
    # Load tournament information
    tournament = t23d_tournament.T23DTournament(tournament_folder)
    if comparisons is None: 
        comparisons = osp.join(
            "comparisons",
            "full-%s" % (time.strftime('%Y-%b-%d-%H-%M-%S')))
        question_methods = tournament.create_comparisons_for_tournament(
            comparisons,
            budget=budget,
            repeats=repeats,
        )
        print("Create %d comparisons." % len(question_methods))
        if len(question_methods) == 0:
            print("Not enough budget.")
            return
    else:
        print("Use comparison results from %s" % comparisons)
        question_methods = json.load(
            open(osp.join(comparisons, "question_methods.json")))

    # Run GPT-4V
    apikey = os.environ["OPENAI_API_KEY"] if apikey is None else apikey
    if apikey is None:
        raise ValueError("OpenAI API key is not provided.")
    results, info = tournament.run_gpt4_judge(
        apikey, baseurl, question_methods,
        comparisons, max_round=10)

    # Compute ELO scores
    all_scores = tournament.get_elo_scores(results)
    if output is None:
        output = comparisons
    os.makedirs(output, exist_ok=True)
    json.dump(all_scores, open(osp.join(output, "scores.json"), "w"), indent=4)

def score_with_existing_tournament(tournament_folder, method_folder, apikey=None, baseurl=None, comparisons=None, budget=1000, repeats=1, output=None):
    
    # Load tournament information
    tournament = t23d_tournament.T23DTournament(tournament_folder)
    # Load new method information
    # All comparisons will be saved
    method_name = osp.basename(method_folder.strip("/"))
    new_method = t23d_tournament.T23DMethod(
        method_name, method_folder, tournament.prompts)
    
    # Schedule games between new method and the existing ones
    if comparisons is None:
        comparisons = osp.join(
            tournament_folder, "comparisons",
            "%s-%s" % (method_name, time.strftime('%Y-%b-%d-%H-%M-%S')))
        question_methods = tournament.create_comparisons_for_new_method(
            new_method, 
            comparisons,
            budget=budget, 
            repeats=repeats, 
            method_names=None
        )
        print("Create %d comparisons." % len(question_methods))
        if len(question_methods) == 0:
            print("Not enough budget.")
            return
    else:
        print("Use comparison results from %s" % comparisons)
        question_methods = json.load(
            open(osp.join(comparisons, "question_methods.json")))

    # Run GPT-4V
    apikey = os.environ["OPENAI_API_KEY"] if apikey is None else apikey
    if apikey is None:
        raise ValueError("OpenAI API key is not provided.")
    results, info = tournament.run_gpt4_judge(
        apikey, baseurl, question_methods,
        comparisons, max_round=10)
    
    # Compute ELO scores
    all_scores = tournament.get_elo_scores_for_new_method(new_method, results)
    if output is None:
        output = comparisons
    os.makedirs(output, exist_ok=True)
    json.dump(all_scores, open(osp.join(output, "scores.json"), "w"), indent=4)

def test_elo(tournament_folder, gpt_results, output):
    # Load tournament information
    tournament = t23d_tournament.T23DTournament(tournament_folder)
    # Compute ELO scores
    results = json.load(open(gpt_results, 'r'))
    all_scores = tournament.get_elo_scores(results)
    os.makedirs(output, exist_ok=True)
    json.dump(all_scores, open(osp.join(args.output, "scores.json"), "w"))

if __name__ == '__main__':
    args = __parse_args()
    if args.eval == "new_method":
        score_with_existing_tournament(args.tournament, args.method, args.apikey, args.baseurl, args.comparisons, args.budget, args.repeats, args.output)
    elif args.eval == "tournament":
        score_with_new_tournament(args.tournament, args.apikey, args.baseurl, args.comparisons, args.budget, args.repeats, args.output)
    elif args.eval == "score-comparisons":
        raise NotImplementedError
    elif args.eval == "load_previous_gpt":
        test_elo(args.tournament, args.gpt_results, args.output)
    else:
        raise ValueError