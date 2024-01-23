import os
import time
import json
import argparse
import os.path as osp
from utils import t23d_tournament


def parse_args():
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
        '-b', '--budget', default=10, type=int,
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


def score_with_new_tournament(args): 
    # Load tournament information
    tournament = t23d_tournament.T23DTournament(args.tournament)
    if args.comparisons is None: 
        args.comparisons = osp.join(
            "comparisons",
            "full-%s" % (time.strftime('%Y-%b-%d-%H-%M-%S')))
        question_methods = tournament.create_comparisons_for_tournament(
            args.comparisons,
            budget=args.budget, 
            repeats=args.repeats, 
        )
        print("Create %d comparisons." % len(question_methods))
        if len(question_methods) == 0:
            print("Not enough budget.")
            return
    else:
        print("Use comparison results from %s" % args.comparisons)
        question_methods = json.load(
            open(osp.join(args.comparisons, "question_methods.json")))

    # Run GPT-4V
    apikey = args.apikey
    if apikey is None:
        apikey = os.environ["OPENAI_API_KEY"]
    results, info = tournament.run_gpt4_judge(
        apikey, args.baseurl, question_methods,
        args.comparisons, max_round=10)

    # Compute ELO scores
    all_scores = tournament.get_elo_scores(results)
    if args.output is None:
        args.output = args.comparisons
    os.makedirs(args.output, exist_ok=True)
    json.dump(all_scores, open(osp.join(args.output, "scores.json"), "w"), indent=4)
    

def score_with_existing_tournament(args):
    
    # Load tournament information
    tournament = t23d_tournament.T23DTournament(args.tournament)
    # Load new method information
    # All comparisons will be saved
    method_name = osp.basename(args.method.strip("/"))
    new_method = t23d_tournament.T23DMethod(
        method_name, args.method, tournament.prompts)
    
    # Schedule games between new method and the existing ones
    if args.comparisons is None:
        args.comparisons = osp.join(
            args.tournament, "comparisons",
            "%s-%s" % (method_name, time.strftime('%Y-%b-%d-%H-%M-%S')))
        question_methods = tournament.create_comparisons_for_new_method(
            new_method, 
            args.comparisons,
            budget=args.budget, 
            repeats=args.repeats, 
            method_names=None
        )
        print("Create %d comparisons." % len(question_methods))
        if len(question_methods) == 0:
            print("Not enough budget.")
            return
    else:
        print("Use comparison results from %s" % args.comparisons)
        question_methods = json.load(
            open(osp.join(args.comparisons, "question_methods.json")))

    # Run GPT-4V
    apikey = args.apikey
    if apikey is None:
        apikey = os.environ["OPENAI_API_KEY"]
    results, info = tournament.run_gpt4_judge(
        apikey, args.baseurl, question_methods,
        args.comparisons, max_round=10)
    
    # Compute ELO scores
    all_scores = tournament.get_elo_scores_for_new_method(new_method, results)
    if args.output is None:
        args.output = args.comparisons
    os.makedirs(args.output, exist_ok=True)
    json.dump(all_scores, open(osp.join(args.output, "scores.json"), "w"), indent=4)
    
def test_elo(args):
    # Load tournament information
    tournament = t23d_tournament.T23DTournament(args.tournament)
    # Compute ELO scores
    results = json.load(open(args.gpt_results, 'r'))
    all_scores = tournament.get_elo_scores(results)
    os.makedirs(args.output, exist_ok=True)
    json.dump(all_scores, open(osp.join(args.output, "scores.json"), "w"))

if __name__ == '__main__':
    args = parse_args()
    if args.eval == "new_method":
        score_with_existing_tournament(args)
    elif args.eval == "tournament":
        score_with_new_tournament(args)
    elif args.eval == "score-comparisons":
        raise NotImplemented
    elif args.eval == "load_previous_gpt":
        test_elo(args)
    else:
        raise ValueError