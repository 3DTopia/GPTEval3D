"""
Capture the information of a tournamenet among a fixed set of text-to-3D data.

"""
import os
import glob
import json
import tqdm
import time
import random
import itertools
import numpy as np
import os.path as osp
from PIL import Image
from openai import OpenAI
from typing import Literal
from gpteval3d.utils import glide_elo
from gpteval3d.utils import image_utils
from gpteval3d.utils import gpt4v_utils
from multiprocessing import Pool, cpu_count


class T23DMethod():
    def __init__(self, name, path, prompts):
        self.name = name
        self.path = path
        # full set of text prompts
        self.prompts = prompts 
        # the set of prompts 
        self.prompt_ids = set(
            int(f) for f in os.listdir(path) 
            if not f.startswith(".") and
            int(f) in range(len(self.prompts))
        )
        
    def get_image_for_promptid(
        self, pid, seed=0,
        nviews=4,
        img_layout: Literal["horizontal", "vertical", "square"]="square",
        img_type: Literal["rgb", "sfn", "rgb-sfn"] = "rgb-sfn"
    ):
        '''
        all_seeds = [int(s) for s in
                     os.listdir(osp.join(self.path, str(pid)))
                     if not s.startswith(".")
                    ]
        if seed not in all_seeds:
            seed = random.choice(all_seeds)
        '''
        all_rgbs = sorted(
            glob.glob(osp.join(self.path, str(pid), str(seed), "rgb_*.png")))
        all_sfns = sorted(
            glob.glob(osp.join(self.path, str(pid), str(seed), "normal_*.png")))
        assert len(all_rgbs) == len(all_sfns)

        # select views
        if len(all_rgbs) == 0:
            print(os.path.join(self.path, str(pid)), 'does not exists, please manually remove it.')
            exit()
        # if start_view_id < 0:
        #     start_view_id = random.choice(range(len(all_rgbs)))
        view_ids = [
            (i % len(all_rgbs)) for i 
            in range(0, len(all_rgbs), len(all_rgbs) // nviews)
        ]

        # Load images
        if img_type in ["rgb", "rgb-sfn"]:
            rgb_img_lst = []
            for vid in view_ids:
                curr_img_fname = osp.join(
                    self.path, str(pid), str(seed), "rgb_%d.png" % vid)
                curr_img = Image.open(curr_img_fname)
                rgb_img_lst.append(curr_img)
           
        if img_type in ["sfn", "rgb-sfn"]:
            sfn_img_lst = []
            for vid in view_ids:
                curr_img_fname = osp.join(
                    self.path, str(pid), str(seed), "normal_%d.png" % vid)
                curr_img = Image.open(curr_img_fname)
                sfn_img_lst.append(curr_img)
                
        if img_layout == "square":
            ncols = nrows = int(np.sqrt(nviews))
            assert ncols * ncols == nrows * nrows == nviews
            if img_type == "rgb":
                out_img = image_utils.create_square_images(
                    rgb_img_lst, ncols)
            elif img_type == "sfn":
                out_img = image_utils.create_square_images(
                    sfn_img_lst, ncols)
            elif img_type == "rgb-sfn":
                rgb_img = image_utils.create_square_images(
                    rgb_img_lst, ncols)
                sfn_img = image_utils.create_square_images(
                    sfn_img_lst, ncols)
                out_img = np.concatenate([rgb_img, sfn_img], axis=0)
            else:
                raise ValueError
        else:
            raise NotImplemented

        return out_img


class T23DTournament():
    
    def __init__(self, path):
        self.path = path
        # REQUIRED: configuration
        self.cfg = json.load(open(osp.join(path, "config.json")))
        
        # REUIQRED: input prompts    
        self.prompts = json.load(open(osp.join(path, "prompts.json")))
        if "prompt_ids" in self.cfg:
            # self.prompts = [self.prompts[i] for i in self.cfg["prompt_ids"]]
            self.valid_prompt_ids = set(self.cfg["prompt_ids"])
        else:
            self.valid_prompt_ids = set(np.arange(len(self.prompts)).tolist())
        print(self.prompts)
        print("valid_ids", self.valid_prompt_ids)

        # OPTIONAL: comparisons 
        c_fname = osp.join(path, "comparisons.json")
        self.comparisons = {n:[] for n in range(len(self.cfg["criteria"]))}
        if osp.isfile(c_fname):
            self.comparisons = json.load(c_fname)
           
        # Now load methods
        self.methods = {}
        for m_name in sorted(list(os.listdir(osp.join(path, "methods")))):
            if m_name.startswith("."):
                continue
            if "methods" in self.cfg:
                if m_name not in self.cfg["methods"]:
                    continue
            self.methods[m_name] = T23DMethod(
                m_name, osp.join(path, "methods", m_name), self.prompts)
        print(self.methods.keys())
        # TODO: Compute statistics for selecting next games
    
    def create_comparisons_for_tournament(
        self, out_folder,
        budget=500, repeats=2, method_names=None,
    ):
        # First make folder
        if osp.isdir(out_folder):
            os.removedirs(out_folder)
        os.makedirs(out_folder)
        
        # Method from the league to be compared 
        if method_names is None: 
            # compare to all methods
            method_names = sorted(list(self.methods.keys()))
        method_lst = {n:self.methods[n] for n in method_names}

        # Schedule and shuffle pairwise comparisons
        requests = 0 
        scheduled = {(n1, n2): set() for n1, n2 in 
                     itertools.combinations(method_names, 2)} # method_name -> set
        ks = list(scheduled.keys())
        random.shuffle(ks)
        scheduled = {k:scheduled[k] for k in ks}

        print("Collecting pairs")

        scheduled_lst = []
        if self.cfg['ensembles'] is not None:
            # with a specific ensemble strategy
            total_comparisons = np.sum([en['num_comparisons'] for en in self.cfg['ensembles']])
            ensemble_modes = []
            for en in self.cfg['ensembles']:
                ensemble_modes.extend([en] * en['num_comparisons'])
            assert total_comparisons < budget
            pbar = tqdm.tqdm(total=total_comparisons)
            ks = list(scheduled.keys())
            random.shuffle(ks)
            print(len(ensemble_modes), "ensemble modes")
            for _ in ensemble_modes:
                pbar.update(1)
                if len(ks) == 0:
                    ks = list(scheduled.keys())
                    random.shuffle(ks)
                mname_1, mname_2 = ks.pop()
                scheduled_pids = scheduled[(mname_1, mname_2)]
                method1 = method_lst[mname_1]
                method2 = method_lst[mname_2]
                common_prompt_ids = list(
                    self.valid_prompt_ids & (method1.prompt_ids & method2.prompt_ids -  scheduled_pids)
                )
                if len(common_prompt_ids) > 0: # fixme
                    prompt_id = random.choice(common_prompt_ids)
                    scheduled[(mname_1, mname_2)].add(prompt_id)
                    scheduled_lst.append([method_lst[mname_1], method_lst[mname_2], prompt_id])
            pbar.close()
        else:
            # no specific ensemble strategy
            pbar = tqdm.tqdm(total=budget)
            while requests + repeats <= budget:
                pbar.update(requests)
                for (mname_1, mname_2), scheduled_pids in scheduled.items():
                    method1 = method_lst[mname_1]
                    method2 = method_lst[mname_2]
                    common_prompt_ids = list(
                        self.valid_prompt_ids & (method1.prompt_ids & method2.prompt_ids -  scheduled_pids)
                    )
                    if len(common_prompt_ids) > 0:
                        prompt_id = random.choice(common_prompt_ids)
                        scheduled[(mname_1, mname_2)].add(prompt_id)
                        scheduled_lst.append([method_lst[mname_1], method_lst[mname_2], prompt_id])
                        requests += repeats

                    if requests + repeats > budget:
                        break
            ensemble_modes = [None] * len(scheduled)

        info = self.augment_comparisons(
            scheduled_lst, ensemble_modes, repeats, out_folder)
        return info
 
    def augment_comparisons(self, scheduled, ensemble_modes, repeats, out_folder):
        # Augment the comparisons and save it to out_folder
        print("Augmenting pairs")
        info = {}
        cnt = 0
        for (method1, method2, pid), mode in tqdm.tqdm(zip(scheduled, ensemble_modes), total=len(ensemble_modes)):
            for eid in range(repeats):
                comparison = self._create_comparison_(
                    method1, method2, pid, flip=None, mode=mode)
                out_path = osp.join(out_folder, "%d.png" % cnt)
                info[cnt] = {
                    "m1": comparison["m1"],
                    "m2": comparison["m2"],
                    "prompt": comparison["prompt"],
                    "prompt_id": comparison["prompt_id"],
                    "image_path": out_path,
                    "augment_type": None, # default will be just random seed
                    "dimensions": mode["dimensions"],
                    "gpt_prompt": mode["gpt_prompt"]
                }
                # Save_images
                out_pil_image = Image.fromarray(comparison["out_image"])
                # width, height = out_pil_image.size
                # scale = 512 / max(width, height)
                # out_pil_image = out_pil_image.resize((
                #     int(width * scale), int(height * scale)))
                out_pil_image.save(out_path)
                cnt += 1
        json.dump(info, 
                  open(osp.join(out_folder, "question_methods.json"), "w"), indent=4)
        return info
  
     
    def create_comparisons_for_new_method(
        self, new_method, out_folder,
        budget=500, repeats=3, method_names=None
    ):
        # First make folder
        if osp.isdir(out_folder):
            os.removedirs(out_folder)
        os.makedirs(out_folder, exist_ok=True)
        
        # Method from the league to be compared 
        if method_names is None: 
            # compare to all methods
            method_names = sorted(list(self.methods.keys()))
        method_lst = {n:self.methods[n] for n in method_names}

        print("Collecting pairs")
        sceduled = {n: set() for n in method_names}
        scheduled_lst = []
        if self.cfg['ensembles'] is not None:
            total_comparisons = np.sum([en['num_comparisons'] for en in self.cfg['ensembles']])
            ensemble_modes = []
            for en in self.cfg['ensembles']:
                ensemble_modes.extend([en] * en['num_comparisons'])
            assert total_comparisons < budget
            pbar = tqdm.tqdm(total=total_comparisons)
            ks = method_names.copy()
            random.shuffle(ks)
            for _ in ensemble_modes:
                pbar.update(1)
                if len(ks) == 0:
                    ks = method_names.copy()
                    random.shuffle(ks)
                m2 = ks.pop()
                scheduled_pids = sceduled[m2]
                method2 = method_lst[m2]
                common_prompt_ids = list(
                    new_method.prompt_ids & method2.prompt_ids -  scheduled_pids
                )
                if len(common_prompt_ids) > 0:
                    prompt_id = random.choice(common_prompt_ids)
                    sceduled[m2].add(prompt_id)
                    scheduled_lst.append([new_method, method2, prompt_id])
        else:
            # Schedule comparisons 
            requests = 0 
            while requests + repeats <= budget:
                for method_name, method in method_lst.items():
                    common_prompt_ids = list(
                        new_method.prompt_ids & method.prompt_ids - 
                        sceduled[method_name]
                    )
                    if len(common_prompt_ids) > 0:
                        prompt_id = random.choice(common_prompt_ids)
                        sceduled[method_name].add(prompt_id)
                        scheduled_lst.append([new_method, method, prompt_id])
                        
                    requests += repeats
                    if requests + repeats > budget:
                        break 
            ensemble_modes = [None] * len(scheduled_lst)

        # Augment the comparisons and save it to out_folder
        print("Augmenting pairs")
        info = self.augment_comparisons(
            scheduled_lst, ensemble_modes, repeats, out_folder) 
        return info
   
    
    def _create_comparison_(self, method1, method2, prompt_id, flip=None, mode=None, **kwargs):
        if flip is None:
            flip = random.choice([True, False])
        if mode is None:
            nviews = 4
            rgb = True
            normal = True
        else:
            nviews = mode['num_views']
            rgb = mode['rgb'] > 0
            normal = mode['normal'] > 0
        if rgb and normal:
            img_type = 'rgb-sfn'
        elif rgb:
            img_type = 'rgb'
        elif normal:
            img_type = 'sfn'
        else:
            raise ValueError
        m1_image = method1.get_image_for_promptid(prompt_id, nviews=nviews, img_type=img_type)
        m2_image = method2.get_image_for_promptid(prompt_id, nviews=nviews, img_type=img_type)
        if flip:
            m1 = method2.name
            m2 = method1.name
            image1 = m2_image
            image2 = m1_image
        else:
            m1 = method1.name
            m2 = method2.name
            image1 = m1_image
            image2 = m2_image
        out_image = np.concatenate([image1, image2], axis=1)
        return {
            "m1": m1, "m2": m2,
            "image1": image1, "image2": image2,
            "out_image": out_image,
            "prompt": self.prompts[prompt_id],
            "prompt_id": prompt_id
        }

    def init_client(self, api_key, base_url=None):
        global client
        # Initialize OpenAI client here
        client = OpenAI(api_key=api_key, base_url=base_url)

    def single_runner(self, args):
        qid, qid_answered, question_methods, prompt_root, answers_dir, n_choices = args
        qinfo = question_methods[qid]
        answers = []
        if osp.exists(osp.join(answers_dir, f'{qid}_0.txt')):
            for i in range(n_choices):
                with open(osp.join(answers_dir, f'{qid}_{i}.txt'), 'r') as f:
                    ansnwer = f.read()
                    answers.append(ansnwer)
            err = None
        else:
            # print("="*40)
            # print("Question:")
            # print(qinfo)
            # print("-"*40)

            gpt_prompt = qinfo['gpt_prompt'] if 'gpt_prompt' in qinfo else random.choice(self.cfg["gpt_prompts"])
            prompt = gpt4v_utils.make_comparison_prompt(
                qinfo["prompt"],
                osp.join(prompt_root, gpt_prompt)
            )
            print("Prompt:")
            print(qinfo["prompt"])
            print("-" * 40)

            time.sleep(5)
            response, err = gpt4v_utils.call_gpt_4v(
                client, prompt, qinfo["image_path"], n_choices=n_choices)
            print("Response:")
            print(qid)
            print(response)
            print("-" * 40)
            if response is not None:
                for i in range(n_choices):
                    answer = response.choices[i].message.content
                    with open(osp.join(answers_dir, f'{qid}_{i}.txt'), 'w') as f:
                        f.write(answer)
                    answers.append(answer)
            else:
                answers = None
        return qid, qinfo, answers, err

    
    def run_gpt4_judge(self, api_key, base_url, question_methods, out_folder,
                       max_round=10, space_secs=5):

        def handle_success(result):
            print(f"Result received: {result}")

        def handle_error(exception):
            print(f"Error: {exception}")

        # criteria -> [(m1, m2, prompt, result)]
        comparison_results = {i:[] for i in range(len(self.cfg["criteria"]))} 
        comaprisons_out_fname = osp.join(out_folder, "comparisons.json")
        if osp.isfile(comaprisons_out_fname):
            comparison_results = json.load(open(comaprisons_out_fname)) # FIXME error (resume from existing)

        # Book keeping
        # TODO: resume from existing?
        qid_lst = list(question_methods.keys())
        os.makedirs(os.path.join(out_folder, 'answers'), exist_ok=True)
        
        qid_answered = {}
        answered_out_fname = osp.join(out_folder, "answered.json")
        if osp.isfile(answered_out_fname):
            qid_answered = json.load(open(answered_out_fname))
            
        qid_errors = []
        errors_out_fname = osp.join(out_folder, "errors.json")
        if osp.isfile(errors_out_fname):
            qid_errors = json.load(open(errors_out_fname))
        curr_round = 0 

        self.init_client(api_key, base_url)
        # call GPT to run comparisons
        while len(qid_answered) < len(qid_lst) and curr_round < max_round:
            print("Round[%d/%d] Questions:[%d/%d]" 
                  % (curr_round, max_round, len(qid_answered), len(qid_lst)))
            todo_qid_lst = [qid for qid in qid_lst if qid not in qid_answered]

            use_multiprocess = base_url is not None

            if not use_multiprocess:
                sp_todo_lst = todo_qid_lst
                todo_result_lst = []
                while len(sp_todo_lst) > 0:
                    failed_qid_lst = []
                    for qid in tqdm.tqdm(sp_todo_lst):
                        answer_dir = os.path.join(out_folder, 'answers')
                        qid, qinfo, answers, err = self.single_runner((qid, qid_answered, question_methods, self.path, answer_dir, 3))
                        if answers is not None:
                            todo_result_lst.append((qid, qinfo, answers, err))
                        else:
                            failed_qid_lst.append(qid)
                    sp_todo_lst = failed_qid_lst 
                todo_result_lst = sorted(todo_result_lst, key=lambda x: x[0])
            else:
                pool = Pool(processes=min(8, len(todo_qid_lst), cpu_count()))
                mp_todo_lst = todo_qid_lst
                todo_result_lst = []
                while len(mp_todo_lst) > 0:
                    failed_qid_lst = []
                    try:
                        # mp_todo_result_lst = pool.map(self.single_runner, [(qid, qid_answered, question_methods, self.path) for qid in todo_qid_lst])
                        mp_todo_result_lst = []
                        args_list = [(qid, qid_answered, question_methods, self.path,
                                      os.path.join(out_folder, 'answers'), 3) for qid in mp_todo_lst]
                        for args in args_list:
                            res = pool.apply_async(self.single_runner, (args,), callback=handle_success, error_callback=handle_error)
                            mp_todo_result_lst.append(res)
                        pool.close()
                        pool.join()
                        # Convert managed list back to normal list and save
                        tmp_todo_result_lst = [res.get(timeout=10) for res in mp_todo_result_lst if res.successful()]
                        tmp_todo_result_lst = sorted(tmp_todo_result_lst, key=lambda x: x[0])
                    except KeyboardInterrupt:
                        print("KeyboardInterrupt detected. Terminating all processes...")
                        if pool is not None:
                            pool.terminate()
                            pool.join()
                        raise
                    finally:
                        if pool is not None and not hasattr(pool, '_state') or pool._state != 2:
                            pool.close()
                            pool.join()
                        tmp_todo_result_lst = [res.get(timeout=10) for res in mp_todo_result_lst if res.successful()]
                        tmp_todo_result_lst = sorted(tmp_todo_result_lst, key=lambda x: x[0])
                    for d_ in tmp_todo_result_lst:
                        if d_[2] is not None:
                            todo_result_lst.append(d_)
                        else:
                            failed_qid_lst.append(d_[0])
                    todo_result_lst = sorted(todo_result_lst, key=lambda x: x[0])
                    print('failed_qid_lst:', failed_qid_lst)
                    mp_todo_lst = failed_qid_lst

            for qid, qinfo, answers, err in todo_result_lst:
                if answers is None:
                    qid_errors.append((qid, str(err)))
                    continue
                answer_lst = self.parse_response(answers, qinfo)
                # print("Answer:")
                # print(answer_lst)
                # print("-"*40)
                if len(answer_lst) > 0:
                    # Book keeping
                    qid_answered[qid] = {
                        "question": qinfo,
                        "answer": answer_lst, # n_choices x n_criteria
                        "response": str(answers)
                    }
                    # Final output
                    for ans in answer_lst:
                        for cid, a in enumerate(ans):
                            if 'dimensions' in qinfo:
                                dim = qinfo['dimensions'][cid]
                            else:
                                dim = cid
                            comparison_results[dim].append({
                                "m1": qinfo["m1"],
                                "m2": qinfo["m2"],
                                "result": a
                            })
                else:
                    qid_errors.append((qid, str(answers)))
            # Save to the out folder
            curr_round += 1
            json.dump(qid_answered, open(answered_out_fname, "w"), indent=4)
            json.dump(qid_errors, open(errors_out_fname, "w"), indent=4)
            json.dump(comparison_results, open(comaprisons_out_fname, "w"), indent=4)
            
        # Finally create comparisons
        return comparison_results, {
            "answered": qid_answered, 
            "errors": qid_errors
        }
                
    def parse_response(self, answers, info):
        # https://platform.openai.com/docs/api-reference/chat/object
        out_lst = []
        # for response_choice in response.choices:
        #     if response_choice.finish_reason == "stop":
        #         response_text = response_choice.message.content
        #     else:
        #         print(response_choice)
        #         breakpoint()
        for ans in answers:
            last_line = ans.strip(" \n").split("\n")[-1]
            try:
                out = [
                    int(x) for x in last_line.strip(" \n").split(" ")]
                assert len(out) == len(info["dimensions"])
                assert all([x in [1,2,3] for x in out])
                # Transform answer format
                def f(x):
                    if x == 1:
                        return -1
                    elif x == 2:
                        return 1
                    elif x == 3 or x == 4:
                        return 0
                    raise ValueError
                out = [f(x) for x in out]
                out_lst.append(out)
            except:
                # out = [False] * len(self.cfg["criteria"])
                pass

        return out_lst
      
            
    def get_elo_scores_for_new_method(self, new_method, comparisons):
        """Finding ELO in a league of T23D methods already scored with ELOs.

        Args:
            new_method: T23DMethod object.
            comparisons (dictionary): Mapping from criteria to list of 
                comparison results. These results shuold be mostly concerned
                about the new methods.
        """
        all_scores = {}
        all_existing_scores = self.cfg["scores"]  # criteria -> method -> score
        for cid in range(len(self.cfg["criteria"])):
            curr_existing_scores = all_existing_scores[cid]
            methods = list(curr_existing_scores.keys()) + [new_method.name]
            init_elo = [
                float(curr_existing_scores[m] if m in curr_existing_scores 
                      else 1000.) 
                for m in methods]
            freeze = [(m in curr_existing_scores) for m in methods]
            curr_comparison = (comparisons[cid] if cid in comparisons 
                               else comparisons[str(cid)])
            new_scores, aux = glide_elo.compute_glide_elo(
                methods, curr_comparison, results=None, freeze=freeze, 
                init_elo=init_elo, return_aux=True
            )
            print("=" * 40)
            print("Criteria: %d" % cid)
            for n, e in aux["ranking"]:
                print("%s\t%4.3f" % (n, e))
            # print(aux["ranking"])
            print("=" * 40)
            
            all_scores[cid] = {
                "scores": new_scores,
                "methods": aux["methods"],
                "ranking": aux["ranking"]
            }
        return all_scores
   
   
    def get_elo_scores(self, comparisons, anchor='dreamfusion',
                       init_from_exist_scores=False):
        """Get

        Args:
            comparisons (_type_): _description_
            anchor (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        all_scores = {}
        # all_existing_scores = self.cfg["scores"]  # criteria -> method -> score
        methods = list(m.name for m in self.methods.values()) 
        for cid in range(len(self.cfg["criteria"])):
            if init_from_exist_scores:
                curr_existing_scores = self.cfg["scores"][cid]
                init_elo = [
                    float(curr_existing_scores[m] if m in curr_existing_scores 
                        else 1000.) 
                    for m in methods]
            else:
                init_elo = [1000. for m in methods]
            freeze = [(m == anchor) for m in methods]
            curr_comparison = (comparisons[cid] if cid in comparisons 
                               else comparisons[str(cid)])
            new_scores, aux = glide_elo.compute_glide_elo(
                methods, curr_comparison, results=None, freeze=freeze, 
                init_elo=init_elo, return_aux=True
            )
            print("=" * 40)
            print("Criteria: %d %s\n\t%s" 
                  % (cid, self.cfg["criteria"][cid][0], 
                     self.cfg["criteria"][cid][1]))
            for n, e in aux["ranking"]:
                print("%s\t%4.3f" % (n, e))
            # print(aux["ranking"])
            print("=" * 40)
            
            all_scores[cid] = {
                "scores": new_scores,
                "methods": aux["methods"],
                "ranking": aux["ranking"]
            }
        return all_scores
