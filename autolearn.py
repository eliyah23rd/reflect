'''
This module implements a version of auto-reflect.py that
is modular and configurable (no-code)

The system depends on modules and nodes. A module
defies functionality and a node is normally a wrapper
of a module that embeds the module in a control flow 
graph.

Details to be found in https://www.reddit.com/r/ExploringGPT

'''

import os
from time import sleep
import numpy as np
import pickle
import threading
import asyncio
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

'''
Configuration params
'''

c_gpt_engine = 'gpt-3.5-turbo'
c_gpt_embed_engine = 'text-embedding-ada-002'

c_num_retries = 5
c_openai_timeout = 180
c_embed_len = 1536
c_batch_size = 50
c_posts_batch_size = 40
c_num_closest = 3 # should be 10

fname_extra_history = 'extra_history.txt'

def input_digits(prompt):
    inp = input(prompt).strip()
    if not inp.isdigit():
        print('Sorry. The input must consist entirely of digits.')
        return 1, None
    return 0, inp


def parse_mod_file(file_name):
    sections = {}
    current_section = None
    with open('modules/'+file_name, 'rt') as file:
        for line in file:
            line = line.strip('\n')
            if line.startswith('>>'):
                current_section = line[2:]
                sections[current_section] = ''
            elif current_section is not None:
                if len(sections[current_section]) > 0:
                    sections[current_section] += '\n' + line
                else:
                    sections[current_section] += line
    return sections

'''
Retrieves the posts from the pickle file
'''

def get_posts_texts(posts_fname):
    with open(f'{posts_fname}.pkl', 'rb') as fh:
        lposts = pickle.load(fh)
    pass
    lpost_texts = []
    for ipost, apost  in enumerate(lposts):
        lpost_texts.append(apost['text'])
    return lposts, lpost_texts

'''
list_posts allows the user to read the posts and returns to the main loop.
If an invalid number is input by the user, we will still return to the main loop.
'''
def list_posts(lposts):
    print('Here are the titles of the posts.')
    for ipost, post in enumerate(lposts):
        title = lposts[ipost]['title']
        print(f'{ipost+1}. {title}')
    try:
        i_sel_post = int(input('Please enter the number of the post you\'d like to see.\n'))
    except ValueError:
        print('I\'m sorry, that is not a valid number.')
        return
    if i_sel_post < 1 or i_sel_post > len(lposts):
        print('I\'m sorry, that is not a valid number.')
        return
    print(lposts[i_sel_post-1]['text'])
    return

'''
A simple function that enables using timeout when getting the vector embedding from
OpenAI. For some reason, this requests sometimes times out.
'''
def raise_timeout():
  global b_timed_out
  b_timed_out = True
  print('raising timeout')
  raise TimeoutError

'''
Simple verfication function used to make sure that the GPT module returned exactly the
format required.
In this case requires a response of yes or no only but can tolerate a '.' at the end
'''
def yes_or_no(from_gpt):
    if from_gpt.lower() in ['yes', 'no']:
        return True, from_gpt.lower()
    return False, from_gpt

'''
Verfication function used to make sure that the GPT module returned exactly the
format required.
In this case requires a response of 1 to 10 only but can tolerate a '.' at the end
'''
def one_to_ten(from_gpt):
    from_gpt = from_gpt.strip().strip('.')
    try:
        val = int(from_gpt)
    except ValueError:
        return False,-1

    return True, val



'''
Function that embeds any text as a vector of some 1500 floats.
Used to build the initial database and to embed any user query.
'''
def get_embeds(ltexts):
    for itry in range(c_num_retries):
        global b_timed_out
        b_timed_out = False
        timer = threading.Timer(c_openai_timeout, raise_timeout)
        timer.start()
        try:
            response = openai.Embedding.create(
                model=c_gpt_embed_engine,
                input=ltexts
            )
            timer.cancel()
            if b_timed_out:
                print(f'uncaught timeout error on try {itry}')
                continue
            return [response["data"][i]['embedding'] for i in range(len(ltexts))]
        except openai.error.RateLimitError:
            timer.cancel()
            sleep(5)
            return get_embeds(ltexts)
        except TimeoutError as e:
            print(f'timeout error on try {itry}')
            continue
        except openai.APIError:
            timer.cancel()
            print(f'api error on try {itry}')
            continue
        except openai.InvalidRequestError:
            timer.cancel()
            print(f'api invalid request error on try {itry}')
            continue
        # Keep the following disable because it masks real problems and
        # keyboard ^C but use if you have a very large batch that you
        # don't want to fail under any circumstances.
        # except:
        #     print(f'generic unrecognised error on try {itry}')
        #     continue

    return [[0.0] * c_embed_len]

'''
Simple helper function that clears the previous status message and creates a new status message.
Status messages are important because the whole process takes much longer than users are used
to when using generic GPT
'''
def print_status(msg):
    print('\x1b[2K\r', end='')
    print(msg, end='\r')

'''
This is the complete cosine similarity function, even though in this case the vectors are already normalized
'''
def cosine_similarity(vec1, vec2):
    # calculate dot product
    dot_product = vec2 @ vec1
    # calculate magnitudes
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2, axis=1)
    # calculate cosine similarity
    cosine_similarity = dot_product / (magnitude1 * magnitude2)
    return cosine_similarity

'''
This is the core function that gets a response from the ChatGPT API
It is written as an async function, so that it can be called multiple times simultaneously
'''
async def chatgpt_req(  lprompts=[], status_msg = '', engine=c_gpt_engine, temp=0.7, 
                        top_p=1.0, tokens=256, freq_pen=0.0, pres_pen=0.0, stop=["\n"]):
    if len(status_msg) > 0:
        print_status(status_msg)

    lmessages = [{"role": role, "content": prompt} for role, prompt in lprompts]
    response = await openai.ChatCompletion.acreate(
        model=c_gpt_engine,
        messages=lmessages,
        temperature=temp,
        max_tokens=tokens,
        top_p=top_p,
        frequency_penalty=freq_pen,
        presence_penalty=pres_pen,
        # stop=stop # ChatCompletion seems to have disabled this option for now and produces an error if you include it.
    )
    text = response['choices'][0]['message']['content'].strip().strip('.')
    return text

'''
Helper function that builds the gpt response and makes one extra try
in case the GPT response was not in exactly the requested format.
'''
async def get_gpt_response(lprompts, status_msg = '', verify='', on_fail='', num_iters_left=0):
    if len(status_msg) > 0:
        print_status(status_msg)
        
    response = await chatgpt_req(lprompts)
    if len(verify) > 0:
        response.replace('\"', '\'')
        bvalid, response = eval(verify.strip() + '(\"' + response + '\")')
    else:
        bvalid = True
    if num_iters_left == 0 or bvalid:
        return 0, response
    else:
        lprompts += [('assistant', response), ('user', on_fail)]
        return await get_gpt_response(lprompts, verify, on_fail, num_iters_left=num_iters_left-1)

async def get_multi_response(system_role, question, ltexts, user_template, status_msg = ''):
    if len(status_msg) > 0:
        print_status(status_msg)
        
    # example template:  'Post:\n{Background}\nQuestion:\n{Question}'
    # Template must include two variable called Background and Question.
    lprompts = [[("system", system_role), ("user", user_template.format(\
            Background = text, Question = question))] for text in ltexts]
    tasks = [chatgpt_req(aprompt) for aprompt in lprompts]
    lanswers = await asyncio.gather(*tasks)
    return 0, lanswers

async def process_multi_texts(system_role, question, lpost_texts, idxs_best, user_template, status_msg = ''):
    ltexts = [lpost_texts[idx] for idx in idxs_best]
    return await get_multi_response(system_role, question, ltexts, user_template, status_msg)



def mod_test_add(sn1, sn2):
    try:
        n1 = int(sn1)
        n2 = int(sn2)
    except ValueError:
        return 1, 'Error'
    
    return 0, str(n1 + n2)
    pass

def mod_input():
    pass

def mod_print(prompt, inp):
    if len(inp) < 1:
        print(prompt)
    else:
        print(prompt.format(inp=inp))



def mod_load_posts(post_fname):
    nd_embeds = np.load(f'{post_fname}_embeds.npy')
    # uncomment the following line if you were forced to abandon some posts during embed
    # l_b_valid_embeds = [val > 0.9 for val in np.linalg.norm(nd_embeds, axis=1)]
    lposts, lpost_texts = get_posts_texts(post_fname)
    num_posts = len(lpost_texts)
    return 0, nd_embeds, lposts, lpost_texts, num_posts

def mod_setup_question(lposts):
    question = input('Please enter a question about Eliyah\'s posts or type LIST to see posts or END to exit.\n')
    if question.lower() == "end":
        return 1, None
    if question.lower() == "list":
        list_posts(lposts[:-3])
        return 2, None
    print('\n\n')
    return 0, question

def mod_embed_and_compare(question, nd_embeds):
    print_status('creating embed for user input...')
    nd_qembed = get_embeds([question])[0]
    nd_scores = cosine_similarity(nd_qembed, nd_embeds)
    nd_idxs_best = nd_scores.argsort()[-c_num_closest:]
    return 0, nd_qembed, nd_scores, nd_idxs_best

'''
def do_verify():
        print_status('checking with GPT which post is relevant ...')
        l_score_strs, b_extra_validation = await verify_relevant(nd_scores, nd_idxs_best, lpost_texts, question)
        nd_relevance_scores = np.zeros(len(l_score_strs))
        for iscore, score_str in enumerate(l_score_strs):
            try:
                nd_relevance_scores[iscore] = float(score_str)
            except ValueError:
                nd_relevance_scores[iscore] = -1
        i_argmax = np.argmax(nd_relevance_scores)
'''
        
def verified_chat():
    flow = {}
    flow['start_node'] = 'load_posts'
    flow['load_posts'] = {}
    flow['load_posts']['mod'] = 'load_embeds_mod.txt'
    flow['load_posts']['fname'] = ['philposts']
    flow['load_posts']['next_node_options'] = ['setup_question']
    flow['setup_question'] = {}
    flow['setup_question']['mod'] = 'setup_question_mod.txt'
    flow['setup_question']['lposts'] = ['load_posts', 'lposts']
    flow['setup_question']['next_node_options'] = ['embed_and_compare', '', 'setup_question']
    flow['embed_and_compare'] = {}
    flow['embed_and_compare']['mod'] = 'embed_and_compare_mod.txt'
    flow['embed_and_compare']['question'] = ['setup_question', 'question']
    flow['embed_and_compare']['nd_embeds'] = ['load_posts', 'nd_embeds']
    flow['embed_and_compare']['next_node_options'] = ['get_extraq_history']
    flow['get_extraq_history'] = {}
    flow['get_extraq_history']['mod'] = 'get_extraq_history_mod.txt'
    flow['get_extraq_history']['fname_extra_history'] = ['extra_history.txt']
    flow['get_extraq_history']['next_node_options'] = ['ask_to_verify']
    flow['ask_to_verify'] = {}
    flow['ask_to_verify']['mod'] = 'ask_to_verify_mod.txt'
    flow['ask_to_verify']['question'] = ['setup_question', 'question']
    flow['ask_to_verify']['history'] = ['get_extraq_history', 'history']
    flow['ask_to_verify']['next_node_options'] = ['yes_no_branch1']
    flow['yes_no_branch1'] = {}
    flow['yes_no_branch1']['mod'] = 'yes_no_branch_mod.txt'
    flow['yes_no_branch1']['input'] = ['ask_to_verify', 'response']
    flow['yes_no_branch1']['next_node_options'] = ['get_relevance', 'prep_similarity_scores']
    flow['prep_similarity_scores'] = {}
    flow['prep_similarity_scores']['mod'] = 'prep_similarity_scores_mod.txt'
    flow['prep_similarity_scores']['nd_scores'] = ['embed_and_compare', 'nd_scores']
    flow['prep_similarity_scores']['nd_idxs_best'] = ['embed_and_compare', 'nd_idxs_best']
    flow['prep_similarity_scores']['next_node_options'] = ['get_top_relevance_score']
    flow['get_relevance'] = {}
    flow['get_relevance']['mod'] = 'get_relevance_mod.txt'
    flow['get_relevance']['question'] = ['setup_question', 'question']
    flow['get_relevance']['lpost_texts'] = ['load_posts', 'lpost_texts']
    flow['get_relevance']['idxs_best'] = ['embed_and_compare', 'nd_idxs_best']
    flow['get_relevance']['next_node_options'] = ['get_top_relevance_score']
    flow['get_top_relevance_score'] = {}
    flow['get_top_relevance_score']['mod'] = 'get_top_relevance_score_mod.txt'
    flow['get_top_relevance_score']['lscores_similarity'] = ['prep_similarity_scores', 'relevance_scores']
    flow['get_top_relevance_score']['lscores_relevance'] = ['get_relevance', 'lscores']
    flow['get_top_relevance_score']['next_node_options'] = ['less_than_branch1']
    flow['less_than_branch1'] = {}
    flow['less_than_branch1']['mod'] = 'less_than_branch_mod.txt'
    flow['less_than_branch1']['input'] = ['get_top_relevance_score', 'score_best']
    flow['less_than_branch1']['thresh'] = [4]
    flow['less_than_branch1']['next_node_options'] = ['print_apology1', 'extract_sections1']
    flow['print_apology1'] = {}
    flow['print_apology1']['mod'] = 'print_mod.txt'
    flow['print_apology1']['msg'] = ['I\'m sorry. None of Eliyah\'s posts address your question.']
    flow['print_apology1']['inp'] = ['']
    flow['print_apology1']['next_node_options'] = ['setup_question']
    flow['extract_sections1'] = {}
    flow['extract_sections1']['mod'] = 'extract_sections_mod.txt'
    flow['extract_sections1']['question'] = ['setup_question', 'question']
    flow['extract_sections1']['ibest'] = ['get_top_relevance_score', 'ibest']
    flow['extract_sections1']['lpost_texts'] = ['load_posts', 'lpost_texts']
    flow['extract_sections1']['nd_idxs_best'] = ['embed_and_compare', 'nd_idxs_best']
    flow['extract_sections1']['next_node_options'] = ['answer_question1']
    flow['answer_question1'] = {}
    flow['answer_question1']['mod'] = 'answer_question_mod.txt'
    flow['answer_question1']['question'] = ['setup_question', 'question']
    flow['answer_question1']['relevant_sections'] = ['extract_sections1', 'relevant_sections']
    # flow['answer_question1']['next_node_options'] = ['print_response']
    flow['answer_question1']['next_node_options'] = ['verify_response1']
    flow['verify_response1'] = {}
    flow['verify_response1']['mod'] = 'verify_response_mod.txt'
    flow['verify_response1']['question'] = ['setup_question', 'question']
    flow['verify_response1']['relevant_sections'] = ['extract_sections1', 'relevant_sections']
    flow['verify_response1']['response'] = ['answer_question1', 'response']
    flow['verify_response1']['next_node_options'] = ['less_than_branch2']
    flow['less_than_branch2'] = {}
    flow['less_than_branch2']['mod'] = 'less_than_branch_mod.txt'
    flow['less_than_branch2']['input'] = ['verify_response1', 'score']
    flow['less_than_branch2']['thresh'] = [3]
    flow['less_than_branch2']['next_node_options'] = ['print_apology2', 'print_response']
    flow['print_apology2'] = {}
    flow['print_apology2']['mod'] = 'print_mod.txt'
    flow['print_apology2']['msg'] = ['I\'m sorry. I cannot verify my response using any of the text in any of Eliyah\'s posts.']
    flow['print_apology2']['inp'] = ['']
    flow['print_apology2']['next_node_options'] = ['setup_question']
    flow['print_response'] = {}
    flow['print_response']['mod'] = 'print_mod.txt'
    flow['print_response']['msg'] = ['GPT\'s answer: {inp}']
    flow['print_response']['inp'] = ['answer_question1', 'response']
    flow['print_response']['next_node_options'] = ['get_feedback']
    flow['get_feedback'] = {}
    flow['get_feedback']['mod'] = 'input_text_mod.txt'
    flow['get_feedback']['prompt'] = ['Please tell me your whether you are satisfied with this answer.']
    flow['get_feedback']['next_node_options'] = ['eval_feedback1']
    flow['eval_feedback1'] = {}
    flow['eval_feedback1']['mod'] = 'eval_feedback_mod.txt'
    flow['eval_feedback1']['feedback_question'] = ['get_feedback', 'prompt']
    flow['eval_feedback1']['user_feedback'] = ['get_feedback', 'output']
    flow['eval_feedback1']['next_node_options'] = ['store_feedback1']
    flow['store_feedback1'] = {}
    flow['store_feedback1']['mod'] = 'store_feedback_mod.txt'
    flow['store_feedback1']['fname_extra_history'] = ['get_extraq_history', 'fname_extra_history']
    flow['store_feedback1']['extra_validation_response'] = ['ask_to_verify', 'response']
    flow['store_feedback1']['question'] = ['setup_question', 'question']
    flow['store_feedback1']['feedback_score'] = ['eval_feedback1', 'feedback_score']
    flow['store_feedback1']['next_node_options'] = ['setup_question']


    return flow

def test_add():
    flow = {}
    flow['start_node'] = 'inp1'
    flow['inp1'] = {}
    flow['inp1']['mod'] = 'input_int_mod.txt'
    flow['inp1']['prompt'] = ['Please enter the first number.']
    flow['inp1']['next_node_options'] = ['inp2', 'inp1'] # 0 == success
    flow['inp2'] = {}
    flow['inp2']['mod'] = 'input_int_mod.txt'
    flow['inp2']['prompt'] = ['Please enter the second number.']
    flow['inp2']['next_node_options'] = ['add1', 'inp2'] # 0 == success
    flow['add1'] = {}
    flow['add1']['mod'] = 'test_add_mod.txt'
    flow['add1']['i1'] = ['inp1', 'output']
    flow['add1']['i2'] = ['inp2', 'output']
    flow['add1']['next_node_options'] = ['print1']
    flow['print1'] = {}
    flow['print1']['mod'] = 'print_mod.txt'
    flow['print1']['inp'] = ['add1', 'sum']
    flow['print1']['msg'] = ["Your result is {inp}."]
    flow['print1']['next_node_options'] = ['inp1']

    return flow
    
async def exec_flow(conf_nodes):
    flow = conf_nodes()
    nodes = {}
    curr_node_name = flow['start_node']
    while True:
        curr_node = {}
        nodes[curr_node_name] = curr_node
        curr_node['flow'] = flow[curr_node_name]
        curr_node = await eval_module(flow, nodes, curr_node)
        if curr_node is None:
            break
        elif 'next_node' not in curr_node:
            print('flow terminated due to no new requested node.')
            break
        elif curr_node['next_node'] not in flow:
            if len(curr_node['next_node']) > 0:
                print(f"Unknown node {curr_node['next_node']} requested as next node")
            break
        else:
            curr_node_name = curr_node['next_node']
            
async def eval_module(flow, nodes, node : dict = None):
    fname = node['flow']['mod']
    d_mod = parse_mod_file(fname)
    l_input_vars = [name.strip() for name in d_mod['inputs'].split(',')]
    for ivar, input_var in enumerate(l_input_vars):
        input_var_data = node['flow'][input_var]
        if len(input_var_data) == 1:
            node[input_var] = input_var_data[0]
        else:            
            node_name = input_var_data[0]
            if node_name in nodes:
                node[input_var] = nodes[node_name][input_var_data[1]]
            else:
                node[input_var] = None
    l_vars = [name.strip() for name in d_mod['vars'].split(',')] if 'vars' in d_mod else []
    for var in l_vars:
        node[var] = d_mod[var] if var in d_mod else None
    if 'eval_vars' in d_mod:
        l_eval_vars = [name.strip() for name in \
                d_mod['eval_vars'].split(',')] if 'vars' in d_mod else []
        for eval_var in l_eval_vars:
            
            node[eval_var] = eval(d_mod[eval_var]) if eval_var in d_mod else None
    if 'eval' in d_mod or 'await_eval' in d_mod:
        if 'eval' in d_mod:
            exec_results = eval(d_mod['eval'])
        else:
            exec_results = await eval(d_mod['await_eval'])
        l_output_vars = [name.strip() for name in d_mod['outputs'].split(',')]
        for ivar, output_var in enumerate(l_output_vars):
            node[output_var] = exec_results[ivar]
    elif 'exec' in d_mod:
        l_output_vars = [name.strip() for name in d_mod['outputs'].split(',')]
        for ivar, output_var in enumerate(l_output_vars):
            node[output_var] = None
        exec(d_mod['exec'], globals(), node)
    success_var_name = d_mod['success_var'].strip()
    node['next_node'] = node['flow']['next_node_options'][node[success_var_name]]
    return node

def main():
    # test_add()
    asyncio.run(exec_flow(verified_chat)) # test_add # 
    pass

if __name__ == '__main__':
    main()
    print('done')

