
def mod_test_add(sn1, sn2):
    try:
        n1 = int(sn1)
        n2 = int(sn2)
    except ValueError:
        return False, 'Error'
    
    return True, str(n1 + n2)
    pass

def input_digits(prompt):
    inp = input(prompt).strip()
    if not inp.isdigit():
        print('Sorry. The input must consist entirely of digits.')
        return False, None
    return True, inp

def mod_input():
    pass

def mod_print(prompt, inp):
    print(prompt.format(inp=inp))

def parse_mod_file(file_name):
    sections = {}
    current_section = None
    with open('modules/'+file_name, 'rt') as file:
        for line in file:
            line = line.strip()
            if line.startswith('>>'):
                current_section = line[2:]
                sections[current_section] = ' '
            elif current_section is not None:
                sections[current_section] += line + ' '
    return sections

def test_add():
    flow = {}
    flow['start_node'] = 'inp1'
    flow['inp1'] = {}
    flow['inp1']['mod'] = 'input_int_mod.txt'
    flow['inp1']['prompt'] = {}
    flow['inp1']['prompt']['constant'] = True
    flow['inp1']['prompt']['val'] = 'Please enter the first number.'
    flow['inp1']['next_node_on_fail'] = 'inp1'
    flow['inp1']['next_node_on_success'] = 'inp2'
    flow['inp2'] = {}
    flow['inp2']['mod'] = 'input_int_mod.txt'
    flow['inp2']['prompt'] = {}
    flow['inp2']['prompt']['constant'] = True
    flow['inp2']['prompt']['val'] = 'Please enter the second number.'
    flow['inp2']['next_node_on_fail'] = 'inp2'
    flow['inp2']['next_node_on_success'] = 'add1'
    flow['add1'] = {}
    flow['add1']['mod'] = 'test_add_mod.txt'
    flow['add1']['i1'] = {}
    flow['add1']['i1']['node'] = 'inp1'
    flow['add1']['i1']['var'] = 'output'
    flow['add1']['i2'] = {}
    flow['add1']['i2']['node'] = 'inp2'
    flow['add1']['i2']['var'] = 'output'
    flow['add1']['next_node_whenever'] = 'print1'
    flow['print1'] = {}
    flow['print1']['mod'] = 'print_mod.txt'
    flow['print1']['inp'] = {}
    flow['print1']['inp']['node'] = 'add1'
    flow['print1']['inp']['var'] = 'sum'
    flow['print1']['msg'] = {}
    flow['print1']['msg'] = {}
    flow['print1']['msg']['constant'] = True
    flow['print1']['msg']['val'] = "Your result is {inp}."
    flow['print1']['next_node_whenever'] = 'inp1'

    return flow
    
def exec_flow(conf_nodes):
    flow = conf_nodes()
    nodes = {}
    curr_node_name = flow['start_node']
    while True:
        curr_node = {}
        nodes[curr_node_name] = curr_node
        curr_node['flow'] = flow[curr_node_name]
        curr_node = eval_module(flow, nodes, curr_node)
        if 'next_node' not in curr_node:
            print('flow terminated due to no new requested node.')
            break
        if curr_node['next_node'] not in flow:
            print(f"Unknown node {curr_node['next_node']} requested as next node")
        else:
            curr_node_name = curr_node['next_node']
            

def eval_module(flow, nodes, node : dict = None):
    fname = node['flow']['mod']
    d_mod = parse_mod_file(fname)
    l_input_vars = [name.strip() for name in d_mod['inputs'].split(',')]
    for ivar, input_var in enumerate(l_input_vars):
        if 'constant' in node['flow'][input_var] and node['flow'][input_var]['constant']:
            node[input_var] = node['flow'][input_var]['val']
        else:
            node[input_var] = nodes[node['flow'][input_var]['node']][node['flow'][input_var]['var']]
    exec_results = eval(d_mod['exec'])
    l_output_vars = [name.strip() for name in d_mod['outputs'].split(',')]
    for ivar, output_var in enumerate(l_output_vars):
        node[output_var] = exec_results[ivar]
    success_var_name = d_mod['success_var'].strip()
    if 'next_node_whenever' in node['flow']:
        node['next_node'] = node['flow']['next_node_whenever']
    else:
        if node[success_var_name]:
            node['next_node'] = node['flow']['next_node_on_success']
        else:
            node['next_node'] = node['flow']['next_node_on_fail']
    return node

def main():
    # test_add()
    exec_flow(test_add)
    pass

if __name__ == '__main__':
    main()
    print('done')

