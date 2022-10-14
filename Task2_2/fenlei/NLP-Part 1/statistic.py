import re

with open('dataset/train.json', 'r') as f:
    all_content = f.read()

find_program = '"program": "(.*?)",'
programs = re.findall(find_program, all_content)
programs = [', '+program for program in programs]#[:100]
#print(programs[:20])

ops_list = [re.findall(', (.*?)[(].*?[)]', program) for program in programs]
#print(ops_list)

concat_op_list = []
for ops in ops_list:
    concat_op = ''
    i = 0
    for op in ops:
        concat_op += op + str(i) + '_'
        i += 1
    concat_op = concat_op.rstrip('_')
    concat_op_list.append(concat_op)
print(concat_op_list)
