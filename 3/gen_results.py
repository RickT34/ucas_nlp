from pathlib import Path
import json

DIR_OUTPUT = Path('output')

def get_summery():
    csv = open('summery.csv', 'w', encoding='utf-8', newline='\n')

    FILEDS = ['model_name', 'accuracy','precision', 'recall', 'f1', 'matched']

    csv.write(', '.join(FILEDS)+'\n')

    for file in DIR_OUTPUT.glob('*.json'):
        data = json.load(file.open())
        csv.write(', '.join([file.stem[7:-9]]+[str(data['summery'][f]) for f in FILEDS[1:]])+'\n')
        
    csv.close()

def do_compare(output, *files:Path):
    datas = tuple(json.load(f.open()) for f in files)
    assert len(datas) > 0
    samples = tuple(d['raw'] for d in datas)
    length = len(samples[0])
    matched_index = [i for i in range(length) if all(s[i]['result']['matched'] for s in samples)]
    print(f'Matched {len(matched_index)} out of {length} samples.')
    csv = open(output, 'w', encoding='utf-8', newline='\n')
    FILEDS = ['model_name', 'accuracy', 'precision','recall', 'f1']
    csv.write(', '.join(FILEDS)+'\n')
    for j, d in enumerate(samples):
        csv.write(files[j].stem[7:-9] + ', ' + ', '.join(
            str(sum(d[i]['result'][f] for i in matched_index)/len(matched_index)) for f in FILEDS[1:]
            ) + '\n')

do_compare("compare_size.csv",
        Path('output/result_qwen2.5-14b.yaml_100.json'),
        Path('output/result_qwen2.5-32b.yaml_100.json'),
        Path('output/result_qwen2.5-72b.yaml_100.json'))

do_compare("compare_prompt.csv",
        Path('output/result_qwen2.5-72b.yaml_100.json'),
        Path('output/result_qwen2.5-72b+example.yaml_100.json'),
        Path('output/result_qwen2.5-72b+cot.yaml_100.json'),
        Path('output/result_qwen2.5-72b+example+cot.yaml_100.json'))

get_summery()