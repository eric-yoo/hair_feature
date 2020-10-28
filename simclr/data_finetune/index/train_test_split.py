import random, os
def train_test_split():
    with open ('file_names.txt') as f:
        for i,line in enumerate(f.readlines()):
            if i%2 == 1:
                continue
            with open ('list.txt', 'a') as wf:
                wf.write(line)
            if not (line.strip().endswith('v8.png')):
                with open ('train.txt', 'a') as wf:
                    wf.write(line)
            else:
                with open ('test.txt', 'a') as wf:
                    wf.write(line)

def check_data():
    datafiles = os.listdir('..')
    with open ('list.txt') as f:
        for i,line in enumerate(f.readlines()):
            line = line.strip()
            if line+"_v0.png" not in datafiles:
                print(line)
            if line+"_v1.png" not in datafiles:
                print(line)
            

if __name__ == "__main__":
    train_test_split()
    for fn in ['file_names', 'list', 'train', 'test']:
        with open (fn + '.txt') as f:
            lines = sorted(f.readlines())
            with open (fn + '.txt', 'w') as wf:
                wf.writelines(lines)