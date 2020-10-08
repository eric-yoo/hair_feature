import random, os
def train_test_split():
    with open ('file_names.txt') as f:
        for i,line in enumerate(f.readlines()):
            if i%2 == 0:
                line = line.split("_v0")[0] + '\n'
                with open ('list.txt', 'a') as wf:
                    wf.write(line)
                if random.random() <= 0.7:
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
    check_data()
    # print(os.listdir('..'))