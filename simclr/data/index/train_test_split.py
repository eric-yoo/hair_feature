import random
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

if __name__ == "__main__":
    train_test_split()