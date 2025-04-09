import os
import random

INPUT = 'data/format_each_allDatas.txt'
OUTPUT = 'data/shuffledDataList.txt'

def main():
    res = []
    with open(INPUT) as f:
        for line in f:
            img = line.split(',')[0]
            res.append(img)
    
    random.shuffle(res)
    with open(OUTPUT, 'a') as f:
        for img in res:
            f.write(img)
            f.write('\n')

if __name__ == "__main__":
    main()