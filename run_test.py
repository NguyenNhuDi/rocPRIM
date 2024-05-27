from sys import argv
import os
from tqdm import tqdm

def main():
    assert len(argv), 'ussage: python3 run_test.py test_name N'

    test_name = argv[1]
    N = int(argv[2]) 

    os.chdir('/home/zenguyen/forks/rocPRIM/build/release')
    for i in tqdm(range(1, N+1)):
        os.system(f'ctest -R {test_name} > t.txt')

        f = open('t.txt', 'r')

        num_lines = sum(1 for _ in f)
        if num_lines != 7:
            print(f'FAILED ON TEST {i}')
            break

main()