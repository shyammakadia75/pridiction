import sys
path=sys.argv[1]; start=int(sys.argv[2]); end=int(sys.argv[3])
lines=open(path,encoding='utf-8').read().split('\n')
for i in range(start-1, min(end, len(lines))):
    print(f'{i+1:04d}|{lines[i]}')
