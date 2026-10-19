import re, difflib
PATH='src/football_intelligence/features/engineering.py'
lines=open(PATH,encoding='utf-8').read().split('\n')
def find(name): return [i for i,l in enumerate(lines) if l.startswith('def '+name+'(')]
r=find('_add_rest_features'); h=find('_add_h2h_features')
print('rest defs @', [i+1 for i in r], 'h2h defs @', [i+1 for i in h])
assert len(r)==2 and len(h)==2, 'unexpected def count'
def body(start, nextstart): return lines[start:nextstart]
rest1=body(r[0], h[0]); rest2=body(r[1], h[1])
h2h1=body(h[0], r[1]); h2h2=body(h[1], len(lines))
def norm(x): return [l.rstrip() for l in x]
dr=list(difflib.unified_diff(norm(rest1),norm(rest2),'rest_first','rest_active',lineterm=''))
dh=list(difflib.unified_diff(norm(h2h1),norm(h2h2),'h2h_first','h2h_active',lineterm=''))
print('rest bodies differ in non-blank lines:', any(ln for ln in dr if not ln.startswith('@@')))
print('rest diff:'); 
for x in dr: print(x)
print('h2h bodies differ:', any(ln for ln in dh if not ln.startswith('@@')))
for x in dh: print(x)
print('DEAD BLOCK (first copies) 1-based range:', r[0]+1, '..', r[1], '(i.e. from first _add_rest up to line before active _add_rest)')
print('dead block first 2 lines:', lines[r[0]:r[0]+2])
print('dead block last 2 lines:', lines[r[1]-2:r[1]])
