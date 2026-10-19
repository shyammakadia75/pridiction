import re, shutil
PATH='src/football_intelligence/features/engineering.py'
BAK=PATH+'.bak'
shutil.copy2(PATH, BAK)
lines=open(PATH,encoding='utf-8').read().split('\n')
def idx(name): return [i for i,l in enumerate(lines) if l.startswith('def '+name+'(')]
r=idx('_add_rest_features'); h=idx('_add_h2h_features')
assert len(r)==2 and len(h)==2, ('def counts', r, h)
first_rest=r[0]; second_rest=r[1]
dead=lines[first_rest:second_rest]
new=lines[:first_rest]+lines[second_rest:]
newtext='\n'.join(new)
# preservation assertions (competition/season filter untouched)
for key in ['load_matches_for_features','_resolve_competition_id','_ensure_competition_exists','compute_all_features_from_db','def _filter_history','_add_venue_features']:
    assert key in newtext, 'LOST '+key
assert re.search(r'\bload_matches_for_features', newtext) is not None
# exactly one of each duplicated def remains
assert len(re.findall(r'^def _add_rest_features\(', newtext, re.M))==1, 'rest not deduped'
assert len(re.findall(r'^def _add_h2h_features\(', newtext, re.M))==1, 'h2h not deduped'
open(PATH,'w',encoding='utf-8').write(newtext)
print('17.3 removal: before=%d after=%d deleted_lines=%d'%(len(lines),len(new),len(dead)))
print('dead block 1-based range:', first_rest+1, '..', second_rest)
print('backup:', BAK)
