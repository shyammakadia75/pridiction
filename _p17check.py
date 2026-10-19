import difflib, subprocess
PATH = 'src/football_intelligence/features/engineering.py'
l = open(PATH, encoding='utf-8').read().splitlines()
def b(s, e): return [x.rstrip() for x in l[s-1:e]]
d1 = list(difflib.unified_diff(b(568,590), b(639,660), 'rest_first', 'rest_active', lineterm=''))
d2 = list(difflib.unified_diff(b(592,636), b(663,707), 'h2h_first', 'h2h_active', lineterm=''))
print('=== BODY EQUIVALENCE (17.3) ===')
print('lines total:', len(l))
print('rest bodies ->', 'IDENTICAL' if not d1 else 'DIFFERENT')
for x in d1[:25]: print(x)
print('h2h bodies ->', 'IDENTICAL' if not d2 else 'DIFFERENT')
for x in d2[:25]: print(x)
print('=== GIT: competition filter committed vs working-tree (preserve rule) ===')
diff = subprocess.run(['git','--no-pager','diff','HEAD','--',PATH], capture_output=True, text=True).stdout
for key in ['load_matches_for_features','_resolve_competition_id','_ensure_competition_exists','compute_all_features_from_db']:
    print(f'  {key}: working-tree-vs-HEAD occurrences = {diff.count(key)}')
head_lines = subprocess.run(['git','--no-pager','show','HEAD:'+PATH], capture_output=True, text=True).stdout.splitlines()
print('HEAD engineering.py lines:', len(head_lines), '| WORK lines:', len(l))
print('HEAD has duplicate _add_rest?', head_lines_str_count := subprocess.run(['git','--no-pager','show','HEAD:'+PATH], capture_output=True, text=True).stdout.count('def _add_rest_features('), '| HEAD h2h dup count:', subprocess.run(['git','--no-pager','show','HEAD:'+PATH], capture_output=True, text=True).stdout.count('def _add_h2h_features('))
