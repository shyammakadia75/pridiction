"""Fix indentation issue in cli.py."""
p = 'src/football_intelligence/cli.py'
content = open(p, 'r', encoding='utf-8').read()

content = content.replace(
    '        print(f"Status: {result.status}")\n',
    '    print(f"Status: {result.status}")\n',
)

open(p, 'w', encoding='utf-8', newline='').write(content)
print("Done")
