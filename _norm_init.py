import sys
sys.path.insert(0, r"src")
from football_intelligence.cli import build_parser
parser = build_parser()
a = parser.parse_args(["ratings", "poisson", "--db-path", "x.db"])
print("func:", a.func.__name__)
print("competition_id:", a.competition_id)
print("max_goals:", a.max_goals)
print("min_history:", a.min_history)
print("prior_strength:", a.prior_strength)
print("output_dir default contains phase20:", "phase20" in a.output_dir)
# default subparser still resolves elo
a2 = parser.parse_args(["ratings", "elo", "--db-path", "x.db"])
print("elo func:", a2.func.__name__)





