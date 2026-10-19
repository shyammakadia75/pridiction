import ast, pathlib, sys
src = pathlib.Path("src/football_intelligence/data/ingestion/football_data_results.py")
ast.parse(src.read_text(encoding="utf-8"))
print("AST OK")
sys.path.insert(0, str(pathlib.Path("src").resolve()))
from football_intelligence.data.ingestion.football_data_results import (
    FootballDataResultsIngestor, DATASET_NAME, list_result_leagues,
    validate_outcome_score_consistency, _normalize_results_dataframe,
    RESULT_CSV_COLUMNS, _compose_provider_match_id,
)
print("IMPORT OK", DATASET_NAME, list_result_leagues()[:2])
print("RESULT_CSV_COLUMNS", RESULT_CSV_COLUMNS)
# quick logic smoke
print("season 1617 ->", FootballDataResultsIngestor.__module__)
