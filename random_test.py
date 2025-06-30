import csv
import random
from datetime import datetime, timedelta

main_categories = [
    "Physics", "Anthropology", "Biology", "Languages", "Data Science",
    "History", "Computer Science", "Film Studies", "Music Theory",
    "Engineering", "Creative Writing"
]

def random_date(start, end):
    """Generate a random datetime between `start` and `end`"""
    return start + timedelta(
        seconds=random.randint(0, int((end - start).total_seconds()))
    )

# Define the output CSV file
filename = "knowledge_mock_data.csv"

# Time range for random timestamps
start_time = datetime(2025, 4, 1)
end_time = datetime(2025, 6, 30)

# Number of rows to generate
row_count = 500

with open(filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    
    # Write header (columns)
    writer.writerow([
        "category", "score", "total_score", "created_at", "updated_at", "is_completed"
    ])
    
    for _ in range(row_count):
        category = random.choice(main_categories)
        score = random.randint(0, 20)
        total_score = random.randint(score, score + 10)
        created_at = random_date(start_time, end_time).isoformat()
        updated_at = random_date(datetime.fromisoformat(created_at), end_time).isoformat()
        is_completed = True
        writer.writerow([category, score, total_score, created_at, updated_at, is_completed])

print(f"✅ CSV file '{filename}' generated with {row_count} rows.")