from pathlib import Path
from datetime import datetime


def save_quality_issues(issues: list[str], label: str) -> Path:
    """
    Salva as mensagens de problema de qualidade em um arquivo de log por dataset.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"quality_{label}_{timestamp}.log"
    output_path = Path("monitoring") / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for issue in issues:
            f.write(issue.strip() + "\n")

    return output_path
