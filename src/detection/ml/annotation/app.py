from pathlib import Path
import sys


# ``streamlit run path/to/app.py`` adds the script directory, not the repository
# root, so keep project-level imports available for the UI module.
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.ml.annotation.ui import main


if __name__ == "__main__":
    main()
