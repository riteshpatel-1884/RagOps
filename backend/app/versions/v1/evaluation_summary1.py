# Print the full report.  python -m app.versions.v1.evaluation_summary1
from app.versions.toolkit.experiment import summary

if __name__ == "__main__":
    summary("v1")
