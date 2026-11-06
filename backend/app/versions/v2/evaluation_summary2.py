# Print the full report.  python -m app.versions.v2.evaluation_summary2
from app.versions.toolkit.experiment import summary

if __name__ == "__main__":
    summary("v2")
