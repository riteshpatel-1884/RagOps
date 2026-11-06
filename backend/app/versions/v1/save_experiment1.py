# Write experiment_v1.json.  python -m app.versions.v1.save_experiment1
from app.versions.toolkit.experiment import save

if __name__ == "__main__":
    save("v1")
