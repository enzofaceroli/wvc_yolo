from pathlib import Path
import pandas as pd
from ultralytics import YOLO

def main():
    src_dir = Path("WVC2026_detection")
    results = []
    
    for experiment in src_dir.iterdir():
        exp_name = experiment.name.split("_fold")[0]
        fold = experiment.name.split("_fold")[1]
        
        model_path = experiment / "weights" / "last.pt"
        model = YOLO(model_path)
        
        metrics = model.val()
        
        precision = metrics.box.mp
        recall = metrics.box.mr
        map50 = metrics.box.map50
        map50_95 = metrics.box.map
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

        results.append({
            "exp_name": exp_name,
            "fold": fold,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "mAP50": map50,
            "mAP50-95": map50_95
        })
        
        pd.DataFrame(results).to_csv("detection_metrics.csv", index=False)

if __name__ == "__main__":
    main()
    
    