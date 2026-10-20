from ultralytics import YOLO
import time
import gc
import torch

batches = [2, 4, 6, 8]

for batch in batches:
    for i in range(1,6):
        model = YOLO("yolo11n.pt")

        model.train(
            data=f"fold_{i}.yaml",
            epochs=20,
            batch=batch,
            imgsz=640,
            
            optimizer='SGD',
            lr0=0.005,            
            momentum=0.9,         
            weight_decay=0.0005,
            
            workers=4,    
            cache=True,
            
            project="WVC2026_detection",
            name=f"bs{batch}_fold{i}",
        )
        
        del model
        
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        if i < 5:
            time.sleep(60)