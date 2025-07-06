from pathlib import Path
from ultralytics import YOLO

if __name__ == "__main__":
    FINAL_DATA_DIR = Path("./components_data")

    model_general = YOLO("./yolo11n_best_small_component.pt")

    # Train the model
    results = model_general.train(
        data=FINAL_DATA_DIR / "data.yaml",
        device=0,
        batch=-1,
        save_period=3,
        save=True,
        cache="disk",
        amp=True,
        plots=True,
        freeze=9,
        patience=9,
        epochs=75,
        imgsz=640,
        hsv_h=0.5,
        hsv_s=0.7,
        hsv_v=0.6,
        degrees=90,
        translate=0.1,
        scale=0.35,
        shear=5,
        perspective=0.001,
        flipud=0.5,
        fliplr=0.5,
        mosaic=0.5,
        close_mosaic=25,
        erasing=0.0,
        bgr=0.0,
        mixup=0.0,
        cutmix=0.0,
        auto_augment=0.0,
        dropout=0.5,
    )
