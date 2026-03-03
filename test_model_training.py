"""
Test Script - Train models on ETTh1 dataset
"""

from pathlib import Path
from models.trainer import ModelTrainer

def main():
    # Set paths
    pipeline_dir = Path(__file__).parent
    outputs_dir = pipeline_dir / "outputs"
    
    print("="*70)
    print("MODEL TRAINING TEST")
    print("="*70)
    
    # Initialize trainer
    trainer = ModelTrainer(outputs_dir)
    
    # Train all models (with hyperparameter tuning)
    results = trainer.train_all_recommended_models(
        dataset_name="ETTh1",
        tune_hyperparameters=True,  # Enable auto-tuning
        n_trials=20  # Number of trials per model
    )
    
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    
    for model_name, result in results.items():
        print(f"\n{model_name}:")
        if result['status'] == 'success':
            print(f"  ✓ Success")
            print(f"  RMSE: {result['evaluation']['rmse']:.4f}")
            print(f"  R²: {result['evaluation']['r2']:.4f}")
            print(f"  Model saved: {result['model_path']}")
        else:
            print(f"  ✗ Failed: {result.get('error', 'Unknown error')}")
    
    print("\n" + "="*70)
    print("Test complete!")
    print("="*70)

if __name__ == "__main__":
    main()
