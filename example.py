from core.data_preprocessor import ScourDataProcessor
from core.model import PhysicsInformedTransformer
from experiments.train import ScourTrainer

# Initialize with sample configuration
config = {
    'data_path': 'sample_data.csv',
    'input_dim': 5,
    'hidden_dim': 32,
    'epochs': 300
}

processor = ScourDataProcessor(config)
model = PhysicsInformedTransformer(config)
trainer = ScourTrainer(model, processor, config)

# Execute training workflow
normalized_data = processor.load_dataset()
train_loader, test_loader = trainer.prepare_data(normalized_data)
trainer.train_model(train_loader)