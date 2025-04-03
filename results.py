import matplotlib.pyplot as plt


class ResultVisualizer:
    """Handles prediction visualization and analysis"""

    @staticmethod
    def plot_training_progress(loss_history):
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')

    @staticmethod
    def plot_predictions(actual, predicted):
        plt.figure(figsize=(10, 6))
        plt.scatter(actual, predicted, alpha=0.6)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('Normalized Actual Values')
        plt.ylabel('Normalized Predictions')