import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from typing import Tuple, Dict, List, Optional, Any, Union
import logging
import time
from contextlib import contextmanager

from src.config import (
    MODELS_DIR, RESULTS_DIR, EMOTION_LABELS, NUM_CLASSES,
    FER_MODEL_PATH, VER_MODEL_PATH, FUSION_MODEL_PATH,
    LOG_LEVEL, LOG_FILE
)
from src.data_processors import load_fer_data, load_ver_data, check_directory_structure
from src.utils import plot_confusion_matrix, setup_gpu_memory

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@contextmanager
def timer(operation_name: str) -> None:
    """Context manager for timing operations"""
    start_time = time.time()
    yield
    elapsed_time = time.time() - start_time
    logger.info(f"{operation_name} completed in {elapsed_time:.2f} seconds")

class ModelEvaluator:
    """Class to evaluate and compare emotion recognition models"""
    
    def __init__(self, save_dir: str = RESULTS_DIR):
        """Initialize the evaluator with paths and create directories"""
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        logger.info(f"Results will be saved to {self.save_dir}")
        
        # Model paths
        self.fer_model_path = FER_MODEL_PATH
        self.ver_model_path = VER_MODEL_PATH
        self.fusion_model_path = FUSION_MODEL_PATH
        
        # Class labels
        self.emotion_labels = EMOTION_LABELS
        self.num_classes = NUM_CLASSES
        
        # Models and data (to be loaded)
        self.fer_model = None
        self.ver_model = None
        self.fusion_model = None
        self.fer_embedding_model = None
        self.ver_embedding_model = None
    
    def _load_models(self) -> bool:
        """Load all models and create embedding models"""
        try:
            # Check if all model files exist
            if not all(os.path.exists(path) for path in 
                      [self.fer_model_path, self.ver_model_path, self.fusion_model_path]):
                missing = []
                if not os.path.exists(self.fer_model_path): missing.append("FER")
                if not os.path.exists(self.ver_model_path): missing.append("VER")
                if not os.path.exists(self.fusion_model_path): missing.append("Fusion")
                logger.error(f"Missing model files: {', '.join(missing)}")
                return False
            
            # Load models
            with timer("Loading FER model"):
                self.fer_model = tf.keras.models.load_model(self.fer_model_path)
                self.fer_embedding_model = tf.keras.Model(
                    inputs=self.fer_model.input,
                    outputs=self.fer_model.get_layer('fer_embedding').output
                )
            
            with timer("Loading VER model"):
                self.ver_model = tf.keras.models.load_model(self.ver_model_path)
                self.ver_embedding_model = tf.keras.Model(
                    inputs=self.ver_model.input,
                    outputs=self.ver_model.get_layer('ver_embedding').output
                )
            
            with timer("Loading Fusion model"):
                self.fusion_model = tf.keras.models.load_model(self.fusion_model_path)
            
            logger.info("All models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False

    def _evaluate_model(self, 
                       model: tf.keras.Model, 
                       test_data: Union[tf.keras.utils.Sequence, Tuple], 
                       model_name: str) -> Tuple[Optional[float], Optional[Dict]]:
        """Generic model evaluation function"""
        try:
            logger.info(f"Evaluating {model_name} model...")
            
            # Unpack test data if it's a tuple (for fusion model)
            is_fusion = model_name == "Fusion"
            
            with timer(f"{model_name} evaluation"):
                # Evaluate model
                eval_results = model.evaluate(test_data, verbose=1)
                accuracy = eval_results[1]
                logger.info(f"{model_name} Model Test Accuracy: {accuracy:.4f}")
                
                # Get predictions
                y_pred_prob = model.predict(test_data, verbose=1)
                y_pred = np.argmax(y_pred_prob, axis=1)
                
                # Get true labels
                if is_fusion:
                    y_true = np.argmax(test_data[1], axis=1)
                else:
                    test_labels = np.vstack([y for _, y in test_data])[:len(y_pred)]
                    y_true = np.argmax(test_labels, axis=1)
                
                # Generate confusion matrix
                cm_path = os.path.join(self.save_dir, f"{model_name.lower()}_confusion_matrix.png")
                plot_confusion_matrix(
                    y_true, y_pred, self.emotion_labels,
                    title=f"{model_name} Model Confusion Matrix",
                    save_path=cm_path, normalize=True
                )
                
                # Generate classification report
                report = classification_report(
                    y_true, y_pred,
                    target_names=[self.emotion_labels[i] for i in range(self.num_classes)],
                    output_dict=True
                )
                
                # Print classification report
                logger.info(f"{model_name} Classification Report:")
                print(classification_report(
                    y_true, y_pred,
                    target_names=[self.emotion_labels[i] for i in range(self.num_classes)]
                ))
                
                # Calculate emotion-wise accuracies
                emotion_accuracy = {}
                for i in range(self.num_classes):
                    emotion = self.emotion_labels[i]
                    mask = (y_true == i)
                    if np.sum(mask) > 0:
                        emotion_accuracy[emotion] = np.mean(y_pred[mask] == i)
                    else:
                        emotion_accuracy[emotion] = 0.0
                
                # Plot emotion-wise accuracies
                self._plot_emotion_accuracy(emotion_accuracy, f"{model_name.lower()}_emotion_accuracy.png")
                
                return accuracy, report
                
        except Exception as e:
            logger.error(f"Error evaluating {model_name} model: {str(e)}")
            return None, None
    
    def _plot_emotion_accuracy(self, emotion_accuracy: Dict[str, float], filename: str) -> None:
        """Plot accuracy for each emotion"""
        plt.figure(figsize=(10, 6))
        emotions = list(emotion_accuracy.keys())
        accuracies = list(emotion_accuracy.values())
        
        # Create bar plot
        bars = plt.bar(emotions, accuracies, color=sns.color_palette("viridis", len(emotions)))
        
        # Add text labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.2f}',
                    ha='center', va='bottom', rotation=0)
        
        plt.ylim(0, 1.1)
        plt.xlabel('Emotion')
        plt.ylabel('Accuracy')
        plt.title('Emotion-wise Accuracy')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.save_dir, filename))
        plt.close()
    
    def evaluate_fer_model(self) -> Tuple[Optional[float], Optional[Dict]]:
        """Evaluate the Facial Emotion Recognition model"""
        if self.fer_model is None and not self._load_models():
            return None, None
        
        # Load test data
        _, test_generator = load_fer_data()
        
        return self._evaluate_model(self.fer_model, test_generator, "FER")
    
    def evaluate_ver_model(self) -> Tuple[Optional[float], Optional[Dict]]:
        """Evaluate the Vocal Emotion Recognition model"""
        if self.ver_model is None and not self._load_models():
            return None, None
        
        # Load test data
        _, test_generator, _, _ = load_ver_data()
        
        return self._evaluate_model(self.ver_model, test_generator, "VER")
    
    def prepare_fusion_test_data(self) -> Tuple[List[np.ndarray], np.ndarray]:
        """Prepare test data for fusion model evaluation"""
        logger.info("Preparing fusion test data...")
        
        # Load test data generators
        _, fer_test_gen = load_fer_data()
        _, ver_test_gen, _, _ = load_ver_data()
        
        fer_batches = []
        ver_batches = []
        label_batches = []
        
        # Get smaller test set size
        min_batches = min(len(fer_test_gen), len(ver_test_gen))
        
        with timer("Generating embeddings for fusion evaluation"):
            for i in range(min_batches):
                # Get batch from generators
                fer_images, fer_labels = fer_test_gen[i]
                ver_specs, ver_labels = ver_test_gen[i]
                
                # Generate embeddings
                fer_emb = self.fer_embedding_model.predict(fer_images, verbose=0)
                ver_emb = self.ver_embedding_model.predict(ver_specs, verbose=0)
                
                # Store data
                fer_batches.append(fer_emb)
                ver_batches.append(ver_emb)
                label_batches.append(fer_labels)  # Using FER labels
        
        # Concatenate batches
        fer_embeddings = np.vstack(fer_batches)
        ver_embeddings = np.vstack(ver_batches)
        labels = np.vstack(label_batches)
        
        # Trim to same size if needed
        min_len = min(len(fer_embeddings), len(ver_embeddings), len(labels))
        fer_embeddings = fer_embeddings[:min_len]
        ver_embeddings = ver_embeddings[:min_len]
        labels = labels[:min_len]
        
        logger.info(f"Fusion test data prepared: {min_len} samples")
        return [fer_embeddings, ver_embeddings], labels
    
    def evaluate_fusion_model(self) -> Tuple[Optional[float], Optional[Dict]]:
        """Evaluate the multi-modal fusion model"""
        if self.fusion_model is None and not self._load_models():
            return None, None
            
        # Prepare fusion test data
        test_inputs, test_labels = self.prepare_fusion_test_data()
        
        return self._evaluate_model(self.fusion_model, (test_inputs, test_labels), "Fusion")
    
    def compare_models(self) -> Optional[Dict]:
        """Compare the performance of all models"""
        logger.info("Starting model comparison...")
        
        # Load all models first
        if not self._load_models():
            logger.error("Failed to load models. Model comparison aborted.")
            return None
        
        # Evaluate all models
        with timer("FER model evaluation"):
            fer_accuracy, fer_report = self.evaluate_fer_model()
        
        with timer("VER model evaluation"):
            ver_accuracy, ver_report = self.evaluate_ver_model()
        
        with timer("Fusion model evaluation"):
            fusion_accuracy, fusion_report = self.evaluate_fusion_model()
        
        # Check if all evaluations succeeded
        if None in [fer_accuracy, ver_accuracy, fusion_accuracy]:
            logger.error("One or more model evaluations failed.")
            return None
        
        # Get metrics for comparison
        metrics = ['precision', 'recall', 'f1-score']
        
        for metric in metrics:
            # Get class-wise metrics
            fer_metric = [fer_report[self.emotion_labels[i]][metric] for i in range(self.num_classes)]
            ver_metric = [ver_report[self.emotion_labels[i]][metric] for i in range(self.num_classes)]
            fusion_metric = [fusion_report[self.emotion_labels[i]][metric] for i in range(self.num_classes)]
            
            # Plot comparison
            plt.figure(figsize=(12, 8))
            x = np.arange(self.num_classes)
            width = 0.25
            
            plt.bar(x - width, fer_metric, width, label='FER Model')
            plt.bar(x, ver_metric, width, label='VER Model')
            plt.bar(x + width, fusion_metric, width, label='Fusion Model')
            
            plt.xlabel('Emotion')
            plt.ylabel(metric.capitalize())
            plt.title(f'Model Performance Comparison by Emotion ({metric})')
            plt.xticks(x, [self.emotion_labels[i] for i in range(self.num_classes)], rotation=45)
            plt.legend()
            plt.tight_layout()
            
            # Save plot
            plt.savefig(os.path.join(self.save_dir, f'model_comparison_{metric}.png'))
            plt.close()
        
        # Plot overall accuracy comparison
        self._plot_accuracy_comparison(fer_accuracy, ver_accuracy, fusion_accuracy)
        
        # Print overall performance comparison
        logger.info("\nOverall Model Performance Comparison:")
        logger.info(f"FER Model Accuracy: {fer_accuracy:.4f}")
        logger.info(f"VER Model Accuracy: {ver_accuracy:.4f}")
        logger.info(f"Fusion Model Accuracy: {fusion_accuracy:.4f}")
        logger.info(f"Improvement of Fusion over FER: {(fusion_accuracy - fer_accuracy) * 100:.2f}%")
        logger.info(f"Improvement of Fusion over VER: {(fusion_accuracy - ver_accuracy) * 100:.2f}%")
        
        # Return comparison results
        return {
            'fer': {'accuracy': fer_accuracy, 'report': fer_report},
            'ver': {'accuracy': ver_accuracy, 'report': ver_report},
            'fusion': {'accuracy': fusion_accuracy, 'report': fusion_report}
        }
    
    def _plot_accuracy_comparison(self, fer_acc, ver_acc, fusion_acc):
        """Plot overall accuracy comparison of models"""
        plt.figure(figsize=(10, 6))
        models = ['FER', 'VER', 'Fusion']
        accuracies = [fer_acc, ver_acc, fusion_acc]
        
        # Create bar plot with custom colors
        bars = plt.bar(models, accuracies, color=['#3498db', '#2ecc71', '#9b59b6'])
        
        # Add text labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom')
        
        plt.ylim(0, 1.1)
        plt.xlabel('Model')
        plt.ylabel('Accuracy')
        plt.title('Overall Model Accuracy Comparison')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.save_dir, 'overall_accuracy_comparison.png'))
        plt.close()
    
    def generate_summary_report(self, comparison_results):
        """Generate a comprehensive summary report of all model performances"""
        if comparison_results is None:
            logger.error("Cannot generate summary report: no comparison results available")
            return
        
        # Create report file path
        report_path = os.path.join(self.save_dir, 'evaluation_summary.txt')
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("EMOTION RECOGNITION MODEL EVALUATION SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            # Overall accuracy comparison
            f.write("OVERALL ACCURACY\n")
            f.write("-" * 80 + "\n")
            f.write(f"FER Model:    {comparison_results['fer']['accuracy']:.4f}\n")
            f.write(f"VER Model:    {comparison_results['ver']['accuracy']:.4f}\n")
            f.write(f"Fusion Model: {comparison_results['fusion']['accuracy']:.4f}\n\n")
            
            # Improvement percentages
            fer_improvement = (comparison_results['fusion']['accuracy'] - comparison_results['fer']['accuracy']) * 100
            ver_improvement = (comparison_results['fusion']['accuracy'] - comparison_results['ver']['accuracy']) * 100
            f.write(f"Fusion improves over FER by: {fer_improvement:.2f}%\n")
            f.write(f"Fusion improves over VER by: {ver_improvement:.2f}%\n\n")
            
            # Per-emotion performance
            f.write("PER-EMOTION PERFORMANCE (F1-SCORE)\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Emotion':<12} {'FER':<10} {'VER':<10} {'Fusion':<10} {'Best Model':<12}\n")
            
            for i in range(self.num_classes):
                emotion = self.emotion_labels[i]
                fer_f1 = comparison_results['fer']['report'][emotion]['f1-score']
                ver_f1 = comparison_results['ver']['report'][emotion]['f1-score']
                fusion_f1 = comparison_results['fusion']['report'][emotion]['f1-score']
                
                # Determine best model for this emotion
                best_score = max(fer_f1, ver_f1, fusion_f1)
                if best_score == fer_f1:
                    best_model = "FER"
                elif best_score == ver_f1:
                    best_model = "VER"
                else:
                    best_model = "Fusion"
                
                f.write(f"{emotion:<12} {fer_f1:<10.4f} {ver_f1:<10.4f} {fusion_f1:<10.4f} {best_model:<12}\n")
            
            f.write("\n")
            
            # Class-wise insights
            f.write("CLASS-WISE INSIGHTS\n")
            f.write("-" * 80 + "\n")
            
            for i in range(self.num_classes):
                emotion = self.emotion_labels[i]
                fer_f1 = comparison_results['fer']['report'][emotion]['f1-score']
                ver_f1 = comparison_results['ver']['report'][emotion]['f1-score']
                fusion_f1 = comparison_results['fusion']['report'][emotion]['f1-score']
                
                f.write(f"{emotion}:\n")
                
                # Determine which modality is better for this emotion
                if fer_f1 > ver_f1:
                    f.write(f"  - Visual cues are more important (FER: {fer_f1:.4f} vs VER: {ver_f1:.4f})\n")
                elif ver_f1 > fer_f1:
                    f.write(f"  - Acoustic cues are more important (VER: {ver_f1:.4f} vs FER: {fer_f1:.4f})\n")
                else:
                    f.write(f"  - Visual and acoustic cues are equally important\n")
                
                # Fusion effect
                if fusion_f1 > max(fer_f1, ver_f1):
                    f.write(f"  - Fusion successfully improves recognition (Fusion: {fusion_f1:.4f})\n")
                else:
                    f.write(f"  - Fusion does not improve recognition (Fusion: {fusion_f1:.4f})\n")
                
                f.write("\n")
            
            # Overall insights
            f.write("OVERALL INSIGHTS\n")
            f.write("-" * 80 + "\n")
            
            # Count how many emotions are better recognized by each model
            fer_best = 0
            ver_best = 0
            fusion_best = 0
            
            for i in range(self.num_classes):
                emotion = self.emotion_labels[i]
                fer_f1 = comparison_results['fer']['report'][emotion]['f1-score']
                ver_f1 = comparison_results['ver']['report'][emotion]['f1-score']
                fusion_f1 = comparison_results['fusion']['report'][emotion]['f1-score']
                
                best_score = max(fer_f1, ver_f1, fusion_f1)
                if best_score == fer_f1:
                    fer_best += 1
                elif best_score == ver_f1:
                    ver_best += 1
                else:
                    fusion_best += 1
            
            f.write(f"FER model is best for {fer_best} emotions\n")
            f.write(f"VER model is best for {ver_best} emotions\n")
            f.write(f"Fusion model is best for {fusion_best} emotions\n\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 80 + "\n")
            
            if fusion_best > max(fer_best, ver_best):
                f.write("- The fusion approach is effective and should be pursued further\n")
            else:
                f.write("- The fusion approach needs improvement; consider different fusion methods\n")
            
            if fer_best > ver_best:
                f.write("- Visual features are generally more informative for emotion recognition\n")
            elif ver_best > fer_best:
                f.write("- Acoustic features are generally more informative for emotion recognition\n")
            else:
                f.write("- Visual and acoustic features are complementary for emotion recognition\n")
            
            f.write("- Consider emotion-specific models for best performance\n")
            
            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        logger.info(f"Summary report generated: {report_path}")
        return report_path


def main():
    """Main function to run model evaluation"""
    # Check directory structure
    check_directory_structure()
    
    # Set up GPU memory growth
    setup_gpu_memory()
    
    # Create evaluator
    evaluator = ModelEvaluator()
    
    # Run model comparison
    with timer("Complete model comparison"):
        comparison_results = evaluator.compare_models()
    
    # Generate summary report
    if comparison_results:
        evaluator.generate_summary_report(comparison_results)
        logger.info(f"Evaluation complete. Results saved to: {evaluator.save_dir}")
    else:
        logger.error("Evaluation failed. Check logs for details.")

if __name__ == "__main__":
    main()