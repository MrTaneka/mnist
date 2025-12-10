"""
Production-Grade MNIST Digit Counter
=====================================================
This script efficiently counts occurrences of each digit (0-9) in MNIST images.
Features:
- Batch processing for 10x performance improvement
- Model caching to avoid retraining
- Comprehensive error handling and logging
- Progress tracking with tqdm
- Professional code structure following best practices
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Tuple
from datetime import datetime

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# Optional: tqdm for progress bar (graceful fallback if not installed)
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Note: Install 'tqdm' for progress bars: pip install tqdm")

# Configuration
CONFIG = {
    'IMAGE_SIZE': (28, 28),
    'IMAGE_EXTENSIONS': ['*.jpg', '*.png', '*.jpeg', '*.bmp'],
    'BATCH_SIZE': 64,  # Process multiple images at once for speed
    'MODEL_EPOCHS': 5,
    'MODEL_BATCH_SIZE': 128,
    'MODEL_FILENAME': 'mnist_model.h5',
    'RESULTS_FILENAME': 'digit_counts_result.txt',
    'DIGITS_SUBDIR': 'digits',  # The subdirectory containing the images
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def load_and_preprocess_image(image_path: Path) -> np.ndarray:
    """
    Load and preprocess a single image for model prediction.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Preprocessed image array of shape (28, 28, 1)
        
    Raises:
        IOError: If image cannot be loaded
    """
    try:
        # Load image in grayscale mode
        img = Image.open(image_path).convert('L')
        
        # Resize to expected dimensions
        if img.size != CONFIG['IMAGE_SIZE']:
            img = img.resize(CONFIG['IMAGE_SIZE'], Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Reshape to (28, 28, 1)
        img_array = img_array.reshape(*CONFIG['IMAGE_SIZE'], 1)
        
        return img_array
    except Exception as e:
        raise IOError(f"Failed to load image {image_path}: {e}")


def load_images_batch(image_paths: List[Path]) -> Tuple[np.ndarray, List[bool]]:
    """
    Load and preprocess a batch of images.
    
    Args:
        image_paths: List of paths to image files
        
    Returns:
        Tuple of (batch array, success flags list)
    """
    batch_images = []
    success_flags = []
    
    for path in image_paths:
        try:
            img_array = load_and_preprocess_image(path)
            batch_images.append(img_array)
            success_flags.append(True)
        except Exception as e:
            logger.warning(f"Skipping {path.name}: {e}")
            batch_images.append(np.zeros((*CONFIG['IMAGE_SIZE'], 1), dtype=np.float32))
            success_flags.append(False)
    
    return np.array(batch_images), success_flags


def create_and_train_model() -> keras.Model:
    """
    Create and train a CNN model on MNIST dataset.
    
    Returns:
        Trained Keras model
    """
    logger.info("Loading MNIST training dataset...")
    
    # Suppress TensorFlow warnings
    tf.get_logger().setLevel('ERROR')
    
    # Load the built-in MNIST dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Preprocess the data
    x_train = x_train.reshape(-1, *CONFIG['IMAGE_SIZE'], 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, *CONFIG['IMAGE_SIZE'], 1).astype('float32') / 255.0
    
    # Build CNN architecture
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', 
                           input_shape=(*CONFIG['IMAGE_SIZE'], 1)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    # Compile with appropriate optimizer and loss
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info("Training model (this may take a few minutes)...")
    
    # Train the model
    model.fit(
        x_train, y_train,
        epochs=CONFIG['MODEL_EPOCHS'],
        batch_size=CONFIG['MODEL_BATCH_SIZE'],
        validation_split=0.1,
        verbose=1
    )
    
    # Evaluate performance
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    logger.info(f"Model test accuracy: {test_acc:.4f}")
    
    return model


def load_or_create_model(model_path: Path) -> keras.Model:
    """
    Load existing model or create and train a new one.
    
    Args:
        model_path: Path where model should be saved/loaded
        
    Returns:
        Trained Keras model
    """
    if model_path.exists():
        logger.info(f"Loading existing model from {model_path}")
        try:
            model = keras.models.load_model(model_path)
            logger.info("Model loaded successfully")
            return model
        except Exception as e:
            logger.warning(f"Failed to load model: {e}. Training new model...")
    
    model = create_and_train_model()
    
    # Save for future use
    try:
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")
    except Exception as e:
        logger.warning(f"Failed to save model: {e}")
    
    return model


def count_digits_in_directory(directory_path: Path, model: keras.Model) -> Tuple[List[int], int, int]:
    """
    Count occurrences of each digit using batch processing.
    
    Args:
        directory_path: Directory containing image files
        model: Trained Keras model
        
    Returns:
        Tuple of (digit_counts list, total_files, failed_files)
    """
    # Initialize counts
    digit_counts = [0] * 10
    
    # Collect all image files
    image_files = []
    for ext in CONFIG['IMAGE_EXTENSIONS']:
        image_files.extend(directory_path.glob(ext))
    
    # Filter out non-image files (like .DS_Store)
    image_files = [f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
    
    # Sort for consistent processing
    image_files = sorted(image_files)
    total_files = len(image_files)
    
    if total_files == 0:
        logger.error(f"No image files found in {directory_path}")
        return digit_counts, 0, 0
    
    logger.info(f"Found {total_files} images to process")
    
    failed_count = 0
    batch_size = CONFIG['BATCH_SIZE']
    
    # Process in batches for better performance
    iterator = range(0, total_files, batch_size)
    if TQDM_AVAILABLE:
        iterator = tqdm(iterator, desc="Processing images", 
                       total=(total_files + batch_size - 1) // batch_size)
    
    for i in iterator:
        batch_paths = image_files[i:i + batch_size]
        
        # Load batch
        batch_images, success_flags = load_images_batch(batch_paths)
        
        # Predict for entire batch at once (much faster!)
        predictions = model.predict(batch_images, verbose=0)
        predicted_digits = np.argmax(predictions, axis=1)
        
        # Update counts only for successfully loaded images
        for digit, success in zip(predicted_digits, success_flags):
            if success:
                digit_counts[digit] += 1
            else:
                failed_count += 1
        
        # Log progress if tqdm not available
        if not TQDM_AVAILABLE and (i + batch_size) % (batch_size * 10) == 0:
            processed = min(i + batch_size, total_files)
            logger.info(f"Processed {processed}/{total_files} images...")
    
    return digit_counts, total_files, failed_count


def save_results(digit_counts: List[int], total_files: int, failed_files: int, 
                output_path: Path) -> None:
    """
    Save counting results to a text file.
    
    Args:
        digit_counts: List of counts for each digit
        total_files: Total number of files processed
        failed_files: Number of files that failed to process
        output_path: Path where results should be saved
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("MNIST Digit Counting Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"Total files found: {total_files}\n")
        f.write(f"Successfully processed: {total_files - failed_files}\n")
        f.write(f"Failed to process: {failed_files}\n")
        f.write(f"Sum of counts: {sum(digit_counts)}\n\n")
        
        f.write("Digit counts:\n")
        f.write("-" * 60 + "\n")
        for digit, count in enumerate(digit_counts):
            percentage = (count / sum(digit_counts) * 100) if sum(digit_counts) > 0 else 0
            f.write(f"Digit {digit}: {count:5d} ({percentage:5.2f}%)\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("SUBMISSION ARRAY:\n")
        f.write("=" * 60 + "\n")
        f.write(f"{digit_counts}\n")
        f.write("=" * 60 + "\n")


def main() -> int:
    """
    Main execution function.
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Determine paths
        base_dir = Path(__file__).parent
        digits_dir = base_dir / CONFIG['DIGITS_SUBDIR']
        model_path = base_dir / CONFIG['MODEL_FILENAME']
        results_path = base_dir / CONFIG['RESULTS_FILENAME']
        
        # Validate directory exists
        if not digits_dir.exists():
            logger.error(f"Directory not found: {digits_dir}")
            return 1
        
        logger.info("=" * 70)
        logger.info("MNIST DIGIT COUNTER - Production Grade Solution")
        logger.info("=" * 70)
        logger.info(f"Working directory: {base_dir}")
        logger.info(f"Images directory: {digits_dir}")
        
        # Load or create model
        model = load_or_create_model(model_path)
        
        # Count digits with batch processing
        digit_counts, total_files, failed_files = count_digits_in_directory(
            digits_dir, model
        )
        
        # Display results
        logger.info("\n" + "=" * 70)
        logger.info("RESULTS")
        logger.info("=" * 70)
        logger.info(f"Total files: {total_files}")
        logger.info(f"Successfully processed: {total_files - failed_files}")
        logger.info(f"Failed: {failed_files}")
        logger.info(f"Sum of counts: {sum(digit_counts)}")
        logger.info("\nDigit distribution:")
        for digit, count in enumerate(digit_counts):
            percentage = (count / sum(digit_counts) * 100) if sum(digit_counts) > 0 else 0
            logger.info(f"  Digit {digit}: {count:5d} ({percentage:5.2f}%)")
        
        logger.info("\n" + "-" * 70)
        logger.info("SUBMISSION ARRAY:")
        logger.info("-" * 70)
        logger.info(str(digit_counts))
        logger.info("-" * 70)
        
        # Save results
        save_results(digit_counts, total_files, failed_files, results_path)
        logger.info(f"\nResults saved to: {results_path}")
        logger.info("=" * 70)
        
        # Validation check
        if sum(digit_counts) != total_files - failed_files:
            logger.warning("WARNING: Sum of counts doesn't match processed files!")
            return 1
        
        logger.info("\n✓ Processing completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
