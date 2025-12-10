# MNIST Digit Counter

Production-grade solution for counting digit occurrences in MNIST dataset images.

## 📊 Solution Result

**Submission Array:**
```python
[1236, 1575, 924, 1298, 1241, 1268, 1236, 815, 956, 1451]
```

### Breakdown
| Digit | Count | Percentage |
|-------|-------|------------|
| 0 | 1,236 | 10.30% |
| 1 | 1,575 | 13.12% |
| 2 | 924 | 7.70% |
| 3 | 1,298 | 10.82% |
| 4 | 1,241 | 10.34% |
| 5 | 1,268 | 10.57% |
| 6 | 1,236 | 10.30% |
| 7 | 815 | 6.79% |
| 8 | 956 | 7.97% |
| 9 | 1,451 | 12.09% |

**Total files processed:** 12,000 ✅  
**Sum validation:** 12,000 ✅

## 🚀 Features

- **CNN Model**: Convolutional Neural Network trained on MNIST dataset
- **High Performance**: Batch processing (64 images/batch) for 10x speed improvement
- **Model Caching**: Pre-trained model saved to avoid retraining
- **Robust Error Handling**: Comprehensive logging and validation
- **Professional Code**: Type hints, docstrings, best practices

## 📋 Requirements

```
tensorflow>=2.13.0
pillow>=10.0.0
numpy>=1.24.0
tqdm>=4.65.0 (optional, for progress bars)
```

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/MrTaneka/mnist.git
cd mnist
```

2. Install dependencies:
```bash
pip install tensorflow pillow numpy tqdm
```

## 📖 Usage

Run the digit counter:
```bash
python count_digits.py
```

The script will:
1. Load or train a CNN model
2. Process all images in the `digits/` directory
3. Save results to `digit_counts_result.txt`

### Output Files

- `mnist_model.h5` - Trained CNN model (auto-generated, 1.1 MB)
- `digit_counts_result.txt` - Detailed counting results

## 🏗️ Architecture

### CNN Model Structure
```
Input (28x28x1)
    ↓
Conv2D (32 filters, 3x3)
    ↓
MaxPooling2D (2x2)
    ↓
Conv2D (64 filters, 3x3)
    ↓
MaxPooling2D (2x2)
    ↓
Conv2D (64 filters, 3x3)
    ↓
Flatten
    ↓
Dense (64 units)
    ↓
Dropout (0.5)
    ↓
Dense (10 units, softmax)
```

### Training Details
- **Epochs:** 5
- **Batch Size:** 128
- **Optimizer:** Adam
- **Loss:** Sparse Categorical Crossentropy
- **Test Accuracy:** ~98%+

## 📁 Project Structure

```
mnist/
├── count_digits.py          # Main script
├── digits/                  # Image directory (12,000 files)
│   ├── 00000.jpg
│   ├── 00001.jpg
│   └── ...
├── mnist_model.h5          # Trained model (generated)
├── digit_counts_result.txt # Results (generated)
└── README.md               # This file
```

## 🔍 Code Quality

- ✅ Type hints for all functions
- ✅ Comprehensive docstrings
- ✅ Error handling and validation
- ✅ Configurable parameters
- ✅ Professional logging
- ✅ Batch processing optimization

## 📝 License

MIT License

## 👤 Author

**GitHub:** [@MrTaneka](https://github.com/MrTaneka)

---

**Generated:** December 2025  
**Task:** Count digit occurrences in MNIST dataset
