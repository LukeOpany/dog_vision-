# Deep Learning Project Template

A step-by-step guide for structuring future deep learning projects.

---

## 📁 Recommended Project Structure

```
my_dl_project/
├── config.py           # All hyperparameters and paths
├── data_loader.py      # Data loading, preprocessing, augmentation
├── model.py            # Model architecture
├── train.py            # Training loop
├── predict.py          # Inference/prediction
├── visualize.py        # Plotting utilities
├── utils.py            # Helper functions
├── requirements.txt    # Dependencies
├── README.md           # Project documentation
├── data/
│   ├── train/
│   ├── val/
│   └── test/
├── models/             # Saved models
├── logs/               # TensorBoard logs
└── notebooks/
    └── exploration.ipynb  # EDA and experimentation
```

---

## 🔧 Step 1: Environment Setup

### 1.1 Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 1.2 Install Dependencies
```bash
pip install tensorflow numpy pandas matplotlib scikit-learn seaborn
pip freeze > requirements.txt
```

### 1.3 Set Random Seeds (in utils.py)
```python
import random
import numpy as np
import tensorflow as tf

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
```

---

## 📊 Step 2: Data Understanding & Exploration

### 2.1 Questions to Answer
- [ ] What is the problem type? (classification, regression, detection, segmentation)
- [ ] What is the input shape? (image size, channels)
- [ ] What is the output shape? (number of classes, continuous value)
- [ ] How large is the dataset?
- [ ] Is there class imbalance?
- [ ] What is the data quality? (missing values, noise, outliers)

### 2.2 Exploration Notebook Checklist
```python
# Load and inspect data
df = pd.read_csv('labels.csv')
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Missing values:\n{df.isnull().sum()}")

# Class distribution
df['label'].value_counts().plot.bar()
plt.title('Class Distribution')

# Sample images
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, ax in enumerate(axes.flat):
    img = plt.imread(sample_paths[i])
    ax.imshow(img)
    ax.set_title(sample_labels[i])
    ax.axis('off')
```

---

## 🔄 Step 3: Data Pipeline (data_loader.py)

### 3.1 Key Principles
1. **Use relative paths** - Never hardcode absolute paths
2. **Shuffle before splitting** - Avoid data leakage and bias
3. **Stratified splits** - Maintain class distribution
4. **Use tf.data** - Efficient batching and prefetching
5. **Augmentation** - Only on training data

### 3.2 Template Code
```python
from pathlib import Path
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# Preprocessing function
def preprocess_image(filepath, label=None):
    image = tf.io.read_file(filepath)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, IMAGE_SIZE)
    if label is not None:
        return image, label
    return image

# Data augmentation
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
])

# Create dataset
def create_dataset(filepaths, labels, is_training=False, augment=False):
    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    if is_training:
        dataset = dataset.shuffle(buffer_size=1000)
    
    dataset = dataset.batch(BATCH_SIZE)
    
    if augment:
        dataset = dataset.map(lambda x, y: (augmentation(x, training=True), y))
    
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# Split with stratification
x_train, x_val, y_train, y_val = train_test_split(
    filepaths, labels,
    test_size=0.2,
    random_state=42,
    stratify=labels  # Maintains class distribution
)
```

---

## 🏗️ Step 4: Model Architecture (model.py)

### 4.1 Transfer Learning Template
```python
import tensorflow as tf
import tensorflow_hub as hub

def create_model(input_shape, num_classes, model_url, trainable_base=False):
    model = tf.keras.Sequential([
        hub.KerasLayer(model_url, trainable=trainable_base, input_shape=input_shape),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    return model
```

### 4.2 Custom CNN Template
```python
def create_custom_cnn(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    
    # Feature extraction
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    
    # Classification head
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model
```

### 4.3 Popular Pre-trained Models
| Model | URL/Name | Input Size | Use Case |
|-------|----------|------------|----------|
| MobileNetV2 | `tf.keras.applications.MobileNetV2` | 224x224 | Mobile/Edge |
| ResNet50 | `tf.keras.applications.ResNet50` | 224x224 | General |
| EfficientNetB0 | `tf.keras.applications.EfficientNetB0` | 224x224 | Balanced |
| VGG16 | `tf.keras.applications.VGG16` | 224x224 | Classic |

---

## 🎯 Step 5: Training (train.py)

### 5.1 Essential Callbacks
```python
callbacks = [
    # Save best model
    tf.keras.callbacks.ModelCheckpoint(
        'models/best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    ),
    
    # Stop when no improvement
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    ),
    
    # Reduce learning rate on plateau
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=3,
        min_lr=1e-7
    ),
    
    # TensorBoard logging
    tf.keras.callbacks.TensorBoard(
        log_dir='logs/' + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
]
```

### 5.2 Training Loop
```python
history = model.fit(
    train_data,
    epochs=100,
    validation_data=val_data,
    callbacks=callbacks
)

# Save final model
model.save('models/final_model.keras')
```

### 5.3 Fine-tuning Strategy
```python
# Phase 1: Train only the head
base_model.trainable = False
model.fit(train_data, epochs=10, ...)

# Phase 2: Fine-tune top layers
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), ...)  # Lower LR!
model.fit(train_data, epochs=10, ...)
```

---

## 📈 Step 6: Evaluation

### 6.1 Metrics to Track
```python
from sklearn.metrics import classification_report, confusion_matrix

# Predictions
y_pred = model.predict(val_data)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_true, axis=1)

# Classification report
print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
```

### 6.2 Training Curves
```python
def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history.history['accuracy'], label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Val')
    axes[0].set_title('Accuracy')
    axes[0].legend()
    
    axes[1].plot(history.history['loss'], label='Train')
    axes[1].plot(history.history['val_loss'], label='Val')
    axes[1].set_title('Loss')
    axes[1].legend()
    
    plt.show()
```

---

## 🚀 Step 7: Inference & Deployment

### 7.1 Single Image Prediction
```python
def predict_image(model, image_path, class_names):
    image = preprocess_image(image_path)
    image = tf.expand_dims(image, 0)
    
    predictions = model.predict(image)[0]
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    
    return predicted_class, confidence
```

### 7.2 Model Export Options
```python
# TensorFlow SavedModel (for TF Serving)
model.save('saved_model/')

# TFLite (for mobile)
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model/')
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# ONNX (for cross-platform)
# pip install tf2onnx
# python -m tf2onnx.convert --saved-model saved_model/ --output model.onnx
```

---

## ✅ Project Checklist

### Before Training
- [ ] Set random seeds for reproducibility
- [ ] Use relative paths (no hardcoded absolute paths)
- [ ] Shuffle data before train/val split
- [ ] Use stratified splitting for classification
- [ ] Check class distribution
- [ ] Normalize inputs (0-1 or ImageNet mean/std)

### During Training
- [ ] Monitor with TensorBoard
- [ ] Save best model with ModelCheckpoint
- [ ] Use EarlyStopping to prevent overfitting
- [ ] Use ReduceLROnPlateau for adaptive learning rate

### After Training
- [ ] Plot training curves
- [ ] Evaluate on validation set
- [ ] Check confusion matrix for problem classes
- [ ] Test on held-out test set
- [ ] Visualize correct and incorrect predictions

### Code Quality
- [ ] No duplicate code (DRY principle)
- [ ] Functions have docstrings
- [ ] Configuration centralized in config.py
- [ ] No global variables in functions
- [ ] Requirements.txt is up to date

---

## 🐛 Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Overfitting | Model too complex / not enough data | Add dropout, augmentation, early stopping |
| Underfitting | Model too simple / learning rate too low | Use larger model, increase LR |
| Loss is NaN | Learning rate too high / bad data | Reduce LR, check for inf/nan in data |
| OOM Error | Batch size too large | Reduce batch size, use mixed precision |
| Slow training | No GPU / inefficient pipeline | Use GPU, add prefetching, parallel loading |

---

## 📚 Resources

- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [Keras Documentation](https://keras.io/api/)
- [TensorFlow Hub Models](https://tfhub.dev/)
- [Papers with Code](https://paperswithcode.com/)
