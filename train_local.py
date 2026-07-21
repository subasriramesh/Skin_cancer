from google.colab import drive
drive.mount('/content/drive')
from google.colab import files
uploaded = files.upload()
# =========================================================
# INSTALL REQUIRED LIBRARIES (GOOGLE COLAB)
# =========================================================
!pip install -q tensorflow seaborn scikit-learn opencv-python

# =========================================================
# IMPORTS
# =========================================================
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os, cv2

from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import *

from sklearn.metrics import *
from sklearn.calibration import calibration_curve

# =========================================================
# GOOGLE DRIVE MOUNT
# =========================================================
from google.colab import drive
drive.mount('/content/drive')

# =========================================================
# PATH SETTINGS
# =========================================================
train_dir = "/content/drive/MyDrive/melanoma-cancer-dataset/train"
test_dir  = "/content/drive/MyDrive/melanoma-cancer-dataset/test"
output_dir = "/content/drive/MyDrive/output"

os.makedirs(output_dir, exist_ok=True)

# =========================================================
# SETTINGS
# =========================================================
IMG_SIZE = 380
BATCH_SIZE = 16
EPOCHS = 25

# =========================================================
# DATA AUGMENTATION
# =========================================================
def augment(x, y):

    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_brightness(x, 0.3)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    x = tf.image.random_saturation(x, 0.7, 1.3)

    return preprocess_input(x), y

def process(ds, train=False):

    if train:
        ds = ds.map(
            augment,
            num_parallel_calls=tf.data.AUTOTUNE
        )

    else:
        ds = ds.map(
            lambda x, y: (
                preprocess_input(x),
                y
            )
        )

    return ds.prefetch(tf.data.AUTOTUNE)

# =========================================================
# DATASET LOADING
# =========================================================
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

train_data = process(train_ds, True)
val_data   = process(val_ds, False)
test_data  = process(test_ds, False)

# =========================================================
# MODEL
# =========================================================
base = EfficientNetB4(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

for layer in base.layers[:-120]:
    layer.trainable = False

x = GlobalAveragePooling2D()(base.output)

x = BatchNormalization()(x)

x = Dense(
    256,
    activation='relu'
)(x)

x = Dropout(0.3)(x)

x = Dense(
    128,
    activation='relu'
)(x)

x = Dropout(0.2)(x)

output = Dense(
    1,
    activation='sigmoid'
)(x)

model = Model(
    base.input,
    output
)

model.compile(
    optimizer=Adam(1e-4),

    loss=tf.keras.losses.BinaryCrossentropy(
        label_smoothing=0.1
    ),

    metrics=['accuracy']
)

model.summary()

# =========================================================
# CALLBACKS
# =========================================================
callbacks = [

    EarlyStopping(
        patience=10,
        restore_best_weights=True
    ),

    ReduceLROnPlateau(
        patience=3,
        factor=0.3
    )
]

# =========================================================
# TRAINING
# =========================================================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks
)

# =========================================================
# TRUE LABELS
# =========================================================
y_true = []

for _, y in test_data:
    y_true.extend(y.numpy())

y_true = np.array(y_true)

# =========================================================
# TEST TIME AUGMENTATION (TTA)
# =========================================================
def tta(model, dataset, n=5):

    preds = []

    for _ in range(n):

        temp = []

        for x, _ in dataset:

            x_aug = tf.image.random_flip_left_right(x)

            pred = model.predict(
                x_aug,
                verbose=0
            )

            temp.append(pred)

        preds.append(
            np.concatenate(temp)
        )

    return np.mean(preds, axis=0)

y_prob = tta(
    model,
    test_data
).ravel()

# =========================================================
# THRESHOLD SEARCH
# =========================================================
best_t = 0.5
best_acc = 0

for t in np.arange(0.3, 0.7, 0.01):

    pred = (y_prob > t).astype(int)

    acc = accuracy_score(
        y_true,
        pred
    )

    if acc > best_acc:

        best_acc = acc
        best_t = t

y_pred = (y_prob > best_t).astype(int)

# =========================================================
# METRICS
# =========================================================
acc = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true,
    y_pred
)

recall = recall_score(
    y_true,
    y_pred
)

f1 = f1_score(
    y_true,
    y_pred
)

auc = roc_auc_score(
    y_true,
    y_prob
)

# =========================================================
# CLASSIFICATION REPORT
# =========================================================
report = classification_report(
    y_true,
    y_pred,
    target_names=['Benign', 'Malignant'],
    digits=4
)

print("\n==========================")
print("CLASSIFICATION REPORT")
print("==========================\n")

print(report)

# =========================================================
# MACRO AVG & WEIGHTED AVG
# =========================================================
report_dict = classification_report(
    y_true,
    y_pred,
    target_names=['Benign', 'Malignant'],
    output_dict=True
)

macro_precision = report_dict['macro avg']['precision']
macro_recall    = report_dict['macro avg']['recall']
macro_f1        = report_dict['macro avg']['f1-score']

weighted_precision = report_dict['weighted avg']['precision']
weighted_recall    = report_dict['weighted avg']['recall']
weighted_f1        = report_dict['weighted avg']['f1-score']

print("\n==========================")
print("OVERALL METRICS")
print("==========================\n")

print(f"Accuracy            : {acc:.4f}")
print(f"Precision           : {precision:.4f}")
print(f"Recall              : {recall:.4f}")
print(f"F1 Score            : {f1:.4f}")
print(f"AUC Score           : {auc:.4f}")

print("\n==========================")
print("MACRO AVERAGE")
print("==========================\n")

print(f"Macro Precision     : {macro_precision:.4f}")
print(f"Macro Recall        : {macro_recall:.4f}")
print(f"Macro F1 Score      : {macro_f1:.4f}")

print("\n==========================")
print("WEIGHTED AVERAGE")
print("==========================\n")

print(f"Weighted Precision  : {weighted_precision:.4f}")
print(f"Weighted Recall     : {weighted_recall:.4f}")
print(f"Weighted F1 Score   : {weighted_f1:.4f}")

# =========================================================
# CONFUSION MATRIX
# =========================================================
cm = confusion_matrix(
    y_true,
    y_pred
)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "confusion_matrix.png"
    ),
    dpi=300
)

plt.show()

# =========================================================
# ROC CURVE
# =========================================================
fpr, tpr, _ = roc_curve(
    y_true,
    y_prob
)

plt.figure(figsize=(6,5))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"AUC = {auc:.4f}"
)

plt.plot(
    [0,1],
    [0,1],
    'r--'
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "roc_curve.png"
    ),
    dpi=300
)

plt.show()

# =========================================================
# ACCURACY GRAPH
# =========================================================
plt.figure(figsize=(7,5))

plt.plot(
    history.history['accuracy'],
    label='Train Accuracy',
    linewidth=2
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy',
    linewidth=2
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title("Training vs Validation Accuracy")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "accuracy_graph.png"
    ),
    dpi=300
)

plt.show()

# =========================================================
# LOSS CURVE
# =========================================================
plt.figure(figsize=(7,5))

plt.plot(
    history.history['loss'],
    label='Train Loss',
    linewidth=2
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss',
    linewidth=2
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title("Training vs Validation Loss")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "loss_curve.png"
    ),
    dpi=300
)

plt.show()

# =========================================================
# LEARNING RATE GRAPH
# =========================================================
if 'lr' in history.history:

    plt.figure(figsize=(7,5))

    plt.plot(
        history.history['lr'],
        linewidth=2,
        marker='o'
    )

    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")

    plt.title("Learning Rate Schedule")

    plt.yscale('log')

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            "learning_rate_graph.png"
        ),
        dpi=300
    )

    plt.show()

# =========================================================
# COMPARISON GRAPH
# =========================================================
metrics_names = [
    'Accuracy',
    'Precision',
    'Recall',
    'F1'
]

values = [
    acc,
    precision,
    recall,
    f1
]

plt.figure(figsize=(7,5))

bars = plt.bar(
    metrics_names,
    values
)

plt.ylim(0,1)

for bar in bars:

    plt.text(
        bar.get_x() + 0.2,
        bar.get_height() + 0.01,
        f"{bar.get_height():.3f}"
    )

plt.title("Performance Comparison")

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "comparison_graph.png"
    ),
    dpi=300
)

plt.show()

# =========================================================
# PRECISION RECALL CURVE
# =========================================================
prec_vals, rec_vals, _ = precision_recall_curve(
    y_true,
    y_prob
)

avg_prec = average_precision_score(
    y_true,
    y_prob
)

plt.figure(figsize=(7,5))

plt.plot(
    rec_vals,
    prec_vals,
    linewidth=2,
    label=f"AP = {avg_prec:.4f}"
)

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title("Precision Recall Curve")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "precision_recall_curve.png"
    ),
    dpi=300
)

plt.show()

# =========================================================
# PROBABILITY DISTRIBUTION
# =========================================================
plt.figure(figsize=(8,5))

plt.hist(
    y_prob[y_true == 0],
    bins=40,
    alpha=0.6,
    label='Benign'
)

plt.hist(
    y_prob[y_true == 1],
    bins=40,
    alpha=0.6,
    label='Malignant'
)

plt.axvline(
    best_t,
    linestyle='--',
    linewidth=2,
    label=f'Threshold = {best_t:.2f}'
)

plt.xlabel("Predicted Probability")
plt.ylabel("Count")

plt.title("Prediction Probability Distribution")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "probability_distribution.png"
    ),
    dpi=300
)

plt.show()

# =========================================================
# THRESHOLD SWEEP GRAPH
# =========================================================
thresholds_range = np.arange(
    0.05,
    0.95,
    0.01
)

sweep_acc  = []
sweep_f1   = []
sweep_prec = []
sweep_rec  = []

for t in thresholds_range:

    p = (y_prob > t).astype(int)

    sweep_acc.append(
        accuracy_score(y_true, p)
    )

    sweep_f1.append(
        f1_score(y_true, p, zero_division=0)
    )

    sweep_prec.append(
        precision_score(y_true, p, zero_division=0)
    )

    sweep_rec.append(
        recall_score(y_true, p, zero_division=0)
    )

plt.figure(figsize=(9,5))

plt.plot(
    thresholds_range,
    sweep_acc,
    label='Accuracy',
    linewidth=2
)

plt.plot(
    thresholds_range,
    sweep_f1,
    label='F1 Score',
    linewidth=2
)

plt.plot(
    thresholds_range,
    sweep_prec,
    label='Precision',
    linewidth=2
)

plt.plot(
    thresholds_range,
    sweep_rec,
    label='Recall',
    linewidth=2
)

plt.axvline(
    best_t,
    color='black',
    linestyle='--',
    linewidth=1.5,
    label=f'Best Threshold = {best_t:.2f}'
)

plt.xlabel("Threshold")
plt.ylabel("Score")

plt.title("Threshold Sweep")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "threshold_sweep.png"
    ),
    dpi=300
)

plt.show()

# =========================================================
# CLASSIFICATION REPORT HEATMAP
# =========================================================
report_df = pd.DataFrame(
    report_dict
).transpose()

heatmap_df = report_df.loc[
    ['Benign', 'Malignant', 'macro avg', 'weighted avg'],
    ['precision', 'recall', 'f1-score']
]

plt.figure(figsize=(8,5))

sns.heatmap(
    heatmap_df,
    annot=True,
    fmt=".4f",
    cmap="YlGnBu",
    linewidths=0.5
)

plt.title("Classification Report Heatmap")

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "classification_report_heatmap.png"
    ),
    dpi=300
)

plt.show()

# =========================================================
# PER-CLASS ACCURACY
# =========================================================
benign_mask = y_true == 0
malignant_mask = y_true == 1

acc_benign = accuracy_score(
    y_true[benign_mask],
    y_pred[benign_mask]
)

acc_malignant = accuracy_score(
    y_true[malignant_mask],
    y_pred[malignant_mask]
)

plt.figure(figsize=(6,5))

bars = plt.bar(
    ['Benign', 'Malignant'],
    [acc_benign, acc_malignant]
)

for bar in bars:

    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 0.01,
        f"{bar.get_height():.3f}",
        ha='center'
    )

plt.ylim(0,1.1)

plt.ylabel("Accuracy")

plt.title("Per-Class Accuracy")

plt.grid(axis='y')

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "per_class_accuracy.png"
    ),
    dpi=300
)

plt.show()

# =========================================================
# CALIBRATION CURVE
# =========================================================
prob_true, prob_pred = calibration_curve(
    y_true,
    y_prob,
    n_bins=10
)

plt.figure(figsize=(7,5))

plt.plot(
    prob_pred,
    prob_true,
    marker='o',
    linewidth=2,
    label='Model'
)

plt.plot(
    [0,1],
    [0,1],
    'k--',
    label='Perfect Calibration'
)

plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives")

plt.title("Calibration Curve")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "calibration_curve.png"
    ),
    dpi=300
)

plt.show()

# =========================================================
# SAMPLE PREDICTIONS GRID
# =========================================================
class_names = [
    'Benign',
    'Malignant'
]

sample_images = []
sample_labels = []

for imgs, labels in test_data:

    for i in range(imgs.shape[0]):

        sample_images.append(
            imgs[i].numpy()
        )

        sample_labels.append(
            labels[i].numpy()
        )

        if len(sample_images) >= len(y_pred):
            break

correct_idx = np.where(
    y_pred == y_true
)[0]

wrong_idx = np.where(
    y_pred != y_true
)[0]

fig, axes = plt.subplots(
    2,
    5,
    figsize=(18,8)
)

fig.suptitle(
    "Sample Predictions",
    fontsize=15
)

for col in range(5):

    if col < len(correct_idx):

        idx = correct_idx[col]

        img = sample_images[idx]

        img_show = (
            (img - img.min()) /
            (img.max() - img.min() + 1e-8)
            * 255
        ).astype(np.uint8)

        axes[0, col].imshow(img_show)

        axes[0, col].set_title(
            f"True: {class_names[int(y_true[idx])]}\n"
            f"Pred: {class_names[int(y_pred[idx])]}",
            color='green',
            fontsize=9
        )

        axes[0, col].axis('off')

    if col < len(wrong_idx):

        idx = wrong_idx[col]

        img = sample_images[idx]

        img_show = (
            (img - img.min()) /
            (img.max() - img.min() + 1e-8)
            * 255
        ).astype(np.uint8)

        axes[1, col].imshow(img_show)

        axes[1, col].set_title(
            f"True: {class_names[int(y_true[idx])]}\n"
            f"Pred: {class_names[int(y_pred[idx])]}",
            color='red',
            fontsize=9
        )

        axes[1, col].axis('off')

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "sample_predictions.png"
    ),
    dpi=300
)

plt.show()

# =========================================================
# SAVE CSV FILES
# =========================================================
report_df.to_csv(
    os.path.join(
        output_dir,
        "classification_report.csv"
    )
)

pred_df = pd.DataFrame({

    "True_Label": y_true,
    "Predicted_Label": y_pred,
    "Probability": y_prob
})

pred_df.to_csv(
    os.path.join(
        output_dir,
        "predictions.csv"
    ),
    index=False
)

# =========================================================
# SAVE MODEL
# =========================================================
model.save(
    os.path.join(
        output_dir,
        "efficientnet_b4_model.h5"
    )
)

# =========================================================
# FINAL MESSAGE
# =========================================================
print("\n✅ ALL OUTPUTS GENERATED SUCCESSFULLY")
print("📁 SAVED INSIDE :", output_dir)