# Melanoma Skin Cancer Detection — Full Stack Walkthrough

## Summary of Changes

Three things were done in the `c:\Users\subasri\Desktop\sk` folder:

1. **Fixed dataset paths** in `train_local.py`
2. **Created Flask backend** (`backend/`) — REST API to serve model predictions
3. **Created React frontend** (`frontend/`) — Modern UI for image upload & results

---

## Final Project Structure

```
sk/
├── train_local.py              ← Training script (paths fixed + model save added)
├── dataset/
│   ├── train/
│   │   ├── Benign/
│   │   └── Malignant/
│   └── test/
│       ├── Benign/
│       └── Malignant/
├── output/                     ← Created after training
│   └── efficientnet_b4_model.h5
├── backend/
│   ├── app.py                  ← Flask API server
│   └── requirements.txt        ← Python dependencies
└── frontend/
    ├── index.html
    ├── package.json
    ├── vite.config.js
    └── src/
        ├── main.jsx
        ├── App.jsx
        ├── index.css
        ├── components/
        │   ├── Header.jsx
        │   ├── ImageUpload.jsx
        │   └── PredictionResult.jsx
        └── pages/
            ├── HomePage.jsx
            └── PredictPage.jsx
```

---

## Changes Made

### 1. train_local.py — Path Fix + Model Save

render_diffs(file:///c:/Users/subasri/Desktop/sk/train_local.py)

**What changed:**
- Dataset paths now use **relative paths** (`dataset/train`, `dataset/test`) instead of hardcoded `C:/Users/acer/...`
- Output directory changed to `sk/output/`
- Added `model.save()` after training so the Flask backend can load the `.h5` model

---

### 2. Flask Backend — `backend/app.py`

**API Endpoints:**

| Method | Endpoint        | Description                          |
|--------|----------------|--------------------------------------|
| GET    | `/api/health`  | Server health check                  |
| POST   | `/api/predict` | Upload image → get prediction result |

**Prediction Response (JSON):**
```json
{
  "prediction": "Malignant",
  "confidence": 96.2,
  "raw_probability": 0.962,
  "threshold": 0.5,
  "gradcam": "<base64 JPEG heatmap>",
  "message": "⚠️ High risk of Malignant Melanoma detected..."
}
```

**Features:**
- Loads trained EfficientNetB4 model from `output/efficientnet_b4_model.h5`
- Image preprocessing (resize to 380×380, EfficientNet preprocess_input)
- Grad-CAM heatmap generation returned as base64
- Flask-CORS enabled for React cross-origin requests
- Error handling for missing model, invalid files, etc.

---

### 3. React Frontend

**Pages:**
- **Home Page** — Hero section, feature cards, medical disclaimer
- **Predict Page** — Drag-and-drop image upload, prediction result display

**Components:**
- `Header` — Navigation bar with logo
- `ImageUpload` — Drag-and-drop + click-to-browse with preview
- `PredictionResult` — Shows diagnosis, confidence meter, Grad-CAM heatmap, recommendation

**Design:**
- Dark glassmorphism theme with gradient accents
- Smooth animations and hover effects
- Responsive layout for all screen sizes
- Inter font from Google Fonts

---

## 🚀 How to Run — Step-by-Step Instructions

> [!IMPORTANT]
> You need **two terminals** running simultaneously — one for backend, one for frontend.

---

### Step 1: Train the Model (one-time only)

> Skip this step if you already have `output/efficientnet_b4_model.h5`

Open a terminal and run:

```bash
cd c:\Users\subasri\Desktop\sk
python train_local.py
```

This will:
- Train the EfficientNetB4 model on your dataset
- Save the model to `output/efficientnet_b4_model.h5`
- Generate all visualization plots in `output/`

⏱️ Training takes time depending on your GPU/CPU.

---

### Step 2: Start the Backend (Terminal 1)

```bash
cd c:\Users\subasri\Desktop\sk\backend
pip install -r requirements.txt
python app.py
```

You should see:
```
==================================================
  Melanoma Detection API Server
  http://localhost:5000
==================================================
```

> [!NOTE]
> Keep this terminal **open and running**. The backend must stay on while you use the app.

---

### Step 3: Start the Frontend (Terminal 2)

Open a **new/second terminal** and run:

```bash
cd c:\Users\subasri\Desktop\sk\frontend
npm install
npm run dev
```

You should see:
```
VITE ready in 500 ms

  ➜  Local:   http://localhost:5173/
```

> [!NOTE]
> Keep this terminal **open and running** too.

---

### Step 4: Use the Application

1. Open your browser and go to **http://localhost:5173/**
2. Click **"Start Prediction"** or the **"Predict"** nav link
3. **Upload** a skin lesion image (drag & drop or click to browse)
4. Click **"🔬 Analyze Skin Lesion"**
5. View the result:
   - **Benign** or **Malignant** classification
   - **Confidence percentage** with animated meter
   - **Grad-CAM heatmap** showing which region influenced the prediction
   - **Medical recommendation** message

---

## Quick Reference

| What              | Where                                     | Port  |
|-------------------|------------------------------------------|-------|
| Training Script   | `c:\Users\subasri\Desktop\sk\train_local.py` | —     |
| Backend API       | `c:\Users\subasri\Desktop\sk\backend\`   | 5000  |
| Frontend UI       | `c:\Users\subasri\Desktop\sk\frontend\`  | 5173  |
| Trained Model     | `c:\Users\subasri\Desktop\sk\output\efficientnet_b4_model.h5` | — |
| Dataset           | `c:\Users\subasri\Desktop\sk\dataset\`   | —     |

---

## Verified

- ✅ Frontend compiles and loads successfully on http://localhost:5173
- ✅ Home page renders with hero, features, and disclaimer
- ✅ Predict page shows upload zone and predict button
- ✅ Dark glassmorphism theme with animations working
- ✅ Backend API code ready to serve predictions once model is trained
