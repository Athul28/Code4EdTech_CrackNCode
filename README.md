# 📘 Automated OMR Evaluation & Scoring System  
*A Hackathon Project for the **Code4EdTech Challenge** by Innomatics Research Labs*  

---

## 🏆 Team Details  
- **Name:** CrackNCode
- **Theme Chosen:** **Computer Vision (Automated OMR Evaluation & Scoring System)**  

---

## 🚀 Problem Statement  
Traditional OMR evaluation is time-consuming and prone to errors. The challenge was to:  

✅ Automate the process of **reading OMR answer sheets**  
✅ Calculate marks per student in real-time  
✅ Maintain a **student dashboard** with results  
✅ Integrate **Power BI** for data visualization & insights  

Our solution simplifies exam evaluation and provides **instant analytics** for teachers and institutions.  

---

## 🔥 Features  
- **Test Creation Flow**  
  ➝ Create test → Add students (with USN) → Upload scanned OMR sheets → Auto-grading → Dashboard visualization  

- **Automated Evaluation**  
  - Backend reads uploaded OMR answer sheets  
  - Uses CV techniques to detect marked answers  
  - Calculates each student’s score instantly  

- **Student Management**  
  - Add/track students with unique USN  
  - Update marks dynamically  

- **Dashboard & Analytics**  
  - Power BI integration for:  
    - Score distribution  
    - Top performers  
    - Comparative analysis  
    - Real-time insights for educators  

---

## 🛠️ Tech Stack  

**Frontend (T3 Stack)**  
- Next.js (App Router)  
- TailwindCSS + ShadCN UI  
- tRPC + Prisma  
- TypeScript  

**Backend**  
- Node.js + Express (API)  
- PostgreSQL (Database)  
- Computer Vision (OMR Detection & Evaluation)  
- Power BI (Visualization & Reporting)  

**Model & AI/ML**  
- Python (OpenCV, NumPy, Scikit-learn)  
- Custom CV-based prediction model  

---

## 🤖 Prediction Model  

Our system includes a **custom prediction model** designed to automatically evaluate OMR sheets.  

### 🔎 How It Works  
1. **Preprocessing**  
   - Uploaded OMR sheets are converted into grayscale.  
   - Image noise is removed using Gaussian blur & thresholding.  
   - The answer regions are extracted with contour detection in OpenCV.  

2. **Answer Detection**  
   - Each bubble area is segmented.  
   - Pixel intensity analysis is applied to check whether a bubble is filled.  
   - A threshold-based classifier decides if the mark is valid.  

3. **Scoring Logic**  
   - Detected answers are compared with the **stored answer key**.  
   - Student marks are calculated dynamically.  

4. **Performance Prediction**  
   - To add extra intelligence, we trained a **basic ML classifier** (Random Forest) on synthetic OMR data.  
   - It predicts likelihood of **false marks** (cases where a bubble is partially filled or double-marked).  
   - This ensures **robust scoring** even in noisy scans.  

### ⚡ Why It’s Special  
- Lightweight, works in real-time.  
- Hybrid approach → **Computer Vision + Machine Learning**.  
- Can be scaled to evaluate **thousands of sheets** quickly.  

---

## ⚙️ Installation & Setup  

### 1️⃣ Clone Repository  
```bash
git clone https://github.com/<your-repo>.git
cd <your-repo>
```

### 2️⃣ Fronted Setup  
```bash
cd frontend
npm install
```

- Create a `.env` file in `frontend/` with:  
```env
DATABASE_URL=postgresql://<user>:<password>@localhost:5432/omr
PORT=5000
```

- Run database migrations:  
```bash
npx prisma migrate dev
```

- Start frontend:  
```bash
npm run dev
```

Frontend will run on: **http://localhost:3000**  

---

### 3️⃣ Backend Model Setup

```bash
cd backend
pip install -r requirements.txt
```

- Create a `.env` file in `backend/` with your config (e.g. database URL, API keys).

- Start the FastAPI backend:
```bash
uvicorn main:app --reload
```

Backend will run on: **http://localhost:8000**

---

## 🖼️ Project Flow  

1️⃣ **Create Test** – Define exam & answer key  
2️⃣ **Add Students** – Register using USN  
3️⃣ **Upload Answer Sheets** – Scanned OMR uploaded to backend  
4️⃣ **Automated Evaluation** – Marks are calculated instantly  
5️⃣ **Dashboard** – Power BI shows analytics & trends  

---

## 📊 Dashboard Sneak Peek  
✔ Score distribution (histograms, pie charts)  
✔ Performance comparison between students  
✔ Real-time updates as new OMRs are uploaded  
✔ Insights for educators & institutions  

---

## 🎯 Why This Project Stands Out  
✨ End-to-end automation of exam evaluation  
✨ Real-time analytics with Power BI  
✨ CV-based prediction model for robust scoring  
✨ Scalable backend with Prisma + PostgreSQL  
✨ Clean UI powered by T3 stack  
✨ Built in **24 hours under hackathon pressure**  

--- 

## 👥 Team  
- **Athul D Bhandary**
- **Samarh H Shetty**
- **Pratham A Kadekar**

---

## 📌 Submission Requirements  
- ✅ **GitHub Repo** : https://github.com/Athul28/Code4EdTech_CrackNCode
- ✅ **Demo Video** : 
- ✅ **Deployed Web App** : https://code4-ed-tech-crack-n-code.vercel.app/

---

## 🙌 Acknowledgements  
This project was built as part of **Code4EdTech Challenge 2025** organized by *Innomatics Research Labs*.  