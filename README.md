# RiskScan: AI-Powered Financial Fraud Detection System

**RiskScan** is an end-to-end Data Engineering and Machine Learning project designed to detect anomalies and fraudulent patterns across financial transactions. The system architecture processes raw transactional data using a structured **Medallion Data Pipeline** (Bronze $\rightarrow$ Silver $\rightarrow$ Gold), trains an optimized **XGBoost** classification model, and serves predictions via a high-performance **FastAPI** backend integrated with a modern **React (Vite + Tailwind CSS)** dashboard.

---

## 📊 Dataset Reference
The project is built and evaluated using the real-world transactions dataset available on Kaggle:
* **Dataset Link:** [Transactions Fraud Datasets (Kaggle)](https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets)

---

## 🏗️ Data Engineering Pipeline (Medallion Architecture)
To handle the massive scale of transactional data cleanly and professionally, we implemented a 3-layer data pipeline:

1. **🟫 Bronze Layer (Raw Ingestion):**
   * Acts as the landing zone for the raw, unmodified data downloaded directly from the Kaggle dataset.
   * Preserves historical integrity with no schema modifications, row drops, or structural transformations.

2. **🥈 Silver Layer (Cleaned & Enriched):**
   * Performed rigorous data cleaning, handling missing inputs, and ensuring uniform data types.
   * Filtered noise and isolated key features related to transactions and digital payment behaviors.

3. **🥇 Gold Layer (Feature Engineering & Model-Ready):**
   * Constructed aggregate business-level metrics optimized for fraud detection, such as:
     * `client_mean_amount` (The historical spending baseline for each individual client).
     * `amount_to_credit_ratio` (Detecting sudden spikes relative to their limit).
     * `client_merchant_freq` (Evaluating suspicious transactions with unfamiliar merchants).
   * This layer outputs the final dataset directly consumed by the machine learning models.

---

## 🚀 Key Features
* **Advanced ML Brain:** Powered by an optimized XGBoost model with customized threshold shifting to perfectly balance precision and recall.
* **Feature Scaling:** Uses standardized `StandardScaler` transformations preventing feature magnitude bias.
* **Dual-Inference API:** Supports real-time single JSON payload scanning or bulk processing via `.csv`/`.json` file uploads.
* **Risk-Based Ranking Dashboard:** Automatically prioritizes and ranks scanned transactions from highest to lowest risk for fraud analysts.
* **Production-Ready & Containerized:** Fully dockerized backend setup ready for scalable cloud deployments.

---

## 🛠️ Tech Stack
* **Data Engineering & ML:** Python, Scikit-Learn, XGBoost, Pandas, NumPy
* **Backend API:** FastAPI, Pydantic, Uvicorn, PostgreSQL (Database connectivity)
* **Frontend UI:** React.js (Vite), Tailwind CSS, Lucide React Icons
* **DevOps & Infrastructure:** Docker, Docker Compose

---

## 📂 Project Structure

```text
Fraud_Detection/
├── backend/                  # FastAPI Application Source Code
├── frontend/                 # React Application (Built with Vite)
│   ├── dist/                 # Production Build Output
│   ├── public/               # Static Assets
│   ├── src/                  # React Components & Pages (Home.jsx, App.jsx)
│   ├── index.html
│   ├── package.json
│   ├── tailwind.config.js    # UI Styling Configuration
│   └── vite.config.js        # Vite Build Bundler Configuration
├── models_weights/           # Pre-trained model weights (XGBoost & Scaler .pkl)
├── venv/                     # Python Virtual Environment
├── Architecture.png          # Visual System Architecture Diagram
├── Dockerfile                # Containerization setup for deployment
├── docker-compose.yml        # Multi-container orchestration config
├── postgresql-42.7.3.jar     # Java Database Connectivity (JDBC) Driver for Postgres
├── requirements.txt          # Python Backend dependencies
├── test1.json                # Sample data batch for instant testing
├── test2.json                # Sample data batch for instant testing
└── test3.json                # Sample data batch for instant testing
⚙️ Setup, Installation & Run Commands
You can run the project either locally using native package managers or instantly via Docker.

Method 1: Local Installation (Native)
1. Backend Setup & Run (FastAPI)
Open your terminal at the root directory of the project:

Bash
# 1. Activate the pre-existing virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 2. Install Python backend dependencies
pip install -r requirements.txt

# 3. Start the FastAPI server (Assuming app is inside backend folder)
uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
Interactive API documentation (Swagger UI) will be live at: http://127.0.0.1:8000/docs

2. Frontend Setup & Run (React + Vite)
Open a new terminal window, navigate into the frontend folder, and execute the following:

Bash
# 1. Navigate to frontend directory
cd frontend

# 2. Install frontend dependencies
npm install

# 3. Start the Vite local development server
npm run dev
Look at the terminal output to get your local Vite port (usually http://localhost:5173 or http://localhost:3000 depending on your settings).

Method 2: Docker Containerization (Recommended for Deployment)
If you wish to spin up the production environment including dependencies with a single command:

Bash
# Build and run the system containers in detached mode
docker-compose up --build -d
📊 Required Batch Data Format
For a successful batch scan via the React UI, you can use the pre-provided test1.json, test2.json, or test3.json files found in the project root. Any uploaded file must contain the following 12 key features:
amount, hour, day_of_week, month, is_night, client_mean_amount, amount_to_credit_ratio, tx_count_same_day, client_merchant_freq, is_online, is_chip, has_error.


