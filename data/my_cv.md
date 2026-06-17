# Rafif Shafwan

**AI/ML Engineer · Data Scientist · Fullstack Developer**

**Email:** rafifshaf@gmail.com | **Phone:** +62851-1705-7582
**LinkedIn:** linkedin.com/in/rafif-shafwan | **GitHub:** github.com/rafifshaf-fun
**Based in:** Garut, West Java, Indonesia | **Open to:** Remote & Hybrid roles

---

## Who I Am

I'm a freelance Data Scientist and ML Engineer with over 6 years of experience building systems that actually ship — not just notebooks that collect dust. My work spans quantitative finance, computer vision, LLM/RAG applications, and more recently full-stack financial software built for real institutional clients.

Most of my best work exists under NDA. The systems I've delivered to enterprise clients were built to production standards: reliable, monitored, and maintained. What you'll find on my GitHub are intentional demonstrations of the same techniques — clean, explainable, and open to review.

If you're looking for someone who can take a complex problem from whiteboard to live deployment and keep it running, that's what I do.

I'm also an open-source contributor — currently contributing to **career-ops**, a project that helps people navigate career opportunities in tech.

---

## What I Bring to the Table

**MLOps & Production ML**
Building ML systems that actually work in production is hard — I've done it repeatedly. I use MLflow for experiment tracking, Docker for reproducible environments, and Grafana for monitoring. I design systems that retrain automatically, alert when models drift, and require minimal human intervention to stay healthy.

**Computer Vision**
I build multi-stage inference pipelines, not just off-the-shelf model calls. My architecture — detect first with YOLO/SSD, then classify the crop with a fine-grained model — is a real engineering approach that significantly outperforms naive single-pass solutions, especially on small objects.

**LLM & RAG Applications**
Recent focus area. I build RAG systems using LangChain, FAISS, and FastEmbed — and importantly, I evaluate them properly using RAGAS metrics (faithfulness, answer relevancy, context precision). Most people build RAG demos; I build ones you can measure.

**Full-Stack Financial Software**
I design and ship production web applications for institutional clients — complete with double-entry accounting, role-based access control, automated batch processing, and professional-grade reporting. I own the full stack: database schema, backend logic, frontend UI, and Docker deployment.

**Data Analysis & Communication**
I've delivered reports and dashboards to non-technical clients in the public sector. I can translate complex findings into decisions, not just charts.

---

## Experience

**Freelance Data Scientist & ML Engineer**
*2020 – Present*

Over 5 years of independent client work across finance, computer vision, data analytics, and institutional software. Delivered end-to-end systems — from scoping and data engineering through model development, deployment, and monitoring. Enterprise deliverables are confidential; selected examples available on request or via GitHub.

**Key capabilities demonstrated across projects:**

- Automated ML retraining pipelines that run without manual intervention
- Real-time production monitoring with alerting and anomaly detection
- Multi-stage computer vision inference at accuracy levels beyond single-stage baselines
- Full MLOps lifecycle ownership: experiment tracking, versioning, serving, and observability
- End-to-end financial management systems with double-entry accounting and audit logging

---

**Data Analyst | Confidential Clients**
*Jun 2021 – Jul 2022*

Embedded with public sector communications teams to analyze digital media performance across platforms. Built dashboards, delivered reports, and helped non-technical decision-makers act on data they previously couldn't interpret. Conducted EDA on engagement patterns to surface trends and guide campaign strategy.

---

**Data Engineer Intern | Star Energy**
*Jan 2019 – Feb 2019*

Early career exposure to enterprise data infrastructure. Supported migration from legacy file systems to SharePoint, including metadata standardization and data consistency work — an early lesson in why clean data pipelines matter before any model gets built.

---

## Project Highlights

These are selected examples from a broader body of work. Each demonstrates a different capability area. Where a production version was delivered to a client, it was built to enterprise standards and remains confidential.

---

**Sovereign Ledger** *(Full-Stack / Fintech / Institutional Software)*
A comprehensive financial management system built for **KPRI Warga Kesehatan Kabupaten Garut** — a health-worker cooperative in Garut, Indonesia. Features include double-entry accounting with a full Chart of Accounts hierarchy, automated monthly batch processing (compulsory savings, social funds, loan installments), multi-type loan management with automatic installment scheduling, role-based access for five distinct user roles, and professional PDF/CSV reporting with KOP letterhead branding. Deployed on-premise via Docker (PHP-FPM + Nginx + PostgreSQL) with Supervisor-managed scheduling and persistent volume backups. The full accounting ledger was seeded by importing and reconciling legacy Excel data covering the cooperative's complete transaction history.
*Stack: Laravel 13, Livewire 4, Alpine.js, Tailwind CSS, PostgreSQL, Laravel DOMPDF, PhpSpreadsheet, Docker*

---

**Indonesian Stock MLOps Platform** *(Finance / MLOps)*
An end-to-end automated ML system that generates BUY/SELL signals for 45 IDXBLUE blue-chip stocks. Models retrain automatically, performance is monitored in real time via Grafana, and the system runs without daily human oversight. The public version demonstrates the architecture; the production version served live trading decisions.
*Stack: Python, MLflow, Docker, Grafana, Scikit-learn, Pandas*

---

**Multi-Stage Computer Vision Pipeline** *(Computer Vision)*
A two-stage inference architecture: broad object detection using YOLO/SSD, followed by a fine-grained classifier applied to cropped detections. This approach outperforms single-pass detection on small or densely packed objects. The enterprise version was delivered under NDA; the public repo demonstrates the pipeline design.
*Stack: Python, YOLO, OpenCV, PyTorch, TensorFlow*

---

**CV RAG Chatbot** *(LLM / NLP / This Portfolio)*
A retrieval-augmented generation system that answers questions about my resume. Built with LangChain LCEL, FAISS vector storage, FastEmbed (ONNX-based), and Groq's Llama 3.3 — evaluated with RAGAS metrics because a chatbot without measurement is just vibes. Live on Streamlit Community Cloud.
*Stack: Python, LangChain, FAISS, FastEmbed, Groq, Streamlit, RAGAS*

---

**Makmur Grosir – E-Commerce Image Scraper** *(Web Scraping / Automation / Anti-Bot)*
Built a Playwright-based web scraper for wholesale store (2,500+ products) that automatically searches for product images across 4 Indonesian e-commerce platforms — Lazada, Blibli, Shopee, and Tokopedia. The scraper uses sophisticated anti-bot evasion including browser fingerprint spoofing, stealth page scripts, randomized delays, and headless detection circumvention. Runs as a scheduled GitHub Actions workflow (50 products per batch, every 3 hours) with resume support for failed items, action logging, and incremental progress tracking across runs. The pipeline converts an Excel product catalog into a fully image-mapped online storefront with zero manual image entry.
*Stack: Node.js, Playwright, GitHub Actions, JavaScript, Anti-Bot Evasion*

---

## Technical Skills

| Area | Skills |
|---|---|
| Languages | Python (primary), JavaScript, SQL, PHP |
| Human Languages | Indonesian (native), English (professional), Sundanese (native), Malay (conversational) |
| ML & DS | Scikit-learn, TensorFlow, PyTorch, Pandas, NumPy |
| MLOps | MLflow, Docker, Grafana, FastAPI |
| Computer Vision | YOLO, SSD, OpenCV, image segmentation |
| LLM / NLP | LangChain, FAISS, FastEmbed, Groq, RAGAS |
| Web / Full-Stack | Laravel, Livewire, Alpine.js, Tailwind CSS, PostgreSQL |
| Web Scraping & Automation | Playwright, Puppeteer, Anti-Bot Evasion, GitHub Actions |
| Tools | Git, Jupyter, VSCode, SharePoint |

---

## Education

**Bachelor of Computer Science** — Universitas Terbuka *(GPA: 3.36 / 4.0)*
Transferred from UIN Syarif Hidayatullah Jakarta to pursue distance learning while building full-time professional experience in parallel. Completed through distance learning while working professionally full-time.

**Computer Information Systems** *(partial)* — UIN Syarif Hidayatullah Jakarta

---

## Certifications

- Python – Data Science, SanberCode (2020)
- Certified International Specialist Data Modelling (CISDM), Cybertrend (2019)
- Certified International Project Manager Associate (CIPMA), Cybertrend (2019)
- Certified International Supply Chain Associate (CISCA), PASAS Institute (2017)
- Microsoft Technology Associate (MTA), Microsoft (2018)

---

## Availability

Open to remote full-time, contract, or freelance engagements in ML Engineering, Data Science, MLOps, or Full-Stack Development. Available immediately. Prefer async-first teams that value documented, maintainable systems over speed-coded demos.
