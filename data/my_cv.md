# Rafif Shafwan

**AI/ML Engineer · Data Scientist · Full-Stack Developer**

**Email:** rafifshaf@gmail.com | **Phone:** +62851-1705-7582
**LinkedIn:** linkedin.com/in/rafif-shafwan | **GitHub:** github.com/rafifshaf-fun | **Website:** rafifshaf.fun
**Based in:** Garut, West Java, Indonesia | **Open to:** Remote & Hybrid roles

---

## Professional Summary

AI/ML Engineer and Data Scientist with 6+ years of experience building production systems across LLM/RAG applications, computer vision, MLOps, and full-stack financial software. Proven track record of delivering end-to-end solutions for enterprise and public-sector clients, from data engineering and model development through deployment and monitoring. Enterprise deliverables are confidential; comparable open-source implementations are available on GitHub for technical review.

---

## Core Competencies

| Area | Summary |
|---|---|
| MLOps & Production ML | Experiment tracking (MLflow), containerized deployment (Docker), and real-time monitoring (Grafana) for self-retraining, drift-aware ML systems. |
| Computer Vision | Multi-stage inference pipelines (YOLO/SSD detection followed by fine-grained classification), improving accuracy on small or densely packed objects over single-pass baselines. |
| LLM & RAG Applications | RAG systems — from single-pass pipelines to multi-agent architectures — built with LangChain, FAISS, and FastEmbed, and evaluated with RAGAS metrics (faithfulness, answer relevancy, context precision) to keep answers grounded in the source. |
| Full-Stack Financial Software | Production web applications for institutional clients, including double-entry accounting, role-based access control, automated batch processing, and reporting. |
| Data Analysis & Communication | Dashboards and reports for non-technical public-sector stakeholders, translating technical findings into actionable decisions. |

---

## Experience

**Freelance Data Scientist & ML Engineer**
*2020 – Present*

Independent consultant delivering end-to-end systems across finance, computer vision, data analytics, and institutional software for enterprise and public-sector clients. Owns full project lifecycle: scoping, data engineering, model development, deployment, and monitoring. Enterprise deliverables are confidential; representative examples available on request or via GitHub.

- Built automated ML retraining pipelines requiring no manual intervention
- Implemented real-time production monitoring with alerting and anomaly detection
- Designed multi-stage computer vision inference exceeding single-stage baseline accuracy
- Owned full MLOps lifecycle: experiment tracking, versioning, serving, and observability
- Delivered end-to-end financial management systems with double-entry accounting and audit logging

---

**Data Analyst | Confidential Clients**
*Jun 2021 – Jul 2022*

Embedded with public sector communications teams to analyze digital media performance across platforms. Built dashboards and delivered reports that enabled non-technical decision-makers to act on engagement data. Conducted exploratory data analysis to identify trends and inform campaign strategy.

---

**Data Engineer Intern | Star Energy**
*Jan 2019 – Feb 2019*

Supported migration from legacy file systems to SharePoint, including metadata standardization and data consistency validation.

---

## Project Highlights

Selected examples demonstrating distinct technical capabilities. Where a production version was delivered to a client, it remains confidential; public repositories demonstrate equivalent architecture.

---

**Sovereign Ledger** *(Full-Stack / Fintech / Institutional Software)*
Financial management system built for KPRI Warga Kesehatan Kabupaten Garut, a health-worker cooperative in Garut, Indonesia. Features double-entry accounting with a full Chart of Accounts hierarchy, automated monthly batch processing (compulsory savings, social funds, loan installments), multi-type loan management with automatic installment scheduling, role-based access for five user roles, and PDF/CSV reporting with letterhead branding. Deployed on-premise via Docker (PHP-FPM, Nginx, PostgreSQL) with Supervisor-managed scheduling and persistent volume backups. Legacy Excel accounting data was imported and reconciled to seed the full transaction history.
*Stack: Laravel 13, Livewire 4, Alpine.js, Tailwind CSS, PostgreSQL, Laravel DOMPDF, PhpSpreadsheet, Docker*

---

**Indonesian Stock MLOps Platform** *(Finance / MLOps)*
Automated ML system generating BUY/SELL signals for 45 IDXBLUE blue-chip stocks. Models retrain automatically with performance monitored in real time via Grafana, operating without daily human oversight. Public version demonstrates the architecture used in a production trading system.
*Stack: Python, MLflow, Docker, Grafana, Scikit-learn, Pandas*

---

**Multi-Stage Computer Vision Pipeline** *(Computer Vision)*
Two-stage inference architecture combining broad object detection (YOLO/SSD) with a fine-grained classifier applied to cropped detections, improving accuracy on small or densely packed objects over single-pass detection. Enterprise version delivered under NDA; public repository demonstrates the pipeline design.
*Stack: Python, YOLO, OpenCV, PyTorch, TensorFlow*

---

**Multi-Agent RAG System** *(LLM / RAG / Multi-Agent Systems)*
Retrieval-augmented generation system that answers questions from any source material using collaborating agents — a router that decides how to answer, a retrieval agent that grounds responses in the source, an evaluator that scores factuality and hallucination risk, and a summarizer that manages conversation context. Typed contracts between agents keep the pipeline traceable, and the evaluator enforces faithfulness to the source. Works with any document set — books, manuals, internal knowledge bases.
*Stack: Python, FAISS, Streamlit, RAG, Multi-Agent Systems*

---

**Makmur Grosir – E-Commerce Image Scraper** *(Web Scraping / Automation)*
Playwright-based scraper for a wholesale store (2,500+ products), automatically sourcing product images across four Indonesian e-commerce platforms (Lazada, Blibli, Shopee, Tokopedia). Implements anti-bot evasion including browser fingerprint spoofing, stealth page scripts, and randomized delays. Runs as a scheduled GitHub Actions workflow (50 products per batch, every 3 hours) with resume support and incremental progress tracking, converting an Excel product catalog into a fully image-mapped storefront.
*Stack: Node.js, Playwright, GitHub Actions, JavaScript*

---

## Open Source Contributions

**career-ops** *(AI Job Search Pipeline)* — Contributor to an open-source career management system (740+ offers evaluated, 100+ CVs generated).

- **Resume/CV terminology support** ([PR #1069](https://github.com/santifer/career-ops/pull/1069)): Introduced resume-specific templates and documentation, enabling users to choose between US/industry "resume" and academic "CV" formats. Updated documentation and README translations project-wide.
- **Indonesia job portal providers** ([PR #1086](https://github.com/santifer/career-ops/pull/1086)): Developed `jobstreet.mjs` and `glints.mjs` provider modules integrating public APIs to discover job postings from Indonesia's largest job platforms, including GraphQL querying, URL validation, SSRF protection, pagination, and unit tests.
- **Glints + Jobstreet API restoration** ([PR #1173](https://github.com/santifer/career-ops/pull/1173)): Migrated Glints provider to the `/api/v2-alc/graphql` endpoint and Jobstreet to its v5 REST API following upstream deprecations, restoring scanner functionality. Added Arabic RTL and Japanese CJK font fallback to the CV template; contributed to CI/CD workflow updates and release automation.

*Stack: Node.js, JavaScript, HTML/CSS, Playwright, REST APIs, GraphQL, YAML*

---

**Natively AI Assistant** *(AI Desktop App / Electron)* — Contributor to an open-source Electron + React desktop AI assistant, improving credential persistence and UI usability.

- **Settings panel close button fix** ([PR #364](https://github.com/Natively-AI-assistant/natively-cluely-ai-assistant/pull/364)): Resolved a layout bug where the Settings panel lacked a visible close control due to a `flex-1` layout issue; added a top-right close button with backdrop blur and hover states.
- **Credential persistence fallback fix** ([PR #370](https://github.com/Natively-AI-assistant/natively-cluely-ai-assistant/pull/370), superseded by [PR #373](https://github.com/Natively-AI-assistant/natively-cluely-ai-assistant/pull/373)): Diagnosed and fixed a bug causing silent loss of saved API keys after app restart, caused by keyring and fallback writes sharing a single try-catch block. Fix isolates the keyring write and falls back to AES-256-GCM encryption on failure; incorporated maintainer review feedback including error message redaction and regression tests.

*Stack: TypeScript, Electron, React, Node.js, safeStorage/DPAPI, AES-256-GCM, Tailwind CSS*

---

## Technical Skills

| Area | Skills |
|---|---|
| Languages | Python (primary), JavaScript, SQL, PHP |
| ML & Data Science | Scikit-learn, TensorFlow, PyTorch, Pandas, NumPy |
| MLOps | MLflow, Docker, Grafana, FastAPI |
| Computer Vision | YOLO, SSD, OpenCV, image segmentation |
| LLM / NLP | LangChain, FAISS, FastEmbed, Groq, RAGAS |
| Web / Full-Stack | Laravel, Livewire, Alpine.js, Tailwind CSS, PostgreSQL |
| Web Scraping & Automation | Playwright, Puppeteer, Anti-Bot Evasion, GitHub Actions |
| Tools | Git, Jupyter, VS Code, GitHub |

---

## Education

**Bachelor of Computer Science** — Universitas Terbuka *(GPA: 3.36 / 4.0)*
Completed via distance learning while working full-time. Transferred from UIN Syarif Hidayatullah Jakarta.

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

Open to remote full-time, contract, or freelance engagements in ML Engineering, Data Science, MLOps, or Full-Stack Development. Available immediately.
