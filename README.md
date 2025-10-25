# predictive-maintenance-power-electronics
ML-based predictive maintenance system for power electronics components (IGBTs, power modules)

## 🎯 Project Overview
An end-to-end machine learning system for predicting failures in power electronics components (IGBTs, power modules) based on real-time sensor data.

## 🔧 Problem Statement
In industrial and medical equipment, unplanned power electronics failures can cause:
- Costly downtime (€5K-10K per day)
- Safety risks
- Customer dissatisfaction

This project builds a predictive maintenance system that:
- Monitors component health in real-time
- Predicts failures 24-48 hours in advance
- Enables preventive maintenance scheduling

## 🏗️ Technical Architecture
- **Data Generation**: Synthetic sensor data based on IGBT thermal and electrical characteristics
- **Data Pipeline**: PostgreSQL database with automated ingestion
- **ML Models**: 
  - Binary classifier for failure prediction
  - Regression model for Remaining Useful Life (RUL)
  - Anomaly detection for abnormal behavior
- **Visualization**: Interactive Streamlit dashboard
- **Deployment**: Docker containerized, cloud-hosted

## 📊 Dataset Features
- Junction temperature (Tj)
- Case temperature (Tc)
- Load current
- Switching frequency
- Thermal resistance
- Operating hours
- Voltage stress

## 🛠️ Tech Stack
- **Languages**: Python
- **Data**: pandas, numpy, scikit-learn
- **Database**: PostgreSQL / InfluxDB
- **ML**: scikit-learn, XGBoost
- **Visualization**: Streamlit, Plotly
- **Deployment**: Docker, AWS/Heroku

## 📈 Project Status
🚧 **IN PROGRESS** - Started October 2024

### Current Phase: Data Generation
- [ ] Define sensor data schema
- [ ] Create synthetic data generator
- [ ] Generate 10,000+ timestamped records
- [ ] Validate data against real-world failure modes

### Upcoming Phases:
- [ ] Data pipeline & database setup
- [ ] Feature engineering
- [ ] ML model development
- [ ] Dashboard creation
- [ ] Deployment

## 💡 Domain Knowledge
This project leverages my 4+ years of experience in power electronics engineering, including:
- Failure analysis of IGBT modules at GE Healthcare
- Understanding of thermal degradation mechanisms
- Real-world knowledge of failure modes (bond wire fatigue, solder degradation, thermal cycling)

## 📫 Contact
Ahmed Selmi - [LinkedIn](your-linkedin-url) - sselmiahmed@gmail.com

---
*This project is part of my transition from Power Electronics Engineering to Industrial Data Engineering / ML.*
