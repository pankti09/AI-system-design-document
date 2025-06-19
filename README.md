
##  **AI-Powered SMS Grief Support System**

### Overview

This project designs an AI-driven system to process SMS replies in a grief support service, categorizing messages into **Sentiment Analysis**, **Response Necessity**, and **Crisis Detection** to ensure appropriate and timely intervention.


### **System Architecture Highlights**

* **Event-Driven Pipeline:** Utilizes **Kafka** for real-time SMS streaming and **FastAPI** for message ingestion.
* **Data Handling:**

  * Structured Data: Stored in **PostgreSQL**
  * Unstructured Data: Managed with **MongoDB/Elasticsearch**
* **NLP & Classification Models:**

  * **Sentiment Analysis:** Fine-tuned DistilBERT
  * **Response Necessity:** BERT-based binary classifier
  * **Crisis Detection:** Hybrid LSTM-Transformer model
* **Vector Database:** Integrates **FAISS/Qdrant/Weaviate** for semantic similarity search using **SBERT/FastText embeddings**.
* **Inference & Deployment:**

  * Real-time inference with **Ray Distributed Computing**
  * On-premise deployment with **Docker + Kubernetes**
  * Model tracking via **MLflow & DVC**
  * Drift Monitoring with **Prometheus & Grafana**


###  **Key Features**

* **Human-in-the-loop Intervention:** Flags low-confidence cases for manual review.
* **Streamlit Dashboard:** For SMS message classification visualization.
* **Continuous Learning:** Retraining pipeline ensures adaptability to evolving grief communication patterns.


###  **Technologies Used**

* **Python, PyTorch, FastAPI, Streamlit**
* **Transformers (BERT, DistilBERT, LSTM, SBERT)**
* **Ray, Kafka, Docker, Kubernetes**
* **PostgreSQL, MongoDB, Elasticsearch, FAISS/Qdrant**
* **MLflow, DVC, Prometheus, Grafana**


### **Ethical & Security Considerations**

* On-premise deployment ensures **privacy and compliance**.
* Bias mitigation strategies included during model training.
* Data anonymization and encryption practices followed.


## **How to Run (Prototype Streamlit Tool)**

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
2. Run Streamlit app:

   ```bash
   streamlit run sms_classifier_app.py
   ```


##  **Future Enhancements**

* Real-time multilingual support.
* Federated learning for distributed training.
* Improved context retention using memory-augmented models.

