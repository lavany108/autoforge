"""
AutoForge Locust Load Testing Script
Simulates realistic concurrent inference traffic targeting POST /predict.
"""

from locust import HttpUser, task, between
import random

SAMPLE_ROWS = [
    {"PassengerId": 1, "Pclass": 3, "Name": "Braund, Mr. Owen Harris", "Sex": "male", "Age": 22.0, "SibSp": 1, "Parch": 0, "Ticket": "A/5 21171", "Fare": 7.25, "Cabin": None, "Embarked": "S"},
    {"PassengerId": 2, "Pclass": 1, "Name": "Cumings, Mrs. John Bradley", "Sex": "female", "Age": 38.0, "SibSp": 1, "Parch": 0, "Ticket": "PC 17599", "Fare": 71.28, "Cabin": "C85", "Embarked": "C"},
    {"PassengerId": 3, "Pclass": 3, "Name": "Heikkinen, Miss. Laina", "Sex": "female", "Age": 26.0, "SibSp": 0, "Parch": 0, "Ticket": "STON/O2. 3101282", "Fare": 7.925, "Cabin": None, "Embarked": "S"},
    {"PassengerId": 4, "Pclass": 1, "Name": "Futrelle, Mrs. Jacques Heath", "Sex": "female", "Age": 35.0, "SibSp": 1, "Parch": 0, "Ticket": "113803", "Fare": 53.1, "Cabin": "C123", "Embarked": "S"},
    {"PassengerId": 5, "Pclass": 3, "Name": "Allen, Mr. William Henry", "Sex": "male", "Age": 35.0, "SibSp": 0, "Parch": 0, "Ticket": "373450", "Fare": 8.05, "Cabin": None, "Embarked": "S"},
]


class AutoForgeInferenceUser(HttpUser):
    wait_time = between(0.01, 0.05)  # Fast requests for load testing

    @task(5)
    def predict_single_row(self):
        row = random.choice(SAMPLE_ROWS)
        self.client.post("/predict", json={"rows": [row]}, name="/predict (1 row)")

    @task(2)
    def predict_batch_rows(self):
        batch = random.sample(SAMPLE_ROWS, k=random.randint(2, len(SAMPLE_ROWS)))
        self.client.post("/predict", json={"rows": batch}, name="/predict (batch)")

    @task(1)
    def health_check(self):
        self.client.get("/health", name="/health")
