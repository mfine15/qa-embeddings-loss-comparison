"""
Ranking test for comparing different loss functions in retrieval tasks
"""

from rank_test.models import QAEmbeddingModel
from rank_test.losses import InfoNCELoss, MSELoss, TripletLoss, ListwiseRankingLoss, create_loss
from rank_test.dataset import QADataset, QABatchedDataset, QATripletDataset, create_dataloaders
from rank_test.evaluate import evaluate_model, compare_models
from rank_test.train import train_model
from rank_test.config import ExperimentConfig, default_configs

def main() -> None:
    print("Hello from rank-test!")