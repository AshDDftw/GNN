# GNN + LLM Project

## Project Overview

This project explores the impact of Language Model-based (LLM) embeddings within Graph Neural Networks (GNNs) for link prediction tasks in knowledge graphs. Specifically, it evaluates the performance differences when initializing a Relational Graph Convolutional Network (RGCN) with embeddings from a pre-trained LLM versus random initialization.

## Objectives

1. **Embedding Impact Analysis**: Assess if and where LLM-based embeddings provide advantages over random initialization within a GNN model.
2. **Link Prediction**: Leverage entity descriptions, where available, to improve link prediction accuracy by enriching embeddings with semantic information.

## Approach

- **Embedding Initialization**:
  - **Pre-trained Model**: The `sentence-transformers/all-MiniLM-L6-v2` model is used to generate initial embeddings for entities.
  - **Fallback Strategy**: For entities lacking descriptions, embeddings are generated based on entity IDs.

- **Model Architecture**:
  - **RGCN**: A Relational Graph Convolutional Network with stacked RGCN layers to learn relationship-specific transformations, crucial for capturing relational nuances in the graph.
  - **Link Prediction Layer**: A fully connected layer calculates link existence probability between node pairs based on their embeddings.

## Dataset and Preprocessing

- **Data Format**: The dataset consists of triples with `head entity`, `relation`, and `tail entity`.
- **Entity Deduplication**: Entities are deduplicated to form a unique set for embedding generation.
- **Embedding Enrichment**: Where available, text-based descriptions enhance the relevance of embeddings.

## Model Workflow

1. **Forward Pass**: Processes initial node features through RGCN layers, embedding both structural and relational information.
2. **Link Prediction**:
   - **Embedding Extraction**: Extracts head and tail node embeddings for each pair.
   - **Feature Concatenation and Scoring**: Concatenates head-tail embeddings and uses a feedforward layer to predict the link probability.

## Training Process

- **Embedding Use**: SentenceTransformer embeddings serve as the initial node features.
- **Training Insights**:
  - Models initialized with LLM embeddings showed improved clustering and link prediction accuracy, particularly for semantically rich entities.
  - Randomly initialized models demonstrated limited relational context and performance.

## Hyperparameter Tuning

| Learning Rate | Layers | Hidden Dimension | Dropout | Final Training Loss |
|---------------|--------|------------------|---------|----------------------|
| 0.001         | 2      | 128              | 0.0     | 0.65                 |
| 0.001         | 2      | 128              | 0.2     | 0.63                 |
| 0.005         | 4      | 128              | 0.2     | 0.30                 |

- **Key Findings**:
  - Higher learning rates (e.g., 0.005) led to faster convergence.
  - Increasing model depth (4 layers) captured more complex patterns.
  - Minor regularization benefits were noted with dropout (0.2).

## Results and Insights

- **Performance Gains**: LLM-based embeddings improved accuracy, especially for complex relationships.
- **Challenges**:
  - **Embedding Fallback**: For entities without descriptions, ID-based embeddings were less effective.
  - **Computational Constraints**: Generating LLM embeddings is resource-intensive for larger datasets.

## Conclusion and Future Work

Using LLM-based embeddings in RGCN models significantly improves link prediction accuracy, especially for complex and nuanced relationships. Future work may involve additional hyperparameter tuning, deeper model architectures, or the integration of attention mechanisms to selectively focus on relevant neighbors.
