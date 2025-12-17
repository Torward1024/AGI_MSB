# Architecture for AGI

## Overview
This document describes an extended fractal graph-based AI architecture designed to achieve real AGI capabilities, overcoming the limitations of current LLMs (sequential processing, vanishing/exploding gradients, rigidity, massive parameter counts, and inefficient resource use).

The system is inspired by the human neocortex: hierarchical, self-similar graphs composed of "columns" (mini-neural networks as nodes). It enables full parallelism during training and inference, autonomous self-learning, dynamic graph expansion, distributed knowledge/memory, and optimal sub-graph routing per query.

Key advantages over LLMs/MoE/HRM:
- No fixed sequential layers — parallel message passing across graph.
- No gradient flow issues — local learning in nodes + distributed RL.
- Dynamic topology — system grows/prunes nodes and clusters on demand.
- Efficient scaling — responsibility and memory distributed fractally.
- True self-evolution — detects knowledge gaps and adapts autonomously.

The architecture is built on top of the existing MSB framework:
- Nodes → `BaseEntity` subclasses
- Clusters → `BaseContainer` subclasses
- Full graph → `Project` subclass
- Orchestration → extended `Manipulator` with neural Orchestrator (GNN)

## Core Components

### 1. Node (Column) — `NodeEntity`
Analogous to a neocortical minicolumn.
- **Mini-Neural Network**: Configurable small network (MLP or lightweight Transformer, 3–5 layers).
- **Long-term Memory**: Vector database (FAISS) storing patterns, experiences, embeddings.
- **Confidence Assessor**: Small MLP evaluating how well the node can handle a given input (output: confidence score 0–1).
- **Interfaces**: Tensor inputs/outputs for message passing.
- **Self-Learning**: Local backpropagation + reinforcement learning (reward signal from global performance).
- **Knowledge Compression**: Autoencoder for sharing subsets of memory/weights.

### 2. Edge — `EdgeEntity`
- **Types**: Excitatory, Inhibitory, Neutral.
- **Adaptive Weights**: Updated via Hebbian rules or RL feedback.
- **Metadata**: Routing priority, knowledge diffusion bandwidth.

### 3. Hierarchical Levels (Fractal Self-Similarity)
- **Micro-level**: Individual nodes and local connections.
- **Meso-level**: Clusters of nodes forming specialized zones (e.g., language, vision, reasoning).
- **Macro-level**: Entire graph representing the full AGI "brain".
- **Recursion**: Each cluster is itself a fractal graph (ClusterContainer can contain sub-Clusters).

### 4. Manipulator — `AGIManipulator` (Orchestrator)
Central controller extended with a neural Orchestrator:
- **Orchestrator NN**: Graph Neural Network (GNN) that receives:
  - Query embedding
  - Current graph state embedding
  - Outputs: optimal routing path, growth signal (bool + location), knowledge-sharing commands.
- **Operations**:
  - Create/delete nodes and edges
  - Trigger parallel learning
  - Manage knowledge diffusion
  - Optimize topology via evolutionary algorithms
- **MSB Integration**: Registers custom `Super` classes (e.g., `GrowthSuper`, `RoutingSuper`, `SharingSuper`).

### 5. Parallelism & Scalability
- **Framework**: Ray for distributed actors (each node can be an actor).
- **Graph Ops**: DGL for efficient message passing.
- **Training**: Asynchronous local updates + periodic global synchronization.

### 6. Self-Learning Mechanisms
- **Local Learning**: Backpropagation + Adam in each node.
- **Global RL**: PPO or similar using query success/efficiency as reward.
- **Dynamic Growth**:
  - Accumulate inputs with low average confidence.
  - Run unsupervised clustering (DBSCAN) on embeddings.
  - If new large cluster detected → spawn new node initialized via distillation from nearest neighbors.
- **Evolutionary Optimization**: Periodic mutation/addition/removal of nodes/edges evaluated by fitness (task performance, latency).

### 7. Knowledge Distribution
- **Diffusion Protocol**:
  - Nodes periodically broadcast compressed knowledge (autoencoder embeddings).
  - Neighbors decide acceptance based on relevance.
  - Manipulator resolves conflicts and approves large transfers.
- **Triggers**: Low confidence, Orchestrator command, or periodic schedule.

## Query Processing Pipeline
1. **Vectorization** — SentenceTransformer or similar → query embedding.
2. **Orchestration** — Orchestrator NN predicts routing, possible growth, sharing.
3. **Propagation** — Parallel message passing along predicted path (DGL + Ray).
4. **Aggregation** — Attention-based combiner or simple mean of node outputs.
5. **De-vectorization** — Decoder NN (fine-tuned transformer) → natural language response.

## Diagrams

### Overall Architecture
```mermaid
graph TD
    A[User Query] --> B[Vectorizer NN]
    B --> C[AGIManipulator (Orchestrator GNN)]
    C --> D[Fractal Graph]
    D --> E[Macro-Level]
    E --> F[Meso-Cluster Language]
    E --> G[Meso-Cluster Reasoning]
    F --> H[Micro-Node 1]
    F --> I[Micro-Node 2]
    H --> J[Mini-NN + Memory + Confidence NN]
    D --> K[Ray Parallelism]
    K --> L[Aggregator NN]
    L --> M[De-Vectorizer NN]
    M --> N[Response]
    C --> O[Growth Decision → New Node/Cluster]
    C --> P[Sharing Commands]
    H --> Q[Knowledge Diffusion]

### Node Internal Structure
```mermaid
graph TD
    A[NodeEntity] --> B[Mini-Neural Network]
    A --> C[FAISS Memory]
    A --> D[Confidence Assessor NN]
    A --> E[Autoencoder (Compression)]
    B --> F[Forward Pass]
    B --> G[Local Learning]
    D --> H[Confidence Score]
### Fractal Hierarchy
```mermaid
graph TD
    A[Macro Graph] --> B[Meso Cluster A]
    A --> C[Meso Cluster B]
    B --> D[Micro Node A1]
    B --> E[Micro Node A2 (contains sub-graph)]
    E --> F[Sub-Micro Node A2.1]