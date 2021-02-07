# Triage ML Training
ML Training CLI for Triage.

Loads data from the Triage DB, trains the model of your choice, and then writes the trained model weights back to the
database.

## Installation

### Docker installation
```bash
git clone https://github.com/TriageCapacityPlanning/Triage-ML-Training && cd Triage-ML-Training
docker build . -t triage-ml-training
```
To run the container:
```bash
# Run without GPU
docker run -it triage-ml-training

# Run with GPU (requires nvidia-docker)
docker run -it --gpus '"device={num}"' triage-ml-training

```

### Local installation (not recommended)
```bash
git clone https://github.com/TriageCapacityPlanning/Triage-ML-Training && cd Triage-ML-Training
pip3 install ml-training
```

## Usage
```bash
triage-train
```