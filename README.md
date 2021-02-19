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
### Training
```bash
triage-train --help
```
Train using local data found in `generated_data.txt`:
```bash
triage-train -m radius_variance -c 1 -s 0 -e 100 -lr 0.001
```

### Data Generation
```bash
triage-datagen --help
```

Example generating data using `cyclic` method:
```bash
triage-datagen -m cyclic -sd 2015-01-01 -ed 2020-12-31 -v generated_data.png
```

## Testing

### Run unit and acceptance tests
```bash
docker run -it triage-ml-training
pytest
```

### Run stress test
Benchmarks your machines ability to perform training.
```bash
docker run -it --gpus '"device={num}"' triage-ml-training
python tests/stress_test.py
```