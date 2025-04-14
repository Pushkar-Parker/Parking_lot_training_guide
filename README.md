# Parking Lot Dataset - Model Training Guide

This repository provides the steps and tools necessary to train a deep learning model on the **Parking Lot Dataset**. Follow the instructions below to set up your environment and start training with ease.

---

## Prerequisites

Before starting, ensure you have Python installed on your system. Then, install the required Python libraries using the provided `requirements.txt` file.

Open the Parking lot dataset folder and open the "data.yaml" and paste the location for the 'train', 'val' and 'test'. Paste the location of only the images folder from the respective directory

### Installation

```bash
pip install -r requirements.txt
```

---

## Running the Training Script

Once the required dependencies are installed, you're ready to launch the training process.

### Step 1: Run the main script

You can run the training script from your favorite Python IDE (such as VSCode, PyCharm) or directly from the terminal:

```bash
python main.py
```

### Step 2: Provide Training Parameters

After running `main.py`, you will be prompted to input the following parameters:

1. **Number of Epochs**  
   Enter training epochs.

2. **Save Path**  
   Enter the save path.

3. **Experiment Name**  
   Enter a experment name for reference.

---
