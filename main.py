# Importing modules
import torch
from ultralytics import YOLO
import os

# Device agnostic code
device = 'cuda:0' if torch.cuda.is_available else 'cpu'

# model training function
def train_model(model: str, # the model path
                data: str, # the .yaml file
                epochs: int, # the number of epochs
                img_size: int, # the image size for training
                save_path: str, # save path for training 
                experiment: str, # experiment name
                device: torch.device= device # defining the device to train on
                ):
    
    # creating save derectory
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    # defining model
    model = YOLO(model).to(device)

    # model training
    model.train(
        data= data,
        epochs= epochs,
        imgsz= img_size,
        device= device,
        project= save_path,
        name= experiment
    )

if __name__ == '__main__':
    model_path = input("Enter the model path: ")
    yaml_path = input("Enter the path for .yaml file path: ")
    epochs_num = int(input("Enter the number of epochs: "))
    save_path = input("Enter the save path for results: ")
    experiment_name = input("Enter the name for experiment: ")
    
    train_model(
        model=model_path,
        data=yaml_path,
        epochs=epochs_num,
        img_size=640,
        save_path=save_path,
        experiment=experiment_name
    )