import os 
import random
import numpy as np
import json

base_path = "./data/Dataset"
json_name = "/data.json"
if os.path.exists(base_path+json_name):
    f = open(base_path+json_name)
    dataset = json.load(f)
    f.close()
    train_images = dataset["train_images"]
    train_labels = dataset["train_labels"]

    valid_images = dataset["valid_images"]
    valid_labels = dataset["valid_labels"]
else:
    imagePaths = []
    maskPaths = []

    for folder in os.listdir(base_path):
        if os.path.isdir(base_path+"/"+folder):
            count = 0
            # if(not (folder == "Sim" or folder == "Modified Carla")):
            #     continue
            print("adding file from to dataset:",folder)
            
            for filename in os.listdir(base_path+"/"+folder+"/inputs"):
                if filename == ".DS_Store":
                    continue
                
                if (folder == "Augmented"):
                    if(random.random() < 0.3333):
                        if (count < 3000):
                            imagePaths = imagePaths + [base_path+"/"+folder+"/inputs/"+filename]
                            mask_name = filename.split("Input")[0] + "Label" + filename.split("Input")[1]
                            maskPaths = maskPaths + [base_path+"/"+folder+"/labels/"+ mask_name]
                        else:
                            break
                        count += 1
                    continue
                

                else:
                    imagePaths = imagePaths + [base_path+"/"+folder+"/inputs/"+filename]
                    maskPaths = maskPaths + [base_path+"/"+folder+"/labels/"+filename.split(".")[0]+"_Label.png"]

    print("Image founded : ", str(len(imagePaths)) )                
    # ------------- Instantiate the custom dataset and dataloaders -----------------
    # Do an 85% - 15% split of the images for training and validation
    

    all_idx = np.arange(0,len(imagePaths)).tolist()

    random.shuffle(all_idx)


    split = int(np.ceil(0.8*len(all_idx)))


    train_images = []
    train_labels = []

    valid_images = []
    valid_labels = []

    for idx in all_idx[:split]:
        train_images.append(imagePaths[idx])
        train_labels.append(maskPaths[idx])

    for idx in all_idx[split:]:
        valid_images.append(imagePaths[idx])
        valid_labels.append(maskPaths[idx])

    dataset = {"train_images":train_images,"train_labels" : train_labels,"valid_images" : valid_images,"valid_labels" :valid_labels}

    with open( base_path + json_name, "w") as f:
        json.dump(dataset,f,indent=2)

    print("dataset created at ",base_path + json_name )