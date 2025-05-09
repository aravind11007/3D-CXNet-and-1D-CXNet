# 3D-CXNet and 1D-CXNet

This repository provides an implementation of **D1D-CXNet and 3D-CXNet: Enhancing Accuracy in Non-Contact HR Monitoring with Limited Computational Resources** Journal paper. 
The required dataset can be downloaded from the [UBFC-rPPG Dataset](https://sites.google.com/view/ybenezeth/ubfcrppg). Please download the dataset 2 realistic

## Steps for Running the Code

### 1. Preparing the Data for 3D-CXNet and 1D-CXNet 
- Intial please download the UBFC dataset from the above link
- Please split the data subject-wise, using 28 subjects for training and 14 subjects for testing. Create separate folders for training and testing, ensuring that the data from the 28 subjects are placed in the training folder and the data from the remaining 14 subjects are placed in the testing folder. This manual split should be organized accordingly, with distinct directories for each dataset.
- To prepare the data for 3D-CXNet, run the "Data_creation_3DCXNet" script to create data chunks for both training and testing. Make sure to update the saving_path variable to specify where you want to save the video chunks, and adjust the Data_path variable to point
  to the appropriate folder for either the training or testing data. You will need to run this data creation script twice: once with the Data_path set to the training folder and once for the testing folder. For 1D-CXNet, use the "Data_creation_1DCXNet" script to generate
  the data using the same above procedure.


For 1D-CXNet, use the "Data_creation_1DCXNet" script to generate the data.

### 2. Data Loader
- Data Loader has been created for both 3D-CXNet and 1D-CXNet under the name Data_loader_3D-CXNet and Data_loader_1D-CXNet
- As long as the Data creation code is not changed, same data loader can be used

### 3. Training the 3D-CXNet and 1D-CXNet
- For Training the 3D-CXNet open the ""Training_3DCXNet" ipynb notebook and "Training_1DCXNet" ipynb notebook for 1D-CXNet.
- The `train_path` and `val_path` variables should be updated to the directories where the video chunks were saved during the execution of the "Data_creation" script.
-  Run the code with-out changing any line for training


### 4. Inferencing the Model
- For inferencing the model, open the same "Training_3DCXNet" or "Training_1D-CXNet" ipynb notebook
- Go to the last part, where the best trained weight can be loaded
- Run the remaining line to the MAE, RMSE, MAPE and pearson-coefficient values



### 6. Heart Rate Estimation
- For heart rate estimation:
  - Execute the `testing_live_stream.ipynb` file as described above.
