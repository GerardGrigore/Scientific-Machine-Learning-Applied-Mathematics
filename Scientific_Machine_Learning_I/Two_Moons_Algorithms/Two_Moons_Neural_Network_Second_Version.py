import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng as dr

# -------------------------------------------------------------------
# 1) Dataset: Two moons dataset generation (Supervised Learning Data)
# -------------------------------------------------------------------
def Make_Two_Moons(Number_Of_Samples=600, Noise=0.10, Seed=0, Vertical_Shift=0.3):
    
    Range = np.random.default_rng(Seed)

    Number_Half = Number_Of_Samples // 2
    Part_1 = Range.random(Number_Half) * np.pi
    Part_2 = Range.random(Number_Half) * np.pi

    # Moon number 1 upper:
    Features_Inputs_Vector_X1 = np.c_[np.cos(Part_1), np.sin(Part_1)]
    Labels_Outputs_Vector_Y1 = np.ones(Number_Half)

    # Moon number 2 lower:
    Features_Inputs_Vector_X2 = np.c_[1.0 - np.cos(Part_2), -np.sin(Part_2) + Vertical_Shift]
    Labels_Outputs_Vector_Y2 = np.zeros(Number_Half)

    Features_Inputs_Vector_X = np.vstack([Features_Inputs_Vector_X1, Features_Inputs_Vector_X2])
    Labels_Outputs_Vector_Y = np.concatenate([Labels_Outputs_Vector_Y1, Labels_Outputs_Vector_Y2])

    # Gaussian noise:
    Features_Inputs_Vector_X += Range.normal(0.0, Noise, size=Features_Inputs_Vector_X.shape)

    # Shuffle:
    Indexes = Range.permutation(len(Features_Inputs_Vector_X))
    return Features_Inputs_Vector_X[Indexes], Labels_Outputs_Vector_Y[Indexes]

# -------------------------------------------------------------------
# 2) Activation & Loss functions definition and derivatives
# -------------------------------------------------------------------
# Tanh activation fonction:
def Activation_Function_Hidden_Layer(Number_In_Input):
    return np.tanh(Number_In_Input)

# Sigmoid activation function:
def Sigmoid_Function_Activation_Scalar(Number_Input):
    Sigma_Current = 1.0/(1.0 + np.exp(-Number_Input))
    return Sigma_Current

# BCE Loss function:
def Loss_Function_BCE(Output_Label_True_Boolean,Output_Neural_Network_Probability,Epsilon = 1e-12):
    Output_Neural_Network_Probability = np.clip(Output_Neural_Network_Probability, Epsilon, 1.0 - Epsilon)
    Loss_Function_BCE_Current = -(Output_Label_True_Boolean*np.log(Output_Neural_Network_Probability) + \
                                 (1 - Output_Label_True_Boolean)*np.log(1 - Output_Neural_Network_Probability))
    return Loss_Function_BCE_Current

# Derivative of the Loss function with respect to the Pre_Activation function Z:
def Derivative_Loss_With_Respect_Pre_Activation(Output_Label_True_Boolean,Output_Neural_Network_Probability):
    return Output_Neural_Network_Probability - Output_Label_True_Boolean

# -------------------------------------------------------------------
# 3) Forward pass Neural Network function
# -------------------------------------------------------------------
def Forward_Pass_Function(Number_Of_Layers,Number_Of_Outputs_Layer,Weight_For_Each_Layers_Storage,\
                          Activation_Layer_Current,Bias_For_Each_Layers_Storage,Pre_Activation_Output_Layers,\
                          Activation_Output_Layers,Labels_Outputs_Vector_Y,Index_Random_Selection_Current_Epoch):
    
    for index_Number_Layer in range(1,Number_Of_Layers + Number_Of_Outputs_Layer + 1):
        
        # Pre-activation outputs:
        Pre_Activation_Output_Layer_Current = Weight_For_Each_Layers_Storage[index_Number_Layer].T @ Activation_Layer_Current \
                                              + Bias_For_Each_Layers_Storage[index_Number_Layer]
        # Storage:
        Pre_Activation_Output_Layers[index_Number_Layer] = Pre_Activation_Output_Layer_Current
        # Activation layer outputs:
        # The tanh function is used by the Hidden layers, but the last layer must be the Sigmoid function:
        if not index_Number_Layer == Number_Of_Layers + Number_Of_Outputs_Layer:
            Activation_Output_Layer_Current = Activation_Function_Hidden_Layer(Pre_Activation_Output_Layer_Current)
        else:
            Activation_Output_Layer_Current = Sigmoid_Function_Activation_Scalar(Pre_Activation_Output_Layer_Current)
        # Storage:
        Activation_Output_Layers[index_Number_Layer] = Activation_Output_Layer_Current
        # Update:
        Activation_Layer_Current = Activation_Output_Layer_Current
        
    # Loss function calculation using the output of the Neural Network:
    Loss_Output_Neural_Network_Forward = Loss_Function_BCE(Labels_Outputs_Vector_Y[Index_Random_Selection_Current_Epoch],\
                                                           Activation_Output_Layer_Current,Epsilon = 1e-12)
    
    # Function output:
    return Pre_Activation_Output_Layers, Activation_Output_Layers, Loss_Output_Neural_Network_Forward, Activation_Output_Layer_Current

# -------------------------------------------------------------------
# 4) Training & Learning function with Forward pass & Backpropagation
# -------------------------------------------------------------------
def Training_Learning_Backpropagation_Function(Number_Of_Epochs,Number_Of_Samples,Features_Inputs_Vector_X,Number_Of_Layers,Number_Of_Outputs_Layer,\
                                               Weight_For_Each_Layers_Storage,Bias_For_Each_Layers_Storage,Labels_Outputs_Vector_Y,\
                                               Learning_Rate):
 
    for index_Epochs in range(1,Number_Of_Epochs + 1):
        
        # Matrices & vectors initialization:
        Activation_Output_Layers = {}
        Pre_Activation_Output_Layers = {}
        
        # For the Stochastic descent gradient, select a random couple from the inputs features:
        Index_Random_Selection_Current_Epoch = np.random.randint(0,Number_Of_Samples)
        
        # Initialization of the activation function of the initial layer:
        Activation_Layer_Current = Features_Inputs_Vector_X[Index_Random_Selection_Current_Epoch]
        # Careful reshape to have (N,1) instead of (N,):
        Activation_Layer_Current = Activation_Layer_Current.reshape(-1,1)
        # Storage:
        Activation_Output_Layers[0] = Activation_Layer_Current
        
        # Forward pass loop:
        # ------------------
        Pre_Activation_Output_Layers,\
        Activation_Output_Layers,\
        Loss_Output_Neural_Network_Forward,\
        Activation_Output_Layer_Current = Forward_Pass_Function(Number_Of_Layers,Number_Of_Outputs_Layer,Weight_For_Each_Layers_Storage,\
                                                                Activation_Layer_Current,Bias_For_Each_Layers_Storage,Pre_Activation_Output_Layers,\
                                                                Activation_Output_Layers,Labels_Outputs_Vector_Y,Index_Random_Selection_Current_Epoch)
        
        # Backpropagation calculations:
        # -----------------------------
        for index_Number_Layers in range(Number_Of_Layers + Number_Of_Outputs_Layer, 0, -1):    
            
            if index_Number_Layers == Number_Of_Layers + Number_Of_Outputs_Layer:
                # Derivative of the last Preactivation Layer with respect to the last weights matrix:
                Derivative_Pre_Activation_With_Respect_To_Weights = Activation_Output_Layers[index_Number_Layers - 1]
                # Derivative of the Loss with respect to the last Pre_Activation_Layer vector:
                Derivative_Loss_With_Respect_To_Activation_Vector = Derivative_Loss_With_Respect_Pre_Activation(Labels_Outputs_Vector_Y[Index_Random_Selection_Current_Epoch],\
                                                                                                                Activation_Output_Layer_Current)
                Number_Input_Local = Pre_Activation_Output_Layers[index_Number_Layers]
                Derivative_Sigmoid_Function_Evaluated = Sigmoid_Function_Activation_Scalar(Number_Input_Local)*(1 - Sigmoid_Function_Activation_Scalar(Number_Input_Local))
                # Hadamard product:
                Delta_Derivative_Loss_With_Respect_To_Pre_Activation = Derivative_Loss_With_Respect_To_Activation_Vector*Derivative_Sigmoid_Function_Evaluated
                # Derivative of the Loss with respect to the weights matrix of the last layer:
                Derivative_Loss_With_Respect_To_Weights = Derivative_Pre_Activation_With_Respect_To_Weights @ Delta_Derivative_Loss_With_Respect_To_Pre_Activation.T
                # Derivative of the Loss with respect to the bias vector of the last layer:
                # Derivative_Loss_With_Respect_To_Bias = 1 * Delta_Derivative_Loss_With_Respect_To_Pre_Activation.T
                Derivative_Loss_With_Respect_To_Bias = 1 * Delta_Derivative_Loss_With_Respect_To_Pre_Activation
            else:
                # Since it concerns all the layers but the last one, the activation function is not the sigmoid but the tanh function.
                # Recall that tanh'(z) = 1 - tanh(z)^2 => hence with a = tanh(z): a' = 1 - a^2.
                Derivative_Tanh_Activation_Function = 1 - Activation_Function_Hidden_Layer(Pre_Activation_Output_Layers[index_Number_Layers])**2
                Delta_Derivative_Loss_With_Respect_To_Pre_Activation_Current = (Weight_For_Each_Layers_Storage[index_Number_Layers + 1]@\
                                                                                Delta_Derivative_Loss_With_Respect_To_Pre_Activation)*\
                                                                                (Derivative_Tanh_Activation_Function)
                # Update the delta for the next iterations:
                Delta_Derivative_Loss_With_Respect_To_Pre_Activation = Delta_Derivative_Loss_With_Respect_To_Pre_Activation_Current
                # Derivative of the Pre_Activation with respect to the current weights matrix:
                Derivative_Pre_Activation_With_Respect_To_Weights = Activation_Output_Layers[index_Number_Layers - 1]
                # Derivative of the Loss with respect to the weights matrix of the current layer:
                Derivative_Loss_With_Respect_To_Weights = Derivative_Pre_Activation_With_Respect_To_Weights @ Delta_Derivative_Loss_With_Respect_To_Pre_Activation.T
                # Derivative of the Loss with respect to the bias vector of the current layer:
                Derivative_Loss_With_Respect_To_Bias = Delta_Derivative_Loss_With_Respect_To_Pre_Activation
                
            # Stochastic weights update:
            Weight_For_Each_Layers_Storage[index_Number_Layers] = Weight_For_Each_Layers_Storage[index_Number_Layers] - Learning_Rate*Derivative_Loss_With_Respect_To_Weights
            Bias_For_Each_Layers_Storage[index_Number_Layers] = Bias_For_Each_Layers_Storage[index_Number_Layers] - Learning_Rate*Derivative_Loss_With_Respect_To_Bias            
            
    return Weight_For_Each_Layers_Storage, Bias_For_Each_Layers_Storage, index_Epochs

# -------------------------------------------------------------------
# 5) Plot and display analysis Figures functions
# -------------------------------------------------------------------
def Plot_Dataset_Function(Features_Inputs_Vector_X, Labels_Outputs_Vector_Y, Title="Two Moons"):
    plt.figure()
    plt.scatter(Features_Inputs_Vector_X[:, 0], Features_Inputs_Vector_X[:, 1], c=Labels_Outputs_Vector_Y, cmap="bwr")
    plt.axis("equal")
    plt.title(Title)
    plt.xlabel("Features_X1")
    plt.ylabel("Features_X2")
    plt.show()
    
    
def Plot_Decision_Boundary(Features_Inputs_Vector_X,Labels_Outputs_Vector_Y,Number_Of_Layers,Number_Of_Outputs_Layer,\
                           Weight_For_Each_Layers_Storage,Bias_For_Each_Layers_Storage):

    Features_Inputs_Vector_X_Min, Features_Inputs_Vector_X_Max = Features_Inputs_Vector_X[:,0].min() - 0.5, Features_Inputs_Vector_X[:,0].max() + 0.5
    Features_Inputs_Vector_Y_Min, Features_Inputs_Vector_Y_Max = Features_Inputs_Vector_X[:,1].min() - 0.5, Features_Inputs_Vector_X[:,1].max() + 0.5

    # Mesh grid creation:
    Mesh_Grid_Points_Abscissa,\
    Mesh_Grid_Points_Ordinates = np.meshgrid(np.linspace(Features_Inputs_Vector_X_Min, Features_Inputs_Vector_X_Max, 200),\
                                             np.linspace(Features_Inputs_Vector_Y_Min, Features_Inputs_Vector_Y_Max, 200))

    # Grid points concatenation:
    Grid_Points = np.c_[Mesh_Grid_Points_Abscissa.ravel(), Mesh_Grid_Points_Ordinates.ravel()]

    # Testing on the grid points:
    Probabilities_Predicted = Predict_Probability_Dataset(Grid_Points,Number_Of_Layers,Number_Of_Outputs_Layer,\
                                                          Weight_For_Each_Layers_Storage,Bias_For_Each_Layers_Storage)

    Predicted_Probability = Probabilities_Predicted.reshape(Mesh_Grid_Points_Abscissa.shape)

    plt.figure()
    plt.contourf(Mesh_Grid_Points_Abscissa, Mesh_Grid_Points_Ordinates, Predicted_Probability, levels=50, alpha=0.7)
    plt.contour(Mesh_Grid_Points_Abscissa, Mesh_Grid_Points_Ordinates, Predicted_Probability, levels=[0.5])

    plt.scatter(Features_Inputs_Vector_X[:,0], Features_Inputs_Vector_X[:,1], c=Labels_Outputs_Vector_Y)
    plt.title("Decision Boundary")
    plt.axis("equal")
    plt.show()
    

# ---------------------------------------------------------------------
# 6) Final fuction (used for training) of the Neural Network & Accuracy
# ---------------------------------------------------------------------
def Predict_Probability_Dataset(Features_Inputs_Vector_X,Number_Of_Layers,Number_Of_Outputs_Layer,\
                                Weight_For_Each_Layers_Storage,Bias_For_Each_Layers_Storage):

    # Initialization of the output probabilities:
    Probabilities = np.zeros(len(Features_Inputs_Vector_X))

    for index_Input_Features in range(len(Features_Inputs_Vector_X)):
        
        # Initialization of the storage of the Neural Networks components:
        Activation_Output_Layers = {}
        Pre_Activation_Output_Layers = {}

        # Initialization of the output of the first activation layer:
        Features_Inputs_Vector_X_Reshaped = Features_Inputs_Vector_X[index_Input_Features].reshape(-1,1)
        Activation_Output_Layers[0] = Features_Inputs_Vector_X_Reshaped

        # Forward pass for function evaluation and use of the tuned Neural Network:
        Pre_Activation_Output_Layers,\
        Activation_Output_Layers,\
        _,\
        Activation_Output_Layer_Current = Forward_Pass_Function(Number_Of_Layers,\
                                                                Number_Of_Outputs_Layer,\
                                                                Weight_For_Each_Layers_Storage,\
                                                                Features_Inputs_Vector_X_Reshaped,\
                                                                Bias_For_Each_Layers_Storage,\
                                                                Pre_Activation_Output_Layers,\
                                                                Activation_Output_Layers,\
                                                                np.zeros(len(Features_Inputs_Vector_X)),\
                                                                index_Input_Features)

        Probabilities[index_Input_Features] = Activation_Output_Layer_Current[0,0]

    return Probabilities


# Conversion from probability to final accuracy:
def Accuracy_From_Probabilities(Probabilities, Label_True):
    Label_Predicted = (Probabilities >= 0.5).astype(int)
    Accuracy = np.mean(Label_Predicted == Label_True)
    return Accuracy, Label_Predicted


# -------------------------------------------------------------------
# 7) Main Function of the Neural Network
# -------------------------------------------------------------------
def Main_Function_Neural_Network():
    
    # Data generation:
    Number_Of_Samples = 1500
    Noise = 0.1
    Seed = 1
    Vertical_Shift = 0.1
    Features_Inputs_Vector_X, Labels_Outputs_Vector_Y = Make_Two_Moons(Number_Of_Samples,Noise,Seed,Vertical_Shift)
    Plot_Dataset_Function(Features_Inputs_Vector_X, Labels_Outputs_Vector_Y, Title="Two Moons (raw)")        
    
    # Split the data into training and testing data:
    Train_Ratio = 0.6
    Split_Index = int(Train_Ratio*len(Features_Inputs_Vector_X))

    # Training data:
    Features_Inputs_Vector_X_Train = Features_Inputs_Vector_X[:Split_Index]
    Labels_Outputs_Vector_Y_Train = Labels_Outputs_Vector_Y[:Split_Index]
    
    # Testing data:
    Features_Inputs_Vector_X_Test = Features_Inputs_Vector_X[Split_Index:]
    Labels_Outputs_Vector_Y_Test = Labels_Outputs_Vector_Y[Split_Index:]
    
    # Initilization of important parameters:
    Number_Of_Epochs = 600
    Number_Of_Neurons = 16
    Number_Of_Layers = 2
    Number_Of_Outputs_Layer = 1
    Learning_Rate = 0.1
    
    # List/Array/Vector initialization:
    Is_Number_Of_Neurons_Constant_Per_Layers = 1
    Number_Of_Output_Layers_Vector = []
    Input_Layers_Number_Vector = []
    Weight_For_Each_Layers_Storage = {}
    Bias_For_Each_Layers_Storage = {}
    
    # Extract the number of inputs based on the feature vectors:
    Number_Of_Input_Features = Features_Inputs_Vector_X.shape[1] # 2 in our case.
    Number_Of_Output_Features_Last_Layer = Labels_Outputs_Vector_Y.ndim # 1 in our case.
    Number_Of_Data_Points = Features_Inputs_Vector_X.shape[0] # Number_Of_Samples in our case.
    
    if Is_Number_Of_Neurons_Constant_Per_Layers == 1:
        # It means that every layer has the same amount of outputs, the neuron numbers:
        for index_Number_Layer in range(1,Number_Of_Layers + Number_Of_Outputs_Layer + 1):
            if not index_Number_Layer == Number_Of_Layers + Number_Of_Outputs_Layer:
                Number_Of_Output_Layers_Vector.append(Number_Of_Neurons)
            else:
                Number_Of_Output_Layers_Vector.append(Number_Of_Outputs_Layer)
    
    # Create a vector that stores the number of inputs of each layer of this NN:
    for index_Number_Layer in range(1,Number_Of_Layers + Number_Of_Outputs_Layer + 1):
        if index_Number_Layer == 1:
            Input_Layers_Number_Vector.append(Number_Of_Input_Features)
        else:
            Input_Layers_Number_Vector.append(Number_Of_Neurons)
    
    # For all layers, initialization of the weights and bias matrices and vectors:
    # Weights & biases initialization
    # -------------------------------
    for index_Number_Layer in range(1,Number_Of_Layers + Number_Of_Outputs_Layer + 1):
        
        # Values of mean and standard deviation:
        Mean_Gaussian_Distribution_Current_Layer = 0
        Standard_Deviation_Gaussian_Current_Layer = 1/Input_Layers_Number_Vector[index_Number_Layer - 1]
        
        # Then initialize all the weight terms of the weights of the current layer:
        # Retreive the number of inputs and outputs of the current layer:
        Number_Inputs_Current_Layer = Input_Layers_Number_Vector[index_Number_Layer - 1]
        Number_Outputs_Current_Layer = Number_Of_Output_Layers_Vector[index_Number_Layer - 1]
        
        # Weight array matrix initialization:
        Weight_Matrix_Current_Layer = np.random.normal(Mean_Gaussian_Distribution_Current_Layer,\
                                                       Standard_Deviation_Gaussian_Current_Layer,\
                                                       size=(Number_Inputs_Current_Layer,Number_Outputs_Current_Layer))
        
        # Bias array vector initialization:
        Bias_Vector_Current_Layer = np.random.normal(Mean_Gaussian_Distribution_Current_Layer,\
                                                     1,size=(Number_Outputs_Current_Layer,1))
            
        # Feed the weights and bias matrices & vectors:
        Weight_For_Each_Layers_Storage[index_Number_Layer] = Weight_Matrix_Current_Layer
        Bias_For_Each_Layers_Storage[index_Number_Layer] = Bias_Vector_Current_Layer
        
    # Learning & training of the Neural Network
    # -----------------------------------------
    Weight_For_Each_Layers_Storage,\
    Bias_For_Each_Layers_Storage,\
    index_Epochs = Training_Learning_Backpropagation_Function(Number_Of_Epochs,len(Features_Inputs_Vector_X_Train),Features_Inputs_Vector_X_Train,Number_Of_Layers,Number_Of_Outputs_Layer,\
                                                              Weight_For_Each_Layers_Storage,Bias_For_Each_Layers_Storage,Labels_Outputs_Vector_Y_Train,\
                                                              Learning_Rate)
    # print(index_Epochs)
    
    # Testing and performance assessment:
    # -----------------------------------
    # Then test the tuned Neural-Network using training data:
    Probabilities_Testing = Predict_Probability_Dataset(Features_Inputs_Vector_X_Test,Number_Of_Layers,Number_Of_Outputs_Layer,\
                                                        Weight_For_Each_Layers_Storage,Bias_For_Each_Layers_Storage)

    # Accuracy measurements
    Accuracy_Testing_Performance,Label_Prediction_Testing = Accuracy_From_Probabilities(Probabilities_Testing,Labels_Outputs_Vector_Y_Test)
    print("Test accuracy:", Accuracy_Testing_Performance)
    
    # Decision boundary plot for visualization:
    Plot_Decision_Boundary(Features_Inputs_Vector_X_Test,Labels_Outputs_Vector_Y_Test,Number_Of_Layers,\
                           Number_Of_Outputs_Layer,Weight_For_Each_Layers_Storage,Bias_For_Each_Layers_Storage)
        
    


# To automatize the launching of this script through the function Main_Function:
if __name__ == "__main__":
    Main_Function_Neural_Network()



















