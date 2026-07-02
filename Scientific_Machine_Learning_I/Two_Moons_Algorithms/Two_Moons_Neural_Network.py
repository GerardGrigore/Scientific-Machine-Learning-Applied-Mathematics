import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------------------------
# 1) Dataset: Two moons dataset generation (Supervised Learning Data)
# -------------------------------------------------------------------
# Goal:
# * Produce a dataset (Features_Inputs_Vector_X, Labels_Outputs_Vector_Y) where:
# * Features_Inputs_Vector_X : list of points [Features_Inputs_Vector_X1, Features_Inputs_Vector_X2].
# * Labels_Outputs_Vector_Y : labels 0/1.
#
# Careful:
# * Shuffle well before training.
# * Add some noise to increase realism.
# * Keep a split train/test to evaluate the generalization.

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

# # Plot of the two-moons generation function:
# # Variables initialization:
# Features_Inputs_Vector_X, Labels_Outputs_Vector_Y = Make_Two_Moons()

# # Coordinates extraction:
# Features_X1 = [Point[0] for Point in Features_Inputs_Vector_X]
# Features_X2 = [Point[1] for Point in Features_Inputs_Vector_X]

# plt.scatter(Features_X1, Features_X2, c=Labels_Outputs_Vector_Y,cmap="bwr")
# plt.title("Two Moons Datapoints")
# plt.xlabel("Features X1")
# plt.ylabel("Features X2")
# plt.axis("equal")
# plt.show()



# Dataset partitioning to have:
# * 80% of data for the training.
# * 20% of the data for the testing.
def Train_Test_Split(Features_Inputs_Vector_X, Labels_Outputs_Vector_Y, Test_Ratio=0.2):
    Number_Of_Features_Input = len(Features_Inputs_Vector_X)
    Number_Of_Test_Features = int(Number_Of_Features_Input*Test_Ratio)
    Features_Inputs_Vector_X_Test, Labels_Outputs_Vector_Y_Test = Features_Inputs_Vector_X[:Number_Of_Test_Features], Labels_Outputs_Vector_Y[:Number_Of_Test_Features]
    Features_Inputs_Vector_X_Train, Labels_Outputs_Vector_Y_Train = Features_Inputs_Vector_X[Number_Of_Test_Features:], Labels_Outputs_Vector_Y[Number_Of_Test_Features:]
    return Features_Inputs_Vector_X_Train, Labels_Outputs_Vector_Y_Train, Features_Inputs_Vector_X_Test, Labels_Outputs_Vector_Y_Test

# -------------------------------------------------------------
# 2) Activation functions
# -------------------------------------------------------------

# Sigmoid activation function:
def Sigmoid_Function_Activation_Scalar(Number_Input):
    Sigma_Current = 1.0/(1.0 + np.exp(-Number_Input))
    return Sigma_Current

# Output layer activation function:
def Tangent_Hyperbolic(Number_Input):
    return np.tanh(Number_Input)

# -------------------------------------------------------------
# 3) Loss function & its derivative
# -------------------------------------------------------------

# BCE Loss function:
def Loss_Function_BCE(Output_Label_True_Boolean,Output_Neural_Network_Probability,Epsilon = 1e-12):
    Output_Neural_Network_Probability = np.clip(Output_Neural_Network_Probability, Epsilon, 1.0 - Epsilon)
    Loss_Function_BCE_Current = -(Output_Label_True_Boolean*np.log(Output_Neural_Network_Probability) + \
                                 (1 - Output_Label_True_Boolean)*np.log(1 - Output_Neural_Network_Probability))
    return Loss_Function_BCE_Current

# Derivative of the Loss function with respect to the pre-activation function Z:
def Derivative_Loss_With_Respect_Pre_Activation(Output_Label_True_Boolean,Output_Neural_Network_Probability):
    return Output_Neural_Network_Probability - Output_Label_True_Boolean

# -------------------------------------------------------------
# 4) Parameters Initialization
# -------------------------------------------------------------
# The goal is to parametrize the weights for each layers:
# * Weight maxtrix layer 1: length 16*2.
# * Bias weight matrix layer 1: length 16*1.
# * Weight matrix layer 2: length 16*16.
# * Bias weight matrix layer 2: length 16*1.
# * Weight matrix output layer: length 1*16.
# * Bias matrix output layer: length 1*1.
def Parameters_Bias_Weight_Initialization(Seed=0,Hidden_Number=16,Scale=0.5):
    Range_Generation = np.random.default_rng(Seed)
    Parameters_Bias_Weights = {
        "Weights_Layer_1": Range_Generation.uniform(-Scale, Scale, size=(Hidden_Number, 2)),
        "Bias_Layer_1": np.zeros(Hidden_Number),
        "Weights_Layer_2": Range_Generation.uniform(-Scale, Scale, size=(Hidden_Number, Hidden_Number)),
        "Bias_Layer_2": np.zeros(Hidden_Number),
        "Weights_Layer_3": Range_Generation.uniform(-Scale, Scale, size=(1, Hidden_Number)),
        "Bias_Layer_3": np.zeros(1),
    }
    return Parameters_Bias_Weights

# -------------------------------------------------------------
# 5) Vectorial Forward Pass of the Neural Network function
# -------------------------------------------------------------
# Goal:
# * Knowing the inputs X, calculate the probability y_hat.
def Forward_Pass_Neural_Network_Vectorial(Parameters_Bias_Weights, Features_Inputs_Vector_X):
    # Weights & biases extraction from the created dictionary:
    Weights_Layer_1, Bias_Layer_1 = Parameters_Bias_Weights["Weights_Layer_1"], Parameters_Bias_Weights["Bias_Layer_1"]
    Weights_Layer_2, Bias_Layer_2 = Parameters_Bias_Weights["Weights_Layer_2"], Parameters_Bias_Weights["Bias_Layer_2"]
    Weights_Layer_3, Bias_Layer_3 = Parameters_Bias_Weights["Weights_Layer_3"], Parameters_Bias_Weights["Bias_Layer_3"]

    Pre_Activation_Layer_1 = Features_Inputs_Vector_X @ Weights_Layer_1.T + Bias_Layer_1                   # Number_Of_Features_Pairs * Hidden_Number (16).
    Activation_Function_Layer_1 = Tangent_Hyperbolic(Pre_Activation_Layer_1)                               # Number_Of_Features_Pairs * Hidden_Number (16).
    Pre_Activation_Layer_2 = Activation_Function_Layer_1 @ Weights_Layer_2.T + Bias_Layer_2                # Number_Of_Features_Pairs * Hidden_Number (16).
    Activation_Function_Layer_2 = Tangent_Hyperbolic(Pre_Activation_Layer_2)                               # Number_Of_Features_Pairs * Hidden_Number (16).
    Pre_Activation_Layer_Output = Activation_Function_Layer_2 @ Weights_Layer_3.T + Bias_Layer_3           # Number_Of_Features_Pairs * Output_Layer_Numner (1).
    # Output_Label_Probability_Estimated stands for y_hat:
    Output_Label_Probability_Estimated = Sigmoid_Function_Activation_Scalar(Pre_Activation_Layer_Output)   # Number_Of_Features_Pairs * Output_Layer_Numner (1).
    # Acts as a cache:
    Forward_Pass_Memory_Save = {"Features_Inputs_Vector_X": Features_Inputs_Vector_X, "Pre_Activation_Layer_1": Pre_Activation_Layer_1,\
                                "Activation_Function_Layer_1": Activation_Function_Layer_1,"Pre_Activation_Layer_2": Pre_Activation_Layer_2,\
                                "Activation_Function_Layer_2": Activation_Function_Layer_2, "Pre_Activation_Layer_Output": Pre_Activation_Layer_Output,\
                                "Output_Label_Probability_Estimated": Output_Label_Probability_Estimated}
        
    return Output_Label_Probability_Estimated, Forward_Pass_Memory_Save


# -------------------------------------------------------------
# 6) Vectorial Backward Pass:
# -------------------------------------------------------------
# The goal is to calculate all the gradients for the weights and bias matrices/vectors.
# The chain rule is used and recall that tanh'(z) = 1 - tanh(z)^2 => hence with a = tanh(z): a' = 1 - a^2.
def Backward_Pass_Neural_Network_Vectorial(Parameters_Bias_Weights, Forward_Pass_Memory_Save, Labels_Outputs_Vector_Y):

    Weights_Layer_2 = Parameters_Bias_Weights["Weights_Layer_2"]
    Weights_Layer_3 = Parameters_Bias_Weights["Weights_Layer_3"]
    Features_Inputs_Vector_X = Forward_Pass_Memory_Save["Features_Inputs_Vector_X"]
    Activation_Function_Layer_1 = Forward_Pass_Memory_Save["Activation_Function_Layer_1"]
    Activation_Function_Layer_2 = Forward_Pass_Memory_Save["Activation_Function_Layer_2"]
    Output_Label_Probability_Estimated = Forward_Pass_Memory_Save["Output_Label_Probability_Estimated"]
    
    # Reshape the actual output probability vector of size (N,) to force it to be of size (N,1):
    Labels_Outputs_Vector_Y = Labels_Outputs_Vector_Y.reshape(-1, 1)        
    
    # Number of samples:
    Number_Of_Samples = Features_Inputs_Vector_X.shape[0]

    # Output layer gradient calculation:
    Derivative_Loss_With_Respect_Pre_Activation_Gradient = Derivative_Loss_With_Respect_Pre_Activation(Labels_Outputs_Vector_Y,Output_Label_Probability_Estimated)

    # Weights and biases of layer 3 calculation:
    Derivative_Loss_With_Respect_Weights_Layer_3 = (Derivative_Loss_With_Respect_Pre_Activation_Gradient.T @ Activation_Function_Layer_2) / Number_Of_Samples                  
    Derivative_Loss_With_Respect_Bias_Layer_3 = Derivative_Loss_With_Respect_Pre_Activation_Gradient.mean(axis=0)                   
    # Backpropagation into Activation_Function_Layer_2:
    Derivative_Loss_With_Respect_Activation_Layer_2 = Derivative_Loss_With_Respect_Pre_Activation_Gradient @ Weights_Layer_3                          
    # Through activation function tanh number 2:
    Derivative_Loss_With_Respect_Pre_Activation_Layer_2 = Derivative_Loss_With_Respect_Activation_Layer_2 * (1.0 - Activation_Function_Layer_2**2)

    # Weights and biases of layer 2 calculation:
    Derivative_Loss_With_Respect_Weights_Layer_2 = (Derivative_Loss_With_Respect_Pre_Activation_Layer_2.T @ Activation_Function_Layer_1) / Number_Of_Samples
    Derivative_Loss_With_Respect_Bias_Layer_2 = Derivative_Loss_With_Respect_Pre_Activation_Layer_2.mean(axis=0)
    # Backpropagation into Activation_Function_Layer_1:
    Derivative_Loss_With_Respect_Activation_Layer_1 = Derivative_Loss_With_Respect_Pre_Activation_Layer_2 @ Weights_Layer_2                              
    # Through activation function tanh number 1:
    Derivative_Loss_With_Respect_Pre_Activation_Layer_1 = Derivative_Loss_With_Respect_Activation_Layer_1 * (1.0 - Activation_Function_Layer_1**2)                   

    # Weights and biases of layer 1 calculation:
    Derivative_Loss_With_Respect_Weights_Layer_1 = (Derivative_Loss_With_Respect_Pre_Activation_Layer_1.T @ Features_Inputs_Vector_X) / Number_Of_Samples                  
    Derivative_Loss_With_Respect_Bias_Layer_1 = Derivative_Loss_With_Respect_Pre_Activation_Layer_1.mean(axis=0)                      

    # Backward calculations storing:
    Gradients_Save = {"Derivative_Loss_With_Respect_Weights_Layer_1": Derivative_Loss_With_Respect_Weights_Layer_1,\
                      "Derivative_Loss_With_Respect_Bias_Layer_1": Derivative_Loss_With_Respect_Bias_Layer_1,\
                      "Derivative_Loss_With_Respect_Weights_Layer_2": Derivative_Loss_With_Respect_Weights_Layer_2,\
                      "Derivative_Loss_With_Respect_Bias_Layer_2": Derivative_Loss_With_Respect_Bias_Layer_2,\
                      "Derivative_Loss_With_Respect_Weights_Layer_3": Derivative_Loss_With_Respect_Weights_Layer_3,\
                      "Derivative_Loss_With_Respect_Bias_Layer_3": Derivative_Loss_With_Respect_Bias_Layer_3}
    return Gradients_Save


# -------------------------------------------------------------
# 7) Stochastic Gradient Descent update
# -------------------------------------------------------------

def Stochastic_Gradient_Descent_Step_Function(Parameters_Bias_Weights, Gradients_Save, Learning_Rate):
    Parameters_Bias_Weights["Weights_Layer_1"] -= Learning_Rate * Gradients_Save["Derivative_Loss_With_Respect_Weights_Layer_1"]
    Parameters_Bias_Weights["Bias_Layer_1"] -= Learning_Rate * Gradients_Save["Derivative_Loss_With_Respect_Bias_Layer_1"]
    Parameters_Bias_Weights["Weights_Layer_2"] -= Learning_Rate * Gradients_Save["Derivative_Loss_With_Respect_Weights_Layer_2"]
    Parameters_Bias_Weights["Bias_Layer_2"] -= Learning_Rate * Gradients_Save["Derivative_Loss_With_Respect_Bias_Layer_2"]
    Parameters_Bias_Weights["Weights_Layer_3"] -= Learning_Rate * Gradients_Save["Derivative_Loss_With_Respect_Weights_Layer_3"]
    Parameters_Bias_Weights["Bias_Layer_3"] -= Learning_Rate * Gradients_Save["Derivative_Loss_With_Respect_Bias_Layer_3"]


# -------------------------------------------------------------
# 8) Metrics for measurements
# -------------------------------------------------------------
# Goal of binary accuracy through a standard threshold of 0.5.

def Accuracy_Metric_Function(Parameters_Bias_Weights, Features_Inputs_Vector_X, Labels_Outputs_Vector_Y, Threshold=0.5):
    Output_Label_Probability_Estimated, _ = Forward_Pass_Neural_Network_Vectorial(Parameters_Bias_Weights, Features_Inputs_Vector_X)
    Prediction = (Output_Label_Probability_Estimated.reshape(-1) >= Threshold).astype(float)
    return (Prediction == Labels_Outputs_Vector_Y).mean()

# -------------------------------------------------------------
# 9) Training loop using Batch Gradient Descent
# -------------------------------------------------------------
# The following actions must be ordered:
# * Actions: Forward_Pass, Loss calculation, Backward Pass, Update of the Weights & Biases.
# Difference between learning & training:
# * Training is the loop hereunder.
# * Learning is the update of the parameters at each epochs, through Stochastic_Gradient_Descent_Step_Function.

def Train_Neural_Network_Function(Parameters_Bias_Weights, Features_Inputs_Vector_X_Train, Labels_Outputs_Vector_Y_Train,\
                                  Features_Inputs_Vector_X_Test, Labels_Outputs_Vector_Y_Test, Learning_Rate=0.1, Epochs=500, Print_Every=50):
    for Epochs in range(1, Epochs + 1):
        Output_Label_Probability_Estimated, Forward_Pass_Memory_Save = Forward_Pass_Neural_Network_Vectorial(Parameters_Bias_Weights, Features_Inputs_Vector_X_Train)
        Loss_Function_BCE_Current = Loss_Function_BCE(Labels_Outputs_Vector_Y_Train.reshape(-1, 1),Output_Label_Probability_Estimated).mean()

        Gradients_Save = Backward_Pass_Neural_Network_Vectorial(Parameters_Bias_Weights, Forward_Pass_Memory_Save, Labels_Outputs_Vector_Y_Train)
        Stochastic_Gradient_Descent_Step_Function(Parameters_Bias_Weights, Gradients_Save, Learning_Rate)

        if Epochs % Print_Every == 0:
            Accuracy_Training = Accuracy_Metric_Function(Parameters_Bias_Weights, Features_Inputs_Vector_X_Train, Labels_Outputs_Vector_Y_Train)
            Accuracy_Testing = Accuracy_Metric_Function(Parameters_Bias_Weights, Features_Inputs_Vector_X_Test, Labels_Outputs_Vector_Y_Test)
            print(f"epoch {Epochs:4d} | Loss_Function_BCE_Current {Loss_Function_BCE_Current:.4f} | train_acc {Accuracy_Training:.3f} | test_acc {Accuracy_Testing:.3f}")



# -------------------------------------------------------------
# 10) Plots: Dataset & decision boundary
# -------------------------------------------------------------
# Goal:
# * Visualize the data and the learned boudary/frontier.
# Careful:
# * The boundary must be evaluated on a grid.

def Plot_Dataset_Function(Features_Inputs_Vector_X, Labels_Outputs_Vector_Y, Title="Two Moons"):
    plt.figure()
    plt.scatter(Features_Inputs_Vector_X[:, 0], Features_Inputs_Vector_X[:, 1], c=Labels_Outputs_Vector_Y, cmap="bwr")
    plt.axis("equal")
    plt.title(Title)
    plt.xlabel("Features_X1")
    plt.ylabel("Features_X2")
    plt.show()


def Plot_Decision_Boundary(Parameters_Bias_Weights, Features_Inputs_Vector_X, Labels_Outputs_Vector_Y, Grid_Step=0.02, Title="Decision boundary"):
    
    Abscissa_Point_Minimal, Abscissa_Point_Maximal = Features_Inputs_Vector_X[:, 0].min() - 0.5, Features_Inputs_Vector_X[:, 0].max() + 0.5
    Ordinates_Point_Minimal, Ordinates_Point_Maximal = Features_Inputs_Vector_X[:, 1].min() - 0.5, Features_Inputs_Vector_X[:, 1].max() + 0.5
    
    # 3D grid creation:
    Grid_X_Axis, Grid_Y_Axis = np.meshgrid(
        np.arange(Abscissa_Point_Minimal, Abscissa_Point_Maximal, Grid_Step),
        np.arange(Ordinates_Point_Minimal, Ordinates_Point_Maximal, Grid_Step)
    )
    Grid = np.c_[Grid_X_Axis.ravel(), Grid_Y_Axis.ravel()]

    Output_Label_Probability_Estimated, _ = Forward_Pass_Neural_Network_Vectorial(Parameters_Bias_Weights, Grid)
    Grid_Z_Axis = Output_Label_Probability_Estimated.reshape(Grid_X_Axis.shape)

    plt.figure()
    plt.contourf(Grid_X_Axis, Grid_Y_Axis, Grid_Z_Axis, levels=30) 
    plt.scatter(Features_Inputs_Vector_X[:, 0], Features_Inputs_Vector_X[:, 1], c=Labels_Outputs_Vector_Y, cmap="bwr", edgecolors="k", linewidths=0.2)
    plt.axis("equal")
    plt.title(Title)
    plt.xlabel("Features_X1")
    plt.ylabel("Features_X2")
    plt.show()

# -------------------------------------------------------------
# 11) Main code section
# -------------------------------------------------------------
def Main_Function():
    # Data generation:
    Features_Inputs_Vector_X, Labels_Outputs_Vector_Y = Make_Two_Moons(Number_Of_Samples=800, Noise=0.10, Seed=1, Vertical_Shift=0.3)
    
    # Split the date for training and testing:
    Features_Inputs_Vector_X_Train, Labels_Outputs_Vector_Y_Train, Features_Inputs_Vector_X_Test, Labels_Outputs_Vector_Y_Test = Train_Test_Split(Features_Inputs_Vector_X, Labels_Outputs_Vector_Y, Test_Ratio=0.2)

    # Plot the points of the dataset:
    Plot_Dataset_Function(Features_Inputs_Vector_X, Labels_Outputs_Vector_Y, Title="Two Moons (raw)")
    
    # Parameters of the Neural Network initialization:
    Parameters_Bias_Weights = Parameters_Bias_Weight_Initialization(Seed=1, Hidden_Number=16, Scale=0.5)
    
    # Training of the Neural Network:
    Train_Neural_Network_Function(Parameters_Bias_Weights, Features_Inputs_Vector_X_Train, Labels_Outputs_Vector_Y_Train, Features_Inputs_Vector_X_Test, Labels_Outputs_Vector_Y_Test, Learning_Rate=0.15, Epochs=800, Print_Every=100)

    Plot_Decision_Boundary(Parameters_Bias_Weights, Features_Inputs_Vector_X, Labels_Outputs_Vector_Y, Title="Decision boundary (2-16-16-1)")


if __name__ == "__main__":
    Main_Function()










