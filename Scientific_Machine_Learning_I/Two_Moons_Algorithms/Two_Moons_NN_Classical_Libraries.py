import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# -------------------------------------------------------------------
# 1) Dataset generation
# -------------------------------------------------------------------
def Make_Two_Moons_Dataset(Number_Of_Samples=1500,
                           Noise=0.10,
                           Seed=1,
                           Vertical_Shift=0.1):
    
    Range = np.random.default_rng(Seed)

    Number_Half = Number_Of_Samples // 2
    Part_1 = Range.random(Number_Half)*np.pi
    Part_2 = Range.random(Number_Half)*np.pi

    # Moon number 1 upper:
    Features_Inputs_Vector_X1 = np.c_[np.cos(Part_1),np.sin(Part_1)]
    Labels_Outputs_Vector_Y1 = np.ones(Number_Half)

    # Moon number 2 lower:
    Features_Inputs_Vector_X2 = np.c_[1.0 - np.cos(Part_2),-np.sin(Part_2) + Vertical_Shift]
    Labels_Outputs_Vector_Y2 = np.zeros(Number_Half)

    Features_Inputs_Vector_X = np.vstack([Features_Inputs_Vector_X1,Features_Inputs_Vector_X2])
    Labels_Outputs_Vector_Y = np.concatenate([Labels_Outputs_Vector_Y1,Labels_Outputs_Vector_Y2])

    # Gaussian noise:
    Features_Inputs_Vector_X += Range.normal(0.0,Noise,size=Features_Inputs_Vector_X.shape)

    # Shuffle:
    Indexes = Range.permutation(len(Features_Inputs_Vector_X))
    return Features_Inputs_Vector_X[Indexes],Labels_Outputs_Vector_Y[Indexes]

# -------------------------------------------------------------------
# 2) Model of the Neural Network
# -------------------------------------------------------------------
class Two_Moons_MLP_NN(nn.Module):
    # The role of the init function is to build the object whenever called. For example, when calling this object through:
    # Model = Two_Moons_MLP_NN(Hidden_Width=16), the init function will create the object first. In our case, it will:
    # * Create the layers.
    # * Create the weights.
    # * Create the bias.
    # Then store everything in the object.
    def __init__(self,Hidden_Width=16):
        # Note that according to the adopted syntax through Two_Moons_MLP_NN(nn.Module), the class Two_Moons_MLP_NN inherits
        # of nn.Module <=> Two_Moons_MLP_NN is seen as a particular case of nn.Module. Here, the parent is nn.Module, which
        # means that super().__init__() is in fact, nn.Module.__init__(self). This is essential because nn.Module handles a 
        # lot of actions internally; for example, it prepares the parameters list, how to handle the gradients, Model.train(),
        # Model.eval(), Model.parameters()..etc.
        super().__init__()
        # Use of nn.Sequential allows to express that one wants the following workflow:
        # Vector_Current = Linear_Layer_1(Vector_Current)
        # Vector_Current = Tanh_Layer(Vector_Current)
        # Vector_Current = Linear_Layer_2(Vector_Current)
        # Vector_Current = Tanh_Layer(Vector_Current)
        # Vector_Current = Linear_Layer_3(Vector_Current).
        # In addition, one can specify the input and output dimensions in the arguments of nn.Linear.
        # Mathematically, it is the following relation: Vector_Output = Weights*Vector_Input + Bias, with:
        # Vector_Input in R² and Vector_Output in R¹⁶ (here Hidden_Width = 16).
        self.Network = nn.Sequential(nn.Linear(2,Hidden_Width),
                                     nn.Tanh(),
                                     nn.Linear(Hidden_Width,Hidden_Width),
                                     nn.Tanh(),
                                     nn.Linear(Hidden_Width,1))

    def forward(self,Input_Point):
        return self.Network(Input_Point)

# -------------------------------------------------------------------
# 3) Evaluation function used for the validation
# -------------------------------------------------------------------
def Compute_Accuracy(Model,Features_Inputs_Vector_X,Labels_Outputs_Vector_Y):
    
    # Force the network to be in evaluation mode. It allows to force Model.training = False. This is important to be precised for Pytorch,
    # because some layers might behave differently wether the mode is in training or not.
    Model.eval()
    # The following line states that the Network will be used, but not the gradients. It means that it does the forward pass,
    # but does not store any information for the future backpropagation. The Autograd is deactivated. It is only a forward pass.
    with torch.no_grad():
        # Compute the last output before the last sigmoid layer:
        Logits = Model(Features_Inputs_Vector_X)
        
        # Conversion to probability:
        Probabilities = torch.sigmoid(Logits)
        
        # Prediction of the belonging:
        Predictions = (Probabilities >= 0.5).float()
        
        # Accuracy measure coparison with the labels
        Accuracy = (Predictions == Labels_Outputs_Vector_Y).float().mean().item()

    return Accuracy


# -------------------------------------------------------------------
# 4) Decision boundary plots for analysis and conclusion
# -------------------------------------------------------------------
def Plot_Decision_Boundary(Model,Features_Inputs_Vector_X,Labels_Outputs_Vector_Y):
    
    # Model mode:
    Model.eval()

    # Minimal values determination for the abscissa and ordinates:
    Features_Inputs_Vector_X_Min, Features_Inputs_Vector_X_Max = Features_Inputs_Vector_X[:, 0].min() - 0.5, Features_Inputs_Vector_X[:, 0].max() + 0.5
    Features_Inputs_Vector_Y_Min, Features_Inputs_Vector_Y_Max = Features_Inputs_Vector_X[:, 1].min() - 0.5, Features_Inputs_Vector_X[:, 1].max() + 0.5

    # Mesh grid creation:
    Mesh_Grid_Points_Abscissa,\
    Mesh_Grid_Points_Ordinates = np.meshgrid(np.linspace(Features_Inputs_Vector_X_Min,Features_Inputs_Vector_X_Max, 300),
                                             np.linspace(Features_Inputs_Vector_Y_Min,Features_Inputs_Vector_Y_Max, 300))

    # Grid points concatenation:
    Grid_Points = np.c_[Mesh_Grid_Points_Abscissa.ravel(),Mesh_Grid_Points_Ordinates.ravel()]
    
    # Testing on the grid points:
    Grid_Torch = torch.tensor(Grid_Points,dtype=torch.float32)
    with torch.no_grad():
        Logits = Model(Grid_Torch)
        Probabilities = torch.sigmoid(Logits).numpy().reshape(Mesh_Grid_Points_Abscissa.shape)

    plt.figure(figsize=(7, 5))
    plt.contourf(Mesh_Grid_Points_Abscissa,Mesh_Grid_Points_Ordinates,Probabilities,levels=50,alpha=0.7)
    plt.contour(Mesh_Grid_Points_Abscissa,Mesh_Grid_Points_Ordinates,Probabilities,levels=[0.5],colors="black")
    plt.scatter(Features_Inputs_Vector_X[:, 0],Features_Inputs_Vector_X[:, 1],c=Labels_Outputs_Vector_Y,cmap="bwr",edgecolor="k",s=20)
    plt.axis("equal")
    plt.title("Decision Boundary")
    plt.xlabel("Dimension points 1")
    plt.ylabel("Dimension points 2")
    plt.show()

# -------------------------------------------------------------------
# 5) Main training script
# -------------------------------------------------------------------
def Main_Function():
    
    # Since the code contains a lot of randomness, possible to fix the seeds to have the same accuracy/probabilistic data at the end
    # for each loops.
    # Fixing the state of Pytorch's random generator:
    torch.manual_seed(1)
    # Fixing the state of Numpy's random generator:
    np.random.seed(1)

    # Dataset creation:
    Features_Inputs_Vector_X,Labels_Outputs_Vector_Y = Make_Two_Moons_Dataset(Number_Of_Samples=1500,
                                                                              Noise=0.10,
                                                                              Seed=1,
                                                                              Vertical_Shift=0.1)

    # Training / Testint data separation:
    Train_Ratio = 0.6
    Split_Indexes = int(Train_Ratio*len(Features_Inputs_Vector_X))

    # Training data:
    Features_Inputs_Vector_Training_X = Features_Inputs_Vector_X[:Split_Indexes]
    Labels_Outputs_Vector_Training_Y = Labels_Outputs_Vector_Y[:Split_Indexes]

    # Testing data:
    Features_Inputs_Vector_Testing_X = Features_Inputs_Vector_X[Split_Indexes:]
    Labels_Outputs_Vector_Testing_Y = Labels_Outputs_Vector_Y[Split_Indexes:]

    # Convert to torch tensors: Allows to convert a Numpy array to Pytorch usable format. The commands "view(-1,1)" allows to 
    # impose the dimension to be of (Length_Vector,) in (Length_Vector,1).
    # Training data:
    Features_Inputs_Vector_Training_Pytorch_X = torch.tensor(Features_Inputs_Vector_Training_X,dtype=torch.float32)
    Labels_Outputs_Vector_Training_Pytorch_Y = torch.tensor(Labels_Outputs_Vector_Training_Y,dtype=torch.float32).view(-1,1)

    # Testing data:
    Features_Inputs_Vector_Testing_Pytorch_X = torch.tensor(Features_Inputs_Vector_Testing_X, dtype=torch.float32)
    Labels_Outputs_Vector_Testing_Pytorch_Y = torch.tensor(Labels_Outputs_Vector_Testing_Y, dtype=torch.float32).view(-1,1)

    # Data loader:
    # To regroup the training data. The object Train_Dataset is of the form (x,y) = (Features_Inputs_X,Labels_Outputs_Y) for all samples.
    # For example: Train_Dataset[0] = (Features_Inputs_Vector_Training_Pytorch_X[0],Labels_Outputs_Vector_Training_Pytorch_Y[0]).
    Train_Dataset = TensorDataset(Features_Inputs_Vector_Training_Pytorch_X,Labels_Outputs_Vector_Training_Pytorch_Y)
    
    # The DataLoader allows to simultaneously use batches for the training. Instead of a training loop with one single case as:
    # for index_Epoch in range(Number_Epochs): 
    #   Features_Inputs_X_Current = Features_Inputs_Vector_Training_Pytorch_X[index_Epoch]
    #   Labels_Outputs_Y = Labels_Outputs_Vector_Training_Pytorch_Y[index_Epoch]
    # Btch size fixed to treat several examples at the time. The shuffle allows to introduce some randomness, acts in the sense of the 
    # SGD.
    Train_Loader = DataLoader(Train_Dataset,
                              batch_size=32,
                              shuffle=True)

    # Model of Neural Network:
    Model = Two_Moons_MLP_NN(Hidden_Width=16)

    # Loss function choice:
    Loss_Function = nn.BCEWithLogitsLoss()
    
    # Optimizer choice:
    Optimizer = torch.optim.SGD(Model.parameters(),lr=0.1)

    # Training loop:
    Number_Epochs = 600
    for index_Epoch in range(1,Number_Epochs + 1):
        
        # Activating the training mode of the NN MLP model:
        Model.train()
        
        # Current Loss_Function at the current epoch initial value:
        Epoch_Loss = 0.0

        # Loop over the batches:
        for Features_Inputs_Vector_Batch_Training,Labels_Outputs_Vector_Batch_Training in Train_Loader:
            
            # Forward pass calling for the current batch. The output will be of the shape (32,1).
            Logits_Penultimate_Layer = Model(Features_Inputs_Vector_Batch_Training)
            
            # Loss function calculation on the overall batch:
            Loss = Loss_Function(Logits_Penultimate_Layer,Labels_Outputs_Vector_Batch_Training)

            # Zero setlling of the previous weights gradients:
            Optimizer.zero_grad()
            
            # Backpropagation calculation and storage of the gradients of the weights in the parameters; Parameter.grad.
            Loss.backward()
            
            # Use of the values of the gradients by the optimizer. 
            # This is the operation of Weigts <- Weights - Step*derivative_Loss_With_Respect_To_Weights for SGD.
            # This is the line that modified the weights of the NN MLP.
            Optimizer.step()
            
            # Calculation of the current Loss_Function of the current epoch. Based on the fact that the Pytorch Loss_Function calculation is:
            # Loss = (1/Size_Batch)*Sum(losses); so to get Sum(losses) => need to do Loss*Size_Batch.
            Epoch_Loss += Loss.item()*Features_Inputs_Vector_Batch_Training.shape[0]
            
        # Regularization of the total loss of the epochs:
        Epoch_Loss /= len(Train_Loader.dataset)

        if index_Epoch % 50 == 0 or index_Epoch == 1:
            Training_Accuracy = Compute_Accuracy(Model,
                                                 Features_Inputs_Vector_Training_Pytorch_X,
                                                 Labels_Outputs_Vector_Training_Pytorch_Y)
            Testing_Accuracy = Compute_Accuracy(Model,
                                                Features_Inputs_Vector_Testing_Pytorch_X,
                                                Labels_Outputs_Vector_Testing_Pytorch_Y)

            print(f"Epoch {index_Epoch:4d} | "
                  f"Loss: {Epoch_Loss:.4f} | "
                  f"Training accuracy: {Training_Accuracy:.4f} | "
                  f"Testing accuracy: {Testing_Accuracy:.4f}")

    # Final evaluation:
    Test_Accuracy = Compute_Accuracy(Model,
                                     Features_Inputs_Vector_Testing_Pytorch_X, 
                                     Labels_Outputs_Vector_Testing_Pytorch_Y)
    print(f"\nFinal test accuracy: {Test_Accuracy:.4f}")

    # Plot decision boundary:
    Plot_Decision_Boundary(Model, 
                           Features_Inputs_Vector_Testing_X, 
                           Labels_Outputs_Vector_Testing_Y)


if __name__ == "__main__":
    Main_Function()