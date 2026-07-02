# Used librairies in this project:
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as Functionnal
from torch.utils.data import TensorDataset, DataLoader
import math

# This algorithm gathers the functions that will be used to design a FNO to solve
# Burger's PDE equation.

# -------------------------------------------------------------------------------------------------------------
# I) Dataset generation
# -------------------------------------------------------------------------------------------------------------

# In this section, a solver will be designed in order to generate the ground truth data used 
# to train ou supervised PINO learning problem afterward.
# For that, various set of functions will be created.

# ---------------------------------------------------------------------------------------------
# 1) PDE separation
# ---------------------------------------------------------------------------------------------

# Function to retreive One dimensional partial spatial derivative of a function:
def Spectral_Derivative_One_Dimension(Function_Current_Point_One_Dimension,Mode_Current_k):
    
    # FFT of the current function:
    Function_Frequency_Fourier_Domain = np.fft.fft(Function_Current_Point_One_Dimension)
    
    # Spectral derivative of the function in Fourier domain:
    Function_Derivative_In_Spectral_Fourier_Domain = (1j*2.0*np.pi*Mode_Current_k)*Function_Frequency_Fourier_Domain
    
    # Inverse Fourier transform to retreive the partial derivative:
    Partial_Derivative_Of_Function_With_Respect_To_Dimension = np.fft.ifft(Function_Derivative_In_Spectral_Fourier_Domain).real
    
    # Output of the function:
    return Partial_Derivative_Of_Function_With_Respect_To_Dimension


# PDE devided into two PDEs suing the strang splitting split-step method. Creation of diffusion
# and convection parts.
# Diffusion part of the solver giving the field after diffusion:
def Diffusion_Split_Spectral_PDE(Function_Current_Point_One_Dimension,Mode_Current_k,Delta_Time_Tau,Viscous_Coefficient):
    
    # FFT of the current function:
    Function_Frequency_Fourier_Domain_Current = np.fft.fft(Function_Current_Point_One_Dimension)
    
    # Multiplicative coefficient:
    # Laplace multiplier:
    Laplace_Multiplier = -(2*np.pi*Mode_Current_k)**2
    Heat_Factor = np.exp(Viscous_Coefficient*Laplace_Multiplier*Delta_Time_Tau)
    
    # Function in Fourier domaine at current mode k at current step plus tau delta time:
    Function_Frequency_Fourier_Domain_Next = Heat_Factor*Function_Frequency_Fourier_Domain_Current
    
    # Back to the physical domain:
    Function_Solution_Next_Step_Diffusive_Linear_Part_PDE = np.fft.ifft(Function_Frequency_Fourier_Domain_Next).real
    
    # Output of the function:
    return Function_Solution_Next_Step_Diffusive_Linear_Part_PDE
    

# Convection part of the solver giving the field after convection. Based on a non-linear equation formulation:
def Convection_Split_Spectral_PDE(Delta_Time_Tau,Function_Current_Point_One_Dimension,Mode_Current_k):

    # Intermediaire non-linear variable calculation:
    Function_Intermediate_Current = 0.5*Function_Current_Point_One_Dimension**2
    
    # Derivative of this function with respect to the current dimension:
    Partial_Derivative_Of_Function_With_Respect_To_Dimension = Spectral_Derivative_One_Dimension(Function_Intermediate_Current,Mode_Current_k)
    
    # Then Euler first order propagation:
    Function_Next_Point_One_Dimension = Function_Current_Point_One_Dimension - Delta_Time_Tau*Partial_Derivative_Of_Function_With_Respect_To_Dimension
    
    # Output of the function:
    return Function_Next_Point_One_Dimension



# ---------------------------------------------------------------------------------------------
# 2) Solver Burger's Equation Strang Splitting method for PDE
# ---------------------------------------------------------------------------------------------

# Burger's equation solver in 1 dimension for a periodic function.
# Strang splitting method:
# - Diffusion step with time step over 2.
# - Convection step.
# - Diffusion step with time step over 2.

# Hereafter resolution:
# - Spatial variable on the interval [0,1[.
# - Temporal variable on the interval [0,T].
# - Periodicity T.

def Burgers_Solver_One_Dimension(Initial_Function_Solution,Viscous_Coefficient = 0.1,Delta_Time_Tau = 1e-4,Time_Periodicity = 1):
    
    # Save the initial value of the function:
    Function_Solution_Current = Initial_Function_Solution.copy()
    
    # Grid length for the resolution:
    Number_Length_Grid = Function_Solution_Current.shape[0]
    
    # Wave number construction in function of the N value of the grid:
    Wave_Number_Modes = np.fft.fftfreq(Number_Length_Grid, d=1.0/Number_Length_Grid)
    
    # Number of step time:
    Number_Step_Times = int(np.round(Time_Periodicity/Delta_Time_Tau))
    
    # Time periodicity limitation verification:
    Time_Step_Current = Number_Step_Times*Delta_Time_Tau
    if abs(Time_Step_Current - Time_Periodicity) > 1e-12:
        raise ValueError("Choose Time_Periodicity as a multiple of Delta_Time_Tau for this simple version of the solver.")
        
    # Temporary half-delta-time variable declaration:
    Delta_Time_Tau_Half = 0.5*Delta_Time_Tau
    
    # Successions of diffusion and convection steps:
    for _ in range(Number_Step_Times):
        
        # Diffusion half step:
        Function_Solution_Current = Diffusion_Split_Spectral_PDE(Function_Solution_Current,Wave_Number_Modes,Delta_Time_Tau_Half,Viscous_Coefficient)
        
        # Convection step entire:
        Function_Solution_Current = Convection_Split_Spectral_PDE(Delta_Time_Tau,Function_Solution_Current,Wave_Number_Modes)
        
        # Diffusion half step:
        Function_Solution_Current = Diffusion_Split_Spectral_PDE(Function_Solution_Current,Wave_Number_Modes,Delta_Time_Tau_Half,Viscous_Coefficient)
        
    # Then output of the function:
    return Function_Solution_Current

# ---------------------------------------------------------------------------------------------
# 3) Initial condition generation
# ---------------------------------------------------------------------------------------------

# The goal for the initial condition is that it must be:
# - Periodical.
# - Smooth.
# - Random.

# A way to control the regularity (smoothness), build the Fourier coefficients of the function
# then use the IFFT; this will allow to control the spectrum hence the regularity.

def Sample_Smooth_Periodic_Initial_Conditions(Number_Points,Coefficient_Alpha = 2.0,Time_Constant = 7.0,Amplitude = 1.0,Seed = None):
    
    # Random generator:
    Random_Generator = np.random.default_rng(Seed)
    
    # Use of positive frequencies only:
    Wave_Number_Positives = np.fft.rfftfreq(Number_Points, d=1.0/Number_Points)
    
    # Aimed spectrum shaping:
    # goal is to favorize the low frequency to ensure smoothness:
    Spectrum_Designed = Amplitude/(1.0 + (Wave_Number_Positives/Time_Constant)**2)**(Coefficient_Alpha/2.0)
    
    # Random draw of the coefficients:
    Real_Part = Random_Generator.standard_normal(len(Wave_Number_Positives))
    Imaginary_Part = Random_Generator.standard_normal(len(Wave_Number_Positives))
    
    # Coefficients creation:
    Coefficient_Spectrum = (Real_Part + 1j*Imaginary_Part)*Spectrum_Designed
    
    # But carefull, the initial zero mode shall be physical: it is the space mean:
    Coefficient_Spectrum[0] = Coefficient_Spectrum[0].real + 0j
    
    # If the Number of points is even, the coefficient is real:
    # This is the special Nyquist mode:
    if Number_Points % 2 == 0:
        Coefficient_Spectrum[-1] = Coefficient_Spectrum[-1].real + 0j
        
    # Physical reconstruction:
    Function_Initial_Point_Generated = np.fft.irfft(Coefficient_Spectrum, n=Number_Points)

    # To avoid constant offset, centering with null mean:
    Function_Initial_Point_Generated = Function_Initial_Point_Generated - np.mean(Function_Initial_Point_Generated)
    
    return Function_Initial_Point_Generated

# Alternative initial function conditions create to be aligned with the papers:
def Initial_Condition_Function_Burger(Number_Modes_Maximal_Function_Burger,\
                                      Number_Grid_Points,
                                      Seed):
    
    # Initial parameters:
    Grid_Points = []
    Initial_Function_Burger = []
    Random_Generator = np.random.default_rng(Seed)
    
    # Mean and variance for the zero mode:
    Mean_Scaling_Mode_Zero = 0
    Standard_Deviation_Scaling_Mode_Zero = 1
    # Mean and variance for the cosine eigenfunction:
    Mean_Scaling_Eigenfunction_Cosine = 0
    Standard_Deviation_Scaling_Eigenfunction_Cosine = 1
    # Mean and variance for the sine eigenfunction:
    Mean_Scaling_Eigenfunction_Sine = 0
    Standard_Deviation_Scaling_Eigenfunction_Sine = 1
    
    # Associated random Gaussian coefficients:
    Scaling_Coefficient_Mode_Zero = Random_Generator.normal(Mean_Scaling_Mode_Zero,Standard_Deviation_Scaling_Mode_Zero) # Scalar.
    Scaling_Coefficient_Eigenfunction_Cosine = Random_Generator.normal(Mean_Scaling_Eigenfunction_Cosine,Standard_Deviation_Scaling_Eigenfunction_Cosine,Number_Modes_Maximal_Function_Burger)
    Scaling_Coefficient_Eigenfunction_Sine = Random_Generator.normal(Mean_Scaling_Eigenfunction_Sine,Standard_Deviation_Scaling_Eigenfunction_Sine,Number_Modes_Maximal_Function_Burger)
    
    # Loop on all the modes:
    for index_Grid_Points in range(Number_Grid_Points):
        
        # Define the current grid point:
        Current_Grid_Point = index_Grid_Points/Number_Grid_Points
        Grid_Points.append(Current_Grid_Point)
        
        # Retreive the value of the current initial function:
        Function_Initial_Burger_Output_Current_Point = Function_Initial_Burger_Calculation(Number_Modes_Maximal_Function_Burger,\
                                                                                           Current_Grid_Point,\
                                                                                           Scaling_Coefficient_Mode_Zero,\
                                                                                           Scaling_Coefficient_Eigenfunction_Cosine,\
                                                                                           Scaling_Coefficient_Eigenfunction_Sine)
        # Storage of the initial function value:
        Initial_Function_Burger.append(Function_Initial_Burger_Output_Current_Point)
        
    # Output of the function:
    return np.array(Grid_Points), np.array(Initial_Function_Burger)
             
    
# Initial function creation function:
def Function_Initial_Burger_Calculation(Number_Modes_Maximal_Function_Burger,\
                                        Current_Grid_Point,\
                                        Scaling_Coefficient_Mode_Zero,\
                                        Scaling_Coefficient_Eigenfunction_Cosine,\
                                        Scaling_Coefficient_Eigenfunction_Sine):
    
    # Initialization:
    Eigenvalue_Initial_Zero_Mode = 625/((25)**2)
    Function_Initial_Burger_Output = math.sqrt(Eigenvalue_Initial_Zero_Mode)*Scaling_Coefficient_Mode_Zero
    
    # Loop on all the possible modes:
    for index_Modes in range(1,Number_Modes_Maximal_Function_Burger + 1):
        
        # Current eigenvalue of the initial operator Burger:
        Eigenvalue_Initial_Operator = (2*np.pi*index_Modes)**2 # Lambda_k in the theory of NO.
        
        # Current eigenvalue of the current operator covariance:
        Eigenvalue_Covariance_Operator = 625/(Eigenvalue_Initial_Operator + 25)**2
        
        # Eigenfunction of the cosine mode:
        Eigenfunction_Cosine_Mode_Current = math.sqrt(Eigenvalue_Covariance_Operator)*Scaling_Coefficient_Eigenfunction_Cosine[index_Modes - 1]*\
                                            math.sqrt(2)*np.cos(2*np.pi*index_Modes*Current_Grid_Point)
                                            
        # Eigenfunction of the sine mode:
        Eigenfunction_Sine_Mode_Current = math.sqrt(Eigenvalue_Covariance_Operator)*Scaling_Coefficient_Eigenfunction_Sine[index_Modes - 1]*\
                                          math.sqrt(2)*np.sin(2*np.pi*index_Modes*Current_Grid_Point)
                                            
        # Total superimposed eigenfunction:
        Function_Initial_Burger_Output = Function_Initial_Burger_Output + Eigenfunction_Cosine_Mode_Current + Eigenfunction_Sine_Mode_Current
    
    # Output of the function:
    return Function_Initial_Burger_Output
                                            

# ---------------------------------------------------------------------------------------------
# 4) Generation of a complete dataset
# ---------------------------------------------------------------------------------------------

# Creation of a general function to generate an entire associated dataset:
def Generate_Burgers_PDE_Dataset(Number_Samples,Number_Points=256,Viscous_Coefficient=0.1,
                                 Time_Periodicity=1.0,Delta_Time_Tau=1e-4,Alpha_Coefficient=2.0,
                                 Time_Constant_Spectrum=7.0,Amplitude_Spectrum=1.0,Seed=1234,
                                 Is_Article_Method_Initial_Conditions=0,Number_Modes_Maximal_Function_Burger=32):
    
    # Generates a dataset of pairs (u0,u(.,Time_Periodicity)) for the 1D Burger's equation.
    # Random generator:
    Random_Generator = np.random.default_rng(Seed)
    
    # Spatial points decomposition:
    One_Dimension_Space_Grid_X = np.linspace(0.0, 1.0,Number_Points,endpoint=False)

    # Inputs and Outputs elements storage initialization:
    Inputs_Function = np.zeros((Number_Samples, Number_Points),dtype=np.float64)
    Outputs_Function = np.zeros((Number_Samples, Number_Points),dtype=np.float64)

    # Ground truth generation by PDE Burger's solving:
    for index_Number_Samples in range(Number_Samples):
        
        # Seed generation:
        Sample_Seed = Random_Generator.integers(0, 10**9)

        # Initial condition generation:
        if Is_Article_Method_Initial_Conditions != 1:
            Function_Initial_Condition_Current_Sample = Sample_Smooth_Periodic_Initial_Conditions(Number_Points,
                                                                                                  Coefficient_Alpha = Alpha_Coefficient,
                                                                                                  Time_Constant = Time_Constant_Spectrum,
                                                                                                  Amplitude = Amplitude_Spectrum,
                                                                                                  Seed = int(Sample_Seed))
        else:
            _, Function_Initial_Condition_Current_Sample = Initial_Condition_Function_Burger(Number_Modes_Maximal_Function_Burger,\
                                                                                             Number_Points,
                                                                                             Seed=int(Sample_Seed))
                
        
        # Burger's equation:
        Function_Current_Sample_Output_Solver_Solved = Burgers_Solver_One_Dimension(Function_Initial_Condition_Current_Sample,
                                                                                    Viscous_Coefficient = Viscous_Coefficient,
                                                                                    Delta_Time_Tau = Delta_Time_Tau,
                                                                                    Time_Periodicity = Time_Periodicity)

        # Functions storing results to form the supervised learning dataset:
        Inputs_Function[index_Number_Samples] = Function_Initial_Condition_Current_Sample
        Outputs_Function[index_Number_Samples] = Function_Current_Sample_Output_Solver_Solved

        if (index_Number_Samples + 1) % 10 == 0 or index_Number_Samples == Number_Samples - 1:
            print(f"Generated {index_Number_Samples + 1}/{Number_Samples} samples")

    return One_Dimension_Space_Grid_X,Inputs_Function,Outputs_Function

# -------------------------------------------------------------------------------------------------------------
# II) FNO construction, training & testing
# -------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------
# 1) Objects and classes creation
# ---------------------------------------------------------------------------------------------
# /!\ Remember than inside classes and trainable object, not possible to use other libraries than torch. Numpy prohibided. /!\
# Fourier layer step:
# -------------------
class Fourier_Layer(nn.Module):
    def __init__(self,Dimension_Width=64,Mode_Maximal=32):
        super().__init__()
        self.Dimension_Width = Dimension_Width
        self.Mode_Maximal = Mode_Maximal
        self.Fourier_Transform_Matrix_R_Step = nn.Parameter(torch.randn(Mode_Maximal,Dimension_Width,Dimension_Width,dtype=torch.cfloat))
        self.Weight_Matrix = nn.Linear(Dimension_Width,Dimension_Width)
        self.Sigma_Function_Activation = nn.ReLU()

    def forward(self,Lifted_Input):
        # Calculation of F^-1(R.(Fv)) step:
        # ---------------------------------
        Number_Grid_Points = Lifted_Input.shape[1] # Shall be 256.
        FFT_Lifted_Inputs = torch.fft.fft(Lifted_Input,dim=1) # Directly in the shape of C^1,256,64. Note that dim = 1 since the 
        # goal is to make the FFT on the variables denoted by the dimension 1 of the Lifted_Input. Since Lifted_Input is of shape of
        # R^Batch_Size,Number_Grid,Dimension_Width = R^1,256,64, dim = 1 corresponds to the 256 hence the grid points => spatial
        # variable x.
        FFT_Lifted_Inputs_Truncated = FFT_Lifted_Inputs[:,:self.Mode_Maximal,:] # In the shape of C^1,Mode_Maximal,64
        Transformed_Matrix = torch.einsum("bki,kio->bko",FFT_Lifted_Inputs_Truncated,self.Fourier_Transform_Matrix_R_Step)
        # Full matrix creation:
        Transformed_Matrix_Full = torch.zeros(Lifted_Input.shape[0],
                                              Number_Grid_Points,
                                              self.Dimension_Width,
                                              dtype=torch.cfloat,
                                              device=Lifted_Input.device)
        Transformed_Matrix_Full[:,:self.Mode_Maximal,:] = Transformed_Matrix
        # Spectral/Non-frequential output:
        Spectral_Output = torch.fft.ifft(Transformed_Matrix_Full,dim=1).real
        
        # Final step of adding the weight matrix and activation function:
        # ---------------------------------------------------------------
        Local_Output = self.Weight_Matrix(Lifted_Input)
        Output_Layer = self.Sigma_Function_Activation(Local_Output + Spectral_Output)
        return Output_Layer

# Lifting operation step:
# -----------------------
class Lifting_Operation(nn.Module):
    def __init__(self,Dimension_Width=64):
        super().__init__()
        self.Dimension_Width = Dimension_Width
        self.Lifting_Matrix = nn.Linear(1,Dimension_Width)
        
    def forward(self,Input_Point_Discretized_Array):
        # The Input_Point_Discretized_Array is supposed to belong in R^256,1 and therefore be an array of 256 initial condition
        # functions for Burger's equation. Hence u0(x) = (u0(x1) u0(x2) ... u0(xJ))^T, with J = 256.
        return self.Lifting_Matrix(Input_Point_Discretized_Array)


# Projection operation step:
# --------------------------
class Projection_Operation(nn.Module):
    def __init__(self,Dimension_Width=64,Dimension_Expressiveness=128):
        super().__init__()
        self.Dimension_Width = Dimension_Width
        # self.Projection_Matrix = nn.Linear(Dimension_Width,1)
        self.Projection_Matrix = nn.Sequential(nn.Linear(Dimension_Width,Dimension_Expressiveness),
                                               nn.GELU(),
                                               nn.Linear(Dimension_Expressiveness,1))
        
    def forward(self,Input_Vector_Matrix):
        # The input is supposed to be of the shape R^256,64.
        return self.Projection_Matrix(Input_Vector_Matrix)
    
# Agregation of the classes:
# --------------------------
class Fourier_Neural_Operator_Overall(nn.Module):
    def __init__(self,Dimension_Width=64,Mode_Maximal=32,Dimension_Expressiveness=128):
        super().__init__()
        # Global FNO:
        self.FNO_Network = nn.Sequential(Lifting_Operation(Dimension_Width=64),
                                         Fourier_Layer(Dimension_Width=64,Mode_Maximal=32),
                                         Fourier_Layer(Dimension_Width=64,Mode_Maximal=32),
                                         Fourier_Layer(Dimension_Width=64,Mode_Maximal=32),
                                         Fourier_Layer(Dimension_Width=64,Mode_Maximal=32),
                                         Projection_Operation(Dimension_Width=64,Dimension_Expressiveness=128))
        
    def forward(self,Input_Point):
        Input_Point = Input_Point.unsqueeze(-1)
        Output = self.FNO_Network(Input_Point)
        return Output.squeeze(-1)
    
# ---------------------------------------------------------------------------------------------
# 2) Various function declaration
# ---------------------------------------------------------------------------------------------
# L2 norm error computation:
def Compute_Relative_L2_Error(Model,Inputs_Function_X,Outputs_Function_Y):
    # Mode of the model:
    Model.eval()
    with torch.no_grad():
        # Compute the last output before the last sigmoid layer:
        Output_Predicted = Model(Inputs_Function_X)
        # Relative error computation:
        Relative_Error = (torch.linalg.norm(Output_Predicted - Outputs_Function_Y)/torch.linalg.norm(Outputs_Function_Y))

    return Relative_Error.item()

# ---------------------------------------------------------------------------------------------
# 3) Main training script
# ---------------------------------------------------------------------------------------------
def Main_Function():
    
    # Since the code contains a lot of randomness, possible to fix the seeds to have the same accuracy/probabilistic data at the end
    # for each loops.
    # Fixing the state of Pytorch's random generator:
    torch.manual_seed(1)
    # Fixing the state of Numpy's random generator:
    np.random.seed(1)
    
    # Storage of the Loss:
    Loss_Storage = []
    Relative_L2_Testing_Storage = []
    
    # Parameters:
    Number_Points = 256
    Number_Grid_Points = Number_Points
    Viscous_Coefficient = 0.01
    Time_Periodicity = 0.2
    Delta_Time_Tau = 1e-4
    Number_Modes_Maximal_Function_Burger = 32
    Is_Article_Method_Initial_Conditions = 1

    # Dataset creation:
    # 1000 sample points for training and 200 for testing:
    One_Dimension_Space_Grid_X_Training,\
    Inputs_Function_Training,\
    Outputs_Function_Training = Generate_Burgers_PDE_Dataset(Number_Samples=1000,
                                                             Number_Points=Number_Points,
                                                             Viscous_Coefficient=Viscous_Coefficient,
                                                             Time_Periodicity=Time_Periodicity,
                                                             Delta_Time_Tau=Delta_Time_Tau,
                                                             Alpha_Coefficient=2.5,
                                                             Time_Constant_Spectrum=6.0,
                                                             Amplitude_Spectrum=8.0,
                                                             Seed=42,
                                                             Is_Article_Method_Initial_Conditions=Is_Article_Method_Initial_Conditions,
                                                             Number_Modes_Maximal_Function_Burger=Number_Modes_Maximal_Function_Burger)
    
    # Testing data generation: Maybe to be tested to generate 1200 samples then devide them into 1000 training and 200 testing.
    One_Dimension_Space_Grid_X_Testing,\
    Inputs_Function_Testing,\
    Outputs_Function_Testing = Generate_Burgers_PDE_Dataset(Number_Samples=200,
                                                            Number_Points=Number_Points,
                                                            Viscous_Coefficient=Viscous_Coefficient,
                                                            Time_Periodicity=Time_Periodicity,
                                                            Delta_Time_Tau=Delta_Time_Tau,
                                                            Alpha_Coefficient=2.5,
                                                            Time_Constant_Spectrum=6.0,
                                                            Amplitude_Spectrum=8.0,
                                                            Seed=43,
                                                            Is_Article_Method_Initial_Conditions=Is_Article_Method_Initial_Conditions,
                                                            Number_Modes_Maximal_Function_Burger=Number_Modes_Maximal_Function_Burger)
    
    # Convert to torch tensors: Allows to convert a Numpy array to Pytorch usable format. The commands "view(-1,1)" allows to 
    # impose the dimension to be of (Length_Vector,) in (Length_Vector,1).
    # Training data:
    Inputs_Function_Training_Pytorch_X = torch.tensor(Inputs_Function_Training,dtype=torch.float32)
    Outputs_Function_Training_Pytorch_Y = torch.tensor(Outputs_Function_Training,dtype=torch.float32)
    
    # Testing data:
    Inputs_Function_Testing_Pytorch_X = torch.tensor(Inputs_Function_Testing,dtype=torch.float32)
    Outputs_Function_Testing_Pytorch_Y = torch.tensor(Outputs_Function_Testing,dtype=torch.float32)
    
    # Data loader:
    # To regroup the training data. The object Train_Dataset is of the form (x,y) = (Features_Inputs_X,Labels_Outputs_Y) for all samples.
    # For example: Train_Dataset[0] = (Inputs_Function_Training_Pytorch_X[0],Outputs_Function_Training_Pytorch_Y[0]).
    Train_Dataset = TensorDataset(Inputs_Function_Training_Pytorch_X,Outputs_Function_Training_Pytorch_Y)
    
    # The DataLoader allows to simultaneously use batches for the training. Instead of a training loop with one single case as:
    # for index_Epoch in range(Number_Epochs): 
    #   Features_Inputs_X_Current = Inputs_Function_Training_Pytorch_X[index_Epoch]
    #   Labels_Outputs_Y = Labels_Outputs_Vector_Training_Pytorch_Y[index_Epoch]
    # Btch size fixed to treat several examples at the time. The shuffle allows to introduce some randomness, acts in the sense of the 
    # SGD.
    Train_Loader = DataLoader(Train_Dataset,
                              batch_size=1,
                              shuffle=True) # Try to settle it as false since not SGD used but Adam.
    
    # Model of FNO:
    Model = Fourier_Neural_Operator_Overall(Dimension_Width=64,Mode_Maximal=32,Dimension_Expressiveness=128)

    # Loss function choice:
    Loss_Function = nn.MSELoss()
    
    # Optimizer choice:
    Optimizer = torch.optim.Adam(Model.parameters(),lr=0.001) # Note that the learning rate might be settled to be halved every 100 epochs as in the article.

    # Training loop:
    Number_Epochs = 500
    for index_Epoch in range(1,Number_Epochs + 1):
        
        # Activating the training mode of the FNO model:
        Model.train()
        
        # Current Loss_Function at the current epoch initial value:
        Epoch_Loss = 0.0
        
        # Loop over the batches:
        for Inputs_Function_Batch_Training,Outputs_Function_Batch_Training in Train_Loader:
            
            # Forward pass calling for the current batch. The output will be of the shape (1,1).
            Output_FNO = Model(Inputs_Function_Batch_Training)
            
            # Loss function calculation on the overall batch:
            Loss = Loss_Function(Output_FNO,Outputs_Function_Batch_Training)

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
            Epoch_Loss += Loss.item()*Inputs_Function_Batch_Training.shape[0]
            
        # Regularization of the total loss of the epochs:
        Epoch_Loss /= len(Train_Loader.dataset)
        
        # Storage of the loss calculations:
        Loss_Storage.append(Epoch_Loss)

        if index_Epoch % 2 == 0 or index_Epoch == 1:
            Training_Accuracy = Compute_Relative_L2_Error(Model,
                                                          Inputs_Function_Training_Pytorch_X,
                                                          Outputs_Function_Training_Pytorch_Y)
            Testing_Accuracy = Compute_Relative_L2_Error(Model,
                                                         Inputs_Function_Testing_Pytorch_X,
                                                         Outputs_Function_Testing_Pytorch_Y)
            # Storage of the accuracy:
            Relative_L2_Testing_Storage.append(Testing_Accuracy)

            print(f"Epoch {index_Epoch:4d} | "
                  f"Loss: {Epoch_Loss:.4f} | "
                  f"Training accuracy: {Training_Accuracy:.4f} | "
                  f"Testing accuracy: {Testing_Accuracy:.4f}")
            
    # Final evaluation
    Test_Accuracy = Compute_Relative_L2_Error(Model,
                                              Inputs_Function_Testing_Pytorch_X, 
                                              Outputs_Function_Testing_Pytorch_Y)
    print(f"\nFinal test accuracy: {Test_Accuracy:.4f}")
    
    
    # Plot of the evolution of the Loss during training:
    Epochs_Vector = np.linspace(1,Number_Epochs,Number_Epochs,dtype=np.int16)
    plt.figure()
    plt.plot(Epochs_Vector,Loss_Storage,'-r')
    plt.axis("equal")
    plt.xlabel('Epoch number (-)')
    plt.ylabel('Loss of the Epoch (-)')
    plt.title('Loss evolution during training of the model')
    plt.show()
    
    # Save of the training parameters:
    torch.save(Model.state_dict(),"Burger_FNO_Trained_1000_Samples_500_Epochs.pth")
    # Save of the training history:
    np.savez("Burger_FNO_Training_History.npz",
             Loss_Storage=Loss_Storage,
             Relative_L2_Testing_Storage=Relative_L2_Testing_Storage)
    
    return Model

if __name__ == "__main__":
    Main_Function()





















