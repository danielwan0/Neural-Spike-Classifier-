#Import Libraries
import numpy
import matplotlib.pyplot as plt
import scipy.special
import scipy.io as spio
import scipy.signal
from scipy.signal import find_peaks
from scipy.signal import butter, filtfilt

# Neural network class definition
class n:
    # Init the network, this gets run whenever we make a new instance of this class
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set the number of nodes in each input, hidden and output layer
        self.i_nodes = input_nodes
        self.h_nodes = hidden_nodes
        self.o_nodes = output_nodes
        # Weight matrices, wih (input -> hidden) and who (hidden -> output)
        self.wih = numpy.random.normal(0.0, pow(self.h_nodes, -0.5), (self.h_nodes, self.i_nodes))
        self.who = numpy.random.normal(0.0, pow(self.o_nodes, -0.5), (self.o_nodes, self.h_nodes))

        # Set the learning rate
        self.lr = learning_rate

        # Set the activation function, the logistic sigmoid
        self.activation_function = lambda x: scipy.special.expit(x)


 # Train the network using back-propagation of errors
    def train(self, inputs_list, targets_list):
        # Convert inputs into 2D arrays
        inputs_array = numpy.array(inputs_list, ndmin=2).T
        targets_array = numpy.array(targets_list, ndmin=2).T
    
        # Calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs_array)
    
        # Calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # Calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
    
        # Calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        # Current error is (target - actual)
        output_errors = targets_array - final_outputs
        
        # Hidden layer errors are the output errors, split by the weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)
    
        # Update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
        numpy.transpose(hidden_outputs))
        
        # Update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
        numpy.transpose(inputs_array))


    # Query the network
    def query(self, inputs_list):
        # Convert the inputs list into a 2D array
        inputs_array = numpy.array(inputs_list, ndmin=2).T
        
        # Calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs_array)
        
        # Calculate output from the hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # Calculate signals into final layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        
        # Calculate outputs from the final layer
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs

#Initialise the input parameters for the Neural Network
input_nodes = 40
hidden_nodes = 40
output_nodes = 4
learning_rate = 0.03

network = n (input_nodes, hidden_nodes, output_nodes, learning_rate)

#Define function to find the nearest number in an array to an inout number
#This function is used when testing the training data against itself
def FindNearest(ArrayToScan,SignalIndex):
    ArrayToScan = numpy.asarray(ArrayToScan)
    idx = (numpy.abs(ArrayToScan-SignalIndex)).argmin()
    return ArrayToScan[idx]

#Initalise the figure for plotting
fig, axs = plt.subplots(2, 1)
axs = axs.flatten()

#Load the training data
training_mat = spio.loadmat('training.mat', squeeze_me=True)
training_d = training_mat['d']
training_Index = training_mat['Index']
training_Class = training_mat['Class']


#Define the function to filter the data using a butterworth bandpass filter
def ButterFilter(RecordingData):
    # Sampling Frequency
    fs = 25000
    # Bandpass low-cut frequency
    lowcutfreq = 50
    # Bandpass high-cut frequency
    highcutfreq = 1500
    # Nyquist frequency (half the sampling frequency)
    nyq = fs/2

    #Use the butterworth filter in the 4th order.
    b, a = butter(4, [lowcutfreq/nyq, highcutfreq/nyq], btype='band')
    Filtered = filtfilt(b, a, RecordingData)
    return Filtered

#Filter the training data
training_d = ButterFilter(training_d)


#Calculate the length of the training Index
length = len(training_Index)

#Loop through each value in the training Index vector
for i in range(0,length):
    #Create the window that coveres the neuron spike for each value in the training index
    j = training_Index[i]
    training_sample_d = training_d[j:j+40]
    # Train the neural network on each training sample
    # Scale and shift the inputs from 0..11.9436 to 0.01..1
    inputs = (numpy.asfarray(training_sample_d[:]) / 11.9436 * 0.99) + 0.01
    # Create the target output values (all 0.01, except the desired label which is 0.99)
    targets = numpy.zeros(output_nodes) + 0.01
    # All_values[0] is the target label for this record
    targets[int(training_Class[i]-1)] = 0.99
    # Train the network
    network.train(inputs, targets)
    pass

#Load the submission data
test_mat = spio.loadmat('submission.mat', squeeze_me=True)
test_d = test_mat['d']

#Plot the un-filtered submission data
y_range = range(0,1440000)
axs[0].plot(y_range, test_d[:])

#Filter the submission data
test_d = ButterFilter(test_d)

#Initalise the peaks vector ready to be filled
peaks = []

#Find all the peaks in the submission data
peaks, _ = find_peaks(test_d[:], height=1, threshold=None, distance=None, prominence=1, width=None, wlen=None, rel_height=None)

#Find the base of each neuron spike so that the vector "peaks" is in as similar format to the vector "training-index" as possible
t=0
for i in peaks:
    u=i
    while test_d[u] > test_d[u-1]:
        u=u-1
    peaks[t] = u
    t=t+1

#Plot the filtered submission data along with the locations of the base of each peak
axs[1].plot(y_range, test_d[:])
axs[1].plot(peaks, test_d[peaks], "x")

# Scorecard list for each class to see how well the network performs, initially empty
scorecard1 = []
scorecard2 = []
scorecard3 = []
scorecard4 = []

#Initalise the vector to store all the classes
Sub_Class = []

#Calculate the lenth of the peaks vector
lengthp = len(peaks)

# Loop through all of the values in the peaks data
for i in range(0,lengthp):
    #Create the window that coveres the neuron spike for each value in the peaks vector which is essentiall the submission index vector
    j = peaks[i]
    test_sample_d = test_d[j:j+40]

    #This section is used when comparing against the training data
    #Value = FindRangeInArray(test_Index.tolist(), j-15, j+35)
    #Value = FindNearest(test_Index, j)
    #C = test_Index.tolist()
    #C = C.index(Value)
    # The correct label is the first value
    #correct_label = int(test_Class[C])
    #print(correct_label, "Correct label")

    # Scale and shift the inputs
    inputs = (numpy.asfarray(test_sample_d[:]) / 15.4372 * 0.99) + 0.01
    # Query the network
    outputs = network.query(inputs)
    # The index of the highest value output corresponds to the label
    label = numpy.argmax(outputs)+1
    print(label)
    # Append a 1 to the scorecard associated to that label
    if (label == 1):
        scorecard1.append(1)
    elif (label ==2):
        scorecard2.append(1)
    elif (label==3):
        scorecard3.append(1)
    elif (label==4):
        scorecard4.append(1)
    Sub_Class.append(label)


    # Append either a 1 or a 0 to the scorecard list
    #Used when comparing against the training data
    #if (label == correct_label):
    #    scorecard.append(1)
    #else:
    #    scorecard.append(0)
    #    pass
    #pass


# Calculate the performance score, the fraction of correct answers
#print("Performance = ", (scorecard_array.sum() / scorecard_array.size)*100, '%')

#Save the Class and Index
final_matlab_file={}
final_matlab_file['Index'] = peaks
final_matlab_file['Class'] = Sub_Class
spio.savemat('11818',final_matlab_file)

#Calculate the number of each class which has been identified
scorecard1_array = numpy.asarray(scorecard1)
scorecard2_array = numpy.asarray(scorecard2)
scorecard3_array = numpy.asarray(scorecard3)
scorecard4_array = numpy.asarray(scorecard4)

#Print the number of each class which has been identified in addition to the total number of neuron spikes
print("Number of Neuron Spikes = ", lengthp)
print("Number of 1s = ", scorecard1_array.sum())
print("Number of 2s = ", scorecard2_array.sum())
print("Number of 3s = ", scorecard3_array.sum())
print("Number of 4s = ", scorecard4_array.sum())
plt.show()
