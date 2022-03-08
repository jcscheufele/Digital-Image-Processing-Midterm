# Digital-Image-Processing-Midterm
Joseph Scheufele and 
Jose Figueroa

please visit our git repository to view our code in its entirety:https://github.com/jcscheufele/Digital-Image-Processing-Midterm


Task:
Learning how Neural Networks can be used for nighttime place recognition.

There are several branches in this repository.

main:
 - disregard

master:
 - disregard

algorithm:
 - This branch is used for testing out our methods of sift and orb feature detection and the different matching techniques.

valid:
 - This is a branch for deep learning that uses a fraction of the training image dataset as validation data so as to not data snoop on the testing images.

single: 
 - This branch changes the architechture of the neural network to use a singular flow of information rather than two.
 - Both the day and night images are given to a convolutional layer as single channels.

simple:
 - This branch is used to test the split linear technique. Each image is flattened and presented to their own linear network and then combined into a single output.

The valid, single and simple branches should have code files labeled dataset.py, network.py, and run_code.py. These files work in concert to create or load the datasets, instantiate the particular network, and to train the network.

The algorithm branch should contain a file algorithm.py, this is the only file necessary, it contains the code for running the pipeline and testing out cosine similarity, structural similarity, sift and orb and the different matching techniques.



