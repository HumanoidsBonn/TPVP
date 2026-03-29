# View_planning_simulator
This is the modified [view planning simulation system](https://github.com/psc0628/MA-SCVP).  
Follow the original one for installion.
## Environment_Config
The folder contains all needed files for bbx and view space of our test cases.  
You can generate your own file using the code in the subfolder BBX_code.
## Prepare
Make sure "model_path" in DefaultConfiguration.yaml contains processed 3D models.  
You can find our processed data from [Greenhouse Multi-Session Row Dataset](https://www.kaggle.com/datasets/sicongpan/greenhouse-multi-session-row-dataset).  
The "pre_path" in DefaultConfiguration.yaml is the results saving path.  
The "nricp_path" should be the python path.  
The "Debug" can be set to 1 to generate the online candidate and 0 for normal use.
## Usage
The mode of the system should be input in the Console.  
Then input 1 for our method and 0 for other methods.  
Next input the method id you want to test. 
Finally give the object model names in the Console (-1 to break input).  
### Mode 6
The system will genertate the oracle visible set for evaluation.
### Mode 1 with Debug (if you are not using our Environment_Config)
The system will genertate the online candidate look-at targets for the method. You should run this frist.
### Mode 4
The system will genertate the ground truth point clouds of all visble voxels for all input objects. This will speed up evluation.  You should run this second.
### Mode 1
The system will test all input objects by the input method.
#### Example Run View Planning
Input 1.
Input 1.
Input 11.
Input room1_20250918.



