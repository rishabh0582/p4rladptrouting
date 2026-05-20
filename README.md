Jitter-Aware Adaptive Routing using P4 + RL

Flow:
0)P4 files are already compiled
1)Run topology file using sudo python3 topology/topology.py
2)Run either:
     the controller file using python3 controller/controller.py (--duration 0 flag added for an indefinite run else the controller runs for 30 seconds) 
     OR
     the training script using python3 rl/train.py --episodes x 
3)Evaluation plots are generated in results subdirectory

