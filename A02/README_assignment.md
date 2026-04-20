ISNN Assignment Reproduction

This folder contains a complete implementation of ISNN-1 and ISNN-2 based on Sections 2 and 3.1 of the paper:
- 2503.00268v1.pdf

Implemented variants:
- PyTorch implementation using constrained parameters and autograd.
- NumPy implementation using matrix multiplication and manual backpropagation.

What is generated:
- Two toy datasets from Eq. (12) and Eq. (13)-(14), with the same sample counts and ranges described in Section 3.1.
- Training and testing loss curves versus epoch (similar to Figure 3 and Figure 5).
- Behavioral response curves for x = y = t = z sweeps (similar to Figure 4 and Figure 6).
- Final training/testing loss summary in CSV.

Run

From this folder:

python run_isnn_assignment.py

Optional arguments:
- --epochs-torch 600
- --epochs-numpy 600
- --lr-torch 0.001
- --lr-numpy 0.002
- --seed 7
- --data-dir generated_data
- --output-dir outputs

Outputs

Datasets:
- generated_data/toy_eq12_train.csv
- generated_data/toy_eq12_test.csv
- generated_data/toy_eq13_14_train.csv
- generated_data/toy_eq13_14_test.csv

Plots and metrics:
- outputs/toy_eq12/loss_curves.png
- outputs/toy_eq12/behavior_curves.png
- outputs/toy_eq12/history.json
- outputs/toy_eq13_14/loss_curves.png
- outputs/toy_eq13_14/behavior_curves.png
- outputs/toy_eq13_14/history.json
- outputs/results_summary.csv
