Steps:
1. conda create -n tv_scheduling python=3.10
2. conda activate tv_scheduling
3. pip install matplotlib>=3.5 numpy>=1.21 pandas>=1.4 scikit-learn>=1.0
4. python extract_features.py all_instances_folder features.csv
5. python dataset_diversity.py features.csv plots