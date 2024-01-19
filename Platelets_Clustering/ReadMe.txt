READ ME

Platelets_Clustering.py
:user script for automatic data processing

usage: Platelets_Clustering.py [-h] [--algorithm {DBSCAN,k-means,Agglom}] [--figures {yes,no}]
                               [--concentration CONCENTRATION [CONCENTRATION ...]]
                               indir outdir

positional arguments:
  indir                 Input directory for folder with datasets
  outdir                Output directory folder

options:
  -h, --help            show this help message and exit
  --algorithm {DBSCAN,k-means,Agglom}
                        Choose algoritm for clusterization
  --figures {yes,no}    Should graphs with clustering be saved
  --concentration       List of concentration for each dataset


Research.ipynb
:research file displaying the progress of data research and 
algorithms for selecting optimal parameters for automatic data processing (details in the Report.pdf)


Test
:data for research (section 2.3)

Data_1, Data_2, Data1.xlsx, Data2.xlsx
:data for research (section 2.4)

Test_Full
:data for testing user script