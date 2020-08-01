This folder contains the code and the preprocessed data (i.e., mean BOLD responses, mouse trajectory data, experimental variables) used in the fMRI study. Note that the code depends on the old versions of TensorFlow and GPflow (TensorFlow 1.X and GPflow 1 is required).

 * Jupyter notebook
   * fMRI_GPJM_complete_3dim.ipynb: Fitting the three-dimensional GPJM to the fMRI dataset. Includes the code of the model.
 * Data
   * coherence_scaled_013_runX.npy: Scaled coherence values
   * mouse_trajectory_centered_scaled_013_runX.npy: Mouse trajectory data (centered and scaled)
   * Timing013_meanTS_block1.npy: Mean BOLD responses for 16 regions of interest
   * ts_dense.npy, ts_sparse.npy: Time index vectors (neural - low frequency, behavioral - high frequency)
