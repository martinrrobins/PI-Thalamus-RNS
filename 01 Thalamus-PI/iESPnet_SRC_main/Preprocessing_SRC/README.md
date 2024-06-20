# Data preprocessing

The following decision were made during the preprocessing:
### Data
1. EDF files were used.
2. Annotation were made, in most of the cases, for two experts (VK and NZ). If an EDF file has both annotation files, the one with the 'NZ' key is used. 
3. Valid name annotations are:
'eof', 'sz_on',  'sz_on_l', 'sz_on_r', 'sz', 'sz_l', 'sz_r'
4. Seizures in which there is no agreement are not used. 
5. Categorical variables ('eof', 'sz_on' and 'sz') were transformed to discrete variables (0,1,2).
6. Epoch of 90 s were extracted accordingly to the information given in the 'end of file' (eof) annotation. If an epoch lasted less than 90 s, was zero padded*. If an epoch lasted more than 90 s, then N non-overlapping windows of 90 s length were extracted. 

*Different padded strategies were evaluated: signal zero padded, signal mirrowing, spectrogram zero padded. After running some analyses, I decided to use signal zero padded as the final way of constructing the spectrogram. You can refer to Nexu2/iESPnet/Net_experiments/code/Experiments/Ablation_studies/Spectrogram_studytudy which outputs are here 
Nexus2/iESPnet/Net_experiments/outputs/AblationStudies/ESPnet/SPE.


### Spectrograms
1. spectrograms were constructed using a kaiser window of 1 s lenght and a beta parameter equal to 10. 
2. Spectrogram were transform to DB, setting the top_db value to 40.
3. A tensor of dimension frequency x times x channel is saved per epoch.


#### Files
'create_spectrograms_timelabel_zeropad_all.py', EDF files from each subject and programming epoch are read and then converted to proper spectrograms. The time onset and sz label is transform to a one-hot sample-wise continous label. The metadata_file is created and saved also here per each subject. All PITT patients are used, regarless the electrode location. This is the data used during ESPnet training.

'create_spectrograms.py', EDF files from each subject and programming epoch are read and then converted to proper spectrograms. The time onset and sz label are saved. The metadata_file is created and saved also here per each subject. All PITT patients are used, regarless the electrode location. This is the data used during ESPdetect training.

'count_short_files.py', to check how many files were padded. This could explain patient's performance.

'viz_gen_spec_timelabel_zeropad_all.py', extracted spectrogram for one channel are saved in a PDF file. The PDF outputs are located at '/Preprocesing_plots'. There is one PDF file per subject. 

'generate_metadafile_annots.py', collect metadata information of the RNS_DataBank folder. 

'generate_subjects_info' generate csv file with patient info, e.g electrode location, #/ of seizures files, etc.

'generate_metadafile_nothalamus' generate nonthalamus metadata file. this is important when training and evaluating the model in non-thalamus patients.

#### Outputs
The created spectrograms can be found here:

''Nexus2:/iESPnet/Data/RNS_Databank_Spectrograms/TimeLabelZeropadAll''

### Steps followed for creating the spectograms Database

1. Create the spectrograms and all_files_medatada.csv (used for the Pytorch Dataloader to access data during training). Function: create_spectrograms_timelabel_zeropad_all.py
2. Detele non-thalamus patients from all_files_metada to avoid them to be used during the net training. Function: Generate_metadafile_nonthalamus.py. This step can be avoided if you are using all patients data. 
3. Create subject info file to get Patient ID during training as well as other demographic info. Here I also added the number of spectrogram per patient. Function: generatue_subject_info.py

 
#### documentation @Jan 2022.Updated @March 2023
#### Author: VPeterson


