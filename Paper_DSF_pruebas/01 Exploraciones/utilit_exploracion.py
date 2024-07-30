import os
import os.path as op
import pandas  as pd
import numpy   as np
import mne
import wfdb
import copy
import sys

from mne.datasets.utils                     import _get_path
from mne.datasets.sleep_physionet._utils    import _fetch_one

from braindecode.datasets                   import BaseDataset, BaseConcatDataset
from braindecode.preprocessing.preprocess   import _preprocess, Preprocessor
from braindecode.preprocessing.windowers    import _create_windows_from_events

sys.path.append(os.path.abspath(os.path.join('../../..','03 Dynamic-Spatial-Filtering')))
from transforms import ensure_valid_positions, AutoRejectDrop

from joblib                                 import Parallel, delayed
from functools                              import partial
from sklearn.model_selection import train_test_split


PC18_DIR           = op.join('..','..','..','03 Dynamic-Spatial-Filtering', 'data', 'pc18')
PC18_RECORDS       = op.join(PC18_DIR, 'sleep_records.csv')
PC18_INFO          = op.join(PC18_DIR, 'age-sex.csv')
PC18_SHA1_TRAINING = op.join(PC18_DIR, 'training_SHA1SUMS')
PC18_SHA1_TEST     = op.join(PC18_DIR, 'test_SHA1SUMS')
PC18_URL           = 'https://physionet.org/files/challenge-2018/1.0.0/'


def get_exp_name(dataset, model, dsf_type, denoising):
    return f'{dataset}-{model}-{dsf_type}-{denoising}'

def _data_path(path=None, force_update=False, update_path=None, verbose=None):
    
    """
    
    Get path to local copy of PC18 dataset.
    
    """

    key =  'PC18_DATASET_PATH'
    name = 'PC18_DATASET_SLEEP'
    path = _get_path(path, key, name)
    subdirs = os.listdir(path)

    if 'training' in subdirs or 'test' in subdirs:  # the specified path is
        # already at the training and test folders level
        return path
    else:
        return op.join('/media/martin/Disco2', 'Dsf_Data')
    

def fetch_pc18_data(subjects, path=None, force_update=False, update_path=None, base_url=PC18_URL, verbose=None):
    
    """
    
    Get paths to local copies of PhysioNet Challenge 2018 dataset files.

    This will fetch data from the publicly available PhysioNet Computing in
    Cardiology Challenge 2018 dataset on sleep arousal detection [1]_ [2]_.
    This corresponds to 1983 recordings from individual subjects with
    (suspected) sleep apnea. The dataset is separated into a training set with
    994 recordings for which arousal annotation are available and a test set
    with 989 recordings for which the labels have not been revealed. Across the
    entire dataset, mean age is 55 years old and 65% of recordings are from
    male subjects.

    More information can be found on the
    `physionet website <https://physionet.org/content/challenge-2018/1.0.0/>`_.

    Parameters
    ----------
    subjects : list of int
        The subjects to use. Can be in the range of 0-1982 (inclusive). Test
        recordings are 0-988, while training recordings are 989-1982.
    path : None | str
        Location of where to look for the PC18 data storing location. If None,
        the environment variable or config parameter ``PC18_DATASET_PATH``
        is used. If it doesn't exist, the "~/mne_data" directory is used. If
        the dataset is not found under the given path, the data will be
        automatically downloaded to the specified folder.
    force_update : bool
        Force update of the dataset even if a local copy exists.
    update_path : bool | None
        If True, set the PC18_DATASET_PATH in mne-python config to the given
        path. If None, the user is prompted.
    base_url : str
        The URL root.
    %(verbose)s

    Returns
    -------
    paths : list
        List of local data paths of the given type.

    References
    ----------
    .. [1] Mohammad M Ghassemi, Benjamin E Moody, Li-wei H Lehman, Christopher
      Song, Qiao Li, Haoqi Sun, Roger G Mark, M Brandon Westover, Gari D
      Clifford. You Snooze, You Win: the PhysioNet/Computing in Cardiology
      Challenge 2018.
    .. [2] Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C.,
      Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and
      PhysioNet: Components of a new research resource for complex physiologic
      signals. Circulation [Online]. 101 (23), pp. e215-e220.)

    """
    
    records         = pd.read_csv(PC18_RECORDS)
    psg_records     = records[records['Record type'] == 'PSG']
    hea_records     = records[records['Record type'] == 'Header']
    arousal_records = records[records['Record type'] == 'Arousal']

    path            = _data_path(path=path, update_path=update_path)
    params          = [path, force_update, base_url]

    fnames          = []
    for subject in subjects:
        for idx in np.where(psg_records['Subject'] == subject)[0]:
            psg_fname = _fetch_one(psg_records['fname'].iloc[idx], psg_records['sha'].iloc[idx], *params)
            hea_fname = _fetch_one(hea_records['fname'].iloc[idx], hea_records['sha'].iloc[idx], *params)
            if psg_records['Split'].iloc[idx] == 'training':
                train_idx = np.where(
                    arousal_records['Subject'] == subject)[0][0]
                arousal_fname = _fetch_one(
                    arousal_records['fname'].iloc[train_idx],
                    arousal_records['sha'].iloc[train_idx], *params)
            else:
                arousal_fname = None
            fnames.append([psg_fname, hea_fname, arousal_fname])

    return fnames

def convert_wfdb_anns_to_mne_annotations(annots):

    """

    Convert wfdb.io.Annotation format to MNE's.

    Parameters
    ----------
    annots : wfdb.io.Annotation
        Annotation object obtained by e.g. loading an annotation file with
        wfdb.rdann().

    Returns
    -------
    mne.Annotations :
        MNE Annotations object.

    """
    
    ann_chs = set(annots.chan)
    onsets = annots.sample / annots.fs
    new_onset, new_duration, new_description = list(), list(), list()
    for ch in ann_chs:
        mask = annots.chan == ch
        ch_onsets = onsets[mask]
        ch_descs = np.array(annots.aux_note)[mask]

        # Events with beginning and end, defined by '(event' and 'event)'
        if all([(i.startswith('(') or i.endswith(')')) for i in ch_descs]):
            pass
        else:  # Sleep stage-like annotations
            ch_durations = np.concatenate([np.diff(ch_onsets), [30]])
            assert all(ch_durations > 0), 'Negative duration'
            new_onset.extend(ch_onsets)
            new_duration.extend(ch_durations)
            new_description.extend(ch_descs)

    mne_annots = mne.Annotations(new_onset, new_duration, new_description, orig_time=None)

    return mne_annots

class PC18(BaseConcatDataset):

    """
    
    Physionet Challenge 2018 polysomnography dataset.

    Sleep dataset from https://physionet.org/content/challenge-2018/1.0.0/.
    Contains overnight recordings from 1983 healthy subjects.

    See `fetch_pc18_data` for a more complete description.

    Parameters
    ----------
    subject_ids: list(int) | str | None
        (list of) int of subject(s) to be loaded. If None, load all available
        subjects. If 'training', load all training recordings. If 'test', load
        all test recordings.
    path : None | str
        Location of where to look for the PC18 data storing location. If None,
        the environment variable or config parameter ``MNE_DATASETS_PC18_PATH``
        is used. If it doesn't exist, the "~/mne_data" directory is used. If
        the dataset is not found under the given path, the data will be
        automatically downloaded to the specified folder.
    load_eeg_only: bool
        If True, only load the EEG channels and discard the others (EOG, EMG,
        temperature, respiration) to avoid resampling the other signals.
    preproc : list(Preprocessor) | None
        List of preprocessors to apply to each file individually. This way the
        data can e.g., be downsampled (temporally and spatially) to limit the
        memory usage of the entire Dataset object. This also enables applying
        preprocessing in parallel over the recordings.
    windower : callable | None
        Function to split the raw data into windows. If provided, windowing is
        integrated into the loading process (after preprocessing) such that
        memory usage is minized while allowing parallelization.
    n_jobs : int
        Number of parallel processes.

    """
    
    def __init__(self, subject_ids=None, path=None, load_eeg_only=True, preproc=None, windower=None, n_jobs=1):
        paths = fetch_pc18_data(subject_ids, path=path)
        self.info_df = pd.read_csv(PC18_INFO)

        if n_jobs == 1:
            all_base_ds = [self._load_raw(subject_id, p[0], p[2], load_eeg_only=load_eeg_only,
                preproc=preproc, windower=windower)
                for subject_id, p in zip(subject_ids, paths)]
        else:
            all_base_ds = Parallel(n_jobs=n_jobs)(delayed(self._load_raw)(
                subject_id, p[0], p[2], load_eeg_only=load_eeg_only,
                preproc=preproc, windower=windower)
                for subject_id, p in zip(subject_ids, paths))
        super().__init__(all_base_ds)

    def _load_raw(self, subj_nb, raw_fname, arousal_fname, load_eeg_only, preproc, windower):
        raw_fname     = raw_fname[0] if isinstance(raw_fname, tuple) else raw_fname
        arousal_fname = arousal_fname[0] if isinstance(arousal_fname, tuple) else arousal_fname


        channel_types = ['eeg'] * 7
        if load_eeg_only:
            channels  = list(range(7))
        else:
            channel_types += ['emg', 'misc', 'misc', 'misc', 'misc', 'ecg']
            channels  = None

        # Load raw signals and header
        record = wfdb.io.rdrecord(op.splitext(raw_fname)[0], channels=channels)

        # Convert to right units for MNE (EEG should be in V)
        data = record.p_signal.T
        data[np.array(record.units) == 'uV'] /= 1e6
        data[np.array(record.units) == 'mV'] /= 1e3
        info = mne.create_info(record.sig_name, record.fs, channel_types)
        out = mne.io.RawArray(data, info)

        # Extract annotations
        if arousal_fname is not None:
            annots = wfdb.rdann(
                                op.splitext(raw_fname)[0], 'arousal', sampfrom=0, sampto=None,
                                shift_samps=False, return_label_elements=['symbol'],
                                summarize_labels=False
                               )
            mne_annots = convert_wfdb_anns_to_mne_annotations(annots)
            out.set_annotations(mne_annots)
        record_name = op.splitext(op.basename(raw_fname))[0]
        print(record_name)
        record_info = self.info_df[
            self.info_df['Record'] == record_name].iloc[0]
        if record_info['Record'].startswith('tr'):
            split = 'training'
        elif record_info['Record'].startswith('te'):
            split = 'test'
        else:
            split = 'unknown'

        desc = pd.Series({
            'subject': subj_nb,
            'record': record_info['Record'],
            'split': split,
            'age': record_info['Age'],
            'sex': record_info['Sex']
        }, name='')
        out = BaseDataset(out, desc)
        print(out)
        if preproc is not None:
            _preprocess(out, None, preproc)

        if windower is not None:
            out = windower(out)
            out.windows.load_data()
        return out
    

def scale(x, k):
    return k * x

def cast(x, dtype):
    return x.astype(dtype)

def load_data(dataset, window_size_s, n_jobs):
    """
    
    Load, preprocess and window data.
    
    """       

    subject_ids = [989, 990, 991]
    ch_names    = ['F3-M2', 'F4-M1', 'O1-M2', 'O2-M1']
    preproc     = [
                  Preprocessor('pick_channels', ch_names=ch_names, ordered=True),
                  Preprocessor('filter', l_freq=None, h_freq=30, n_jobs=1),
                  Preprocessor('resample', sfreq=100., n_jobs=1),
                  Preprocessor(scale, k=1e6),
                  Preprocessor(cast, dtype=np.float32)
                  ]

    window_size_samples = int(window_size_s * 100)
    mapping             = {'W': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'R': 4}
    windower            = partial(
                                 _create_windows_from_events, 
                                 infer_mapping=False,
                                 infer_window_size_stride=False, 
                                 trial_start_offset_samples=0,
                                 trial_stop_offset_samples=0,
                                 window_size_samples=window_size_samples,
                                 window_stride_samples=window_size_samples, 
                                 mapping=mapping
                                 )

    dataset = PC18(subject_ids=subject_ids, preproc=preproc, windower=windower, n_jobs=n_jobs)
    return dataset


def parallel_preproc(windows_dataset, preproc, n_jobs):
    """Apply preprocessor in parallel on BaseDatasets.
    """
    def _apply_preproc(ds, preproc):
        if ds.windows.preload:  # Deep copy so that numpy arrays are modifiable
            ds = copy.deepcopy(ds)
        try:
            _preprocess(ds, None, preproc)
        except Exception as e:
            print(e)
            print('Not applying preproc')
        return ds

    preproc_ds = Parallel(n_jobs=n_jobs)(delayed(_apply_preproc)(
        windows_dataset.datasets.pop(0), preproc)
        for _ in range(len(windows_dataset.datasets)))

    return BaseConcatDataset(preproc_ds)


def apply_autoreject(windows_dataset, random_state, n_jobs):
    ar = AutoRejectDrop(cv=5, random_state=random_state, drop=False, n_jobs=1)
    preproc = [
        Preprocessor(ensure_valid_positions, apply_on_array=False),
        Preprocessor(ar, apply_on_array=False)
    ]
    return parallel_preproc(windows_dataset, preproc, n_jobs)

def split_dataset(base_concat_ds, valid_size, test_size,
                  random_state_valid=None, random_state_test=None):
    """Split dataset into train, valid and test sets.

    Parameters
    ----------
    base_concat_ds : braindecode.datasets.BaseConcatDataset
        Dataset to split.
    valid_size : float
        Proportion of the dataset to include in the valid split.
    test_size : float
        Proportion of the dataset to include in the test split.
    random_state_valid : int | np.random.RandomState | None
        Controls the shuffling applied to the data before applying the
        validation split.
    random_state_test : int | np.random.RandomState | None
        Controls the shuffling applied to the data before applying the test
        split.

    Returns
    -------
    BaseConcatDataset, BaseConcatDataset, BaseConcatDataset :
        Train, valid and test splits.
    """
    rec_inds = np.arange(len(base_concat_ds.datasets))

    train_valid_inds, test_inds = train_test_split(
        rec_inds, test_size=test_size, random_state=random_state_test)
    train_inds, valid_inds = train_test_split(
        train_valid_inds, test_size=valid_size / (1 - test_size),
        random_state=random_state_valid)

    split_ds = base_concat_ds.split(
        [train_inds.tolist(), valid_inds.tolist(), test_inds.tolist()])

    return split_ds['0'], split_ds['1'], split_ds['2']

