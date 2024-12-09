import numpy as np
import tensorflow as tf
import os
import h5py
import pandas as pd
import scipy.signal
from training.preprocess import DataPreprocessor
from copy import deepcopy as dc

class Config_DD:
    seed = 100
    n_class = 2
    fs = 100
    dt = 1.0 / fs
    freq_range = [0, fs / 2]
    time_range = [0, 30]
    nperseg = 30
    nfft = 60
    plot = False
    # nt = 9001
    nt = 15001
    X_shape = [31, 1002, 2]
    Y_shape = [31, 1002, n_class]
    signal_shape = [31, 1002]
    # noise_shape = signal_shape
    use_seed = False
    queue_size = 10
    noise_mean = 2
    noise_std = 1
    # noise_low = 1
    # noise_high = 5
    use_buffer = True
    snr_threshold = 10

class DataReader(DataPreprocessor):
    def __init__(self, args, signal_dir, signal_list, config=Config_DD):
        super().__init__(
            data_channels=['e', 'n', 'z'],
            sampling_rate=100,
            in_samples=args.in_samples,
            min_snr=args.min_snr,
            coda_ratio=args.coda_ratio,
            norm_mode=args.norm_mode,
            p_position_ratio=args.p_position_ratio,
            add_event_rate=args.add_event_rate,
            add_noise_rate=args.add_noise_rate,
            add_gap_rate=args.add_gap_rate,
            drop_channel_rate=args.drop_channel_rate,
            scale_amplitude_rate=args.scale_amplitude_rate,
            pre_emphasis_rate=args.pre_emphasis_rate,
            pre_emphasis_ratio=args.pre_emphasis_ratio,
            max_event_num=args.max_event_num,
            generate_noise_rate=args.generate_noise_rate,
            shift_event_rate=args.shift_event_rate,
            mask_percent=args.mask_percent,
            noise_percent=args.noise_percent,
            min_event_gap_sec=args.min_event_gap,
            soft_label_shape=args.label_shape,
            soft_label_width=int(args.label_width * 100),
            dtype=np.float32,
        )
        self.signal_dir = signal_dir
        # self.noise_dir = noise_dir
        self.signal = pd.read_csv(signal_list, header=0)
        # self.noise = pd.read_csv(noise_list, header=0)
        self.n_signal = len(self.signal)
        self.X_shape = config.X_shape
        self.Y_shape = config.Y_shape
        self.n_class = config.n_class
        self.buffer_signal = {}
        self.buffer_noise = {}
        self.config = config
        self.epsilon = 1e-8

    def get_snr(self, data, itp, dit=300):
        tmp_std = np.std(data[itp - dit : itp])
        return np.std(data[itp : itp + dit]) / tmp_std if tmp_std > 0 else 0

    def _load_signal(self, fname_signal):
        if fname_signal not in self.buffer_signal:
            bucket, array = fname_signal.split('$')
            n, c, l = [int(i) for i in array.split(',:')]
            path = os.path.join(self.signal_dir, f"comcat_waveforms.hdf")
            with h5py.File(path, "r") as f:
                meta = f.get(f"data/{bucket}")[n]
                meta = np.array(meta).astype(np.float32)
                meta = np.nan_to_num(meta)
            data_FT, snr = [], []
            for i in range(3):
                tmp_data = meta[i, :]
                tmp_itp = self.signal.loc[self.signal['trace_name'] == fname_signal, 'trace_P_arrival_sample'].values[0]
                snr.append(self.get_snr(tmp_data, tmp_itp))
                # tmp_data -= np.mean(tmp_data)
                # _, _, tmp_FT = scipy.signal.stft(
                #     tmp_data,
                #     fs=self.sampling_rate,
                #     nperseg=self.config.nperseg,
                #     nfft=self.config.nfft,
                #     boundary='zeros',
                # )
                # data_FT.append(tmp_FT)
            # data_FT = np.stack(data_FT, axis=-1)
            self.buffer_signal[fname_signal] = {'snr': snr}
        return self.buffer_signal[fname_signal]

    def process_sample(self, indices):
        np.random.shuffle(indices)
        for index in indices:
            row = self.signal.iloc[index]
            fname_signal = row['trace_name']
            bucket, array = fname_signal.split('$')
            n, c, l = [int(i) for i in array.split(',:')]
            path = os.path.join(self.signal_dir, f"comcat_waveforms.hdf")
            with h5py.File(path, "r") as f:
                sig = f.get(f"data/{bucket}")[n]
                sig = np.array(sig).astype(np.float32)
                sig = np.nan_to_num(sig)
                tmp_sig = dc(sig)

            meta_signal = self._load_signal(fname_signal)
            
            event = {'data': sig, 'ppks': [], 'spks': [], 'snr': [meta_signal['snr']]}
            event = self.process(event, augmentation=True)

            for j in range(3):
                if meta_signal['snr'][j] <= self.config.snr_threshold:
                    continue

                _, _, tmp_signal = scipy.signal.stft(
                    event['data'][j],
                    fs=self.sampling_rate,
                    nperseg=self.config.nperseg,
                    nfft=self.config.nfft,
                    boundary='zeros',
                )

                _, _, clean_signal = scipy.signal.stft(
                    tmp_sig[j],
                    fs=self.sampling_rate,
                    nperseg=self.config.nperseg,
                    nfft=self.config.nfft,
                    boundary='zeros',
                ) 

                uncleaned_signal = np.stack([tmp_signal.real, tmp_signal.imag], axis=-1)
               
                if np.isnan(uncleaned_signal).any() or np.isinf(uncleaned_signal).any():
                    continue

                uncleaned_signal = uncleaned_signal / np.std(uncleaned_signal + self.epsilon)
            
                mask = np.zeros([clean_signal.shape[0], clean_signal.shape[1], self.n_class])
                tmp_mask = np.abs(clean_signal) / (np.abs(clean_signal) + self.epsilon)
                tmp_mask[tmp_mask >= 1] = 1
                tmp_mask[tmp_mask <= 0] = 0
                mask[:, :, 0] = tmp_mask
                mask[:, :, 1] = 1 - tmp_mask

                # Apply preprocessing steps from DataPreprocessor
                # event = {'data': sig, 'ppks': [], 'spks': [], 'snr': [meta_signal['snr']]}
                # event = self.process(event, augmentation=True)
                # processed_signal = event['data']

                yield (tf.convert_to_tensor(uncleaned_signal, dtype=tf.float32), tf.convert_to_tensor(mask, dtype=tf.float32))

    def prepare_data(self, batch_size=20):
        def generator():
            indices = np.arange(self.n_signal)
            while True:
                for sample in self.process_sample(indices):
                    yield sample

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(self.X_shape[0], self.X_shape[1], 2), dtype=tf.float32),
                tf.TensorSpec(shape=(self.Y_shape[0], self.Y_shape[1], self.n_class), dtype=tf.float32)
            )
        )
        dataset = dataset.repeat().batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset