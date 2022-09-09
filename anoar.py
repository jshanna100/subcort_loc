# -*- coding: utf-8 -*-
'''
Automated non-ocular artefact removal (ANOAR) 2016, Jeff Hanna
Module works with MNE Python. Globally bad channels are removed by
calculating a distance-weighted absolute correlation matrix, and identifying
those that correlate poorly with their neighbours. As for artefact rejection,
the sum of squared deviance from the ERP average is calculated for each trial,
individually for each channel. This results in a matrix of noise measurements,
TxC, where T is the number of trials, and C is the number of channels. Noise
levels laying outside a given threshold of a probability distribution are marked
as bad. Because the values describe both trial and channel, individual channels
can be marked as bad without throwing out the entire trial. The values of these
channels are replaced by interpolation within the trial alone. Trials which have
too many bad channels are marked as bad entirely.
The critical manipulation here that prevents the removal of ocular artefacts
is that the correlation with the EOG channels for each trial-channel are calculated,
and the noise value for that trial-channel is multiplied by 1-r² from the correlations
with other channels, this is effect removes the portion of the noise caused by
parts of the signal that correlate with EOG (or any other recorded source of
noise, EKG, etc).
'''
import mne
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats.mstats import zscore
from scipy.stats import norm

def _get_pos(info, picks):
    # get 3d sensor positions
    pos = []
    for pick in picks:
        pos.append(info["chs"][pick]["loc"][:3])
    pos = np.array(pos)
    return pos

def _get_chan_dists(pos):
    # return channel*channel distance matrix
    chan_n = pos.shape[0]
    dim_n = pos.shape[1]
    if dim_n > 3 or dim_n < 2:
        raise ValueError('Number of dimensions must be 2 or 3. {a} were given.'.format(a=dim_n))
    dist_mat = np.zeros((chan_n, chan_n))
    for pi_idx in range(chan_n):
        for pj_idx in range(pi_idx+1,chan_n):
            dist_mat[pi_idx,pj_idx] = np.linalg.norm(pos[pi_idx,:]-pos[pj_idx,:])
    dist_mat += dist_mat.T
    return dist_mat

def _get_neighbs(dist_mat, neighb_n):
    # return neighb_n nearest neighbours for electrodes, pos channel*coord
    # float array (2 or 3 dim)
    chan_n = dist_mat.shape[0]
    if chan_n < neighb_n:
            raise ValueError('Must be at least as many channels as neighb_n parameter.')
    neighbs = []
    for d_idx in range(chan_n):
        neighbs.append(np.argsort(dist_mat[d_idx,])[1:neighb_n+1].tolist())
    return neighbs

def _twin_builder(linspace):
    # make list of time window partitions in the raw data
    part_n = len(linspace)
    twin_part = np.empty((part_n-1, 2), dtype=int)
    twin_part[0,0] = linspace[0]
    twin_part[-1,1] = linspace[-1]
    for i in range(part_n-2):
        twin_part[i,1] = linspace[i+1]
        twin_part[i+1,0] = linspace[i+1]
    return twin_part

def _corr_table(bcf, data):
    # returns absolute correlations table for chan*sample matrix
    corrs = np.empty((bcf.chan_n, bcf.neighb_n))
    picks = bcf.picks
    for chan in range(bcf.chan_n):
        for neighb_idx in range(bcf.neighb_n):
            corrs[chan,neighb_idx] = np.abs(
                    np.corrcoef(data[picks[chan],:],
                                data[picks[bcf.neighbs[chan][neighb_idx]],:])[0,1])
    return corrs

def _eog_var(lgr, sig, eogs):
    # calculates linear regression of EOG (or other) data against a signal,
    # returns the proportion of variance in the signal they account for.
    lgr.fit(eogs, sig)
    return lgr.score(eogs, sig)


class BadChannelFind():
    '''
    Class for identifying globally bad channels. Raw data are broken up into
    time windows of specified length. A channel which has a low correlation with
    its neighbours during a time window receives a no-vote from that time window.
    Channels whose no-votes across all time windows reach a certain threshold are
    determined as globally bad.
    '''

    def __init__(self, picks, neighb_n=4, thresh=0.85, vote_thresh=0.25, twin_len=10):
        '''
        picks: channel indices to examine

        neighb_n : the n nearest neighbours of an electrode for correlation
        calculations

        thresh: Correlations below this result in a no-vote for that channel
        in that time window

        vote_thresh: channels which get a no-vote for more than this proportion
        of the data are marked as bad

        twin_len: length in seconds of the time windows to partition the raw data
        '''

        self.picks = picks
        self.thresh = thresh
        self.vote_thresh = vote_thresh
        self.twin_len = twin_len
        self.neighb_n = neighb_n
        self.chan_n = len(picks)

    def recommend(self, raw):
        # returns list of potential bad channels
        if not raw.preload:
            raw.load_data()
        self.pos = _get_pos(raw.info, self.picks)
        self.dist_mat = _get_chan_dists(self.pos)
        self.neighbs = _get_neighbs(self.dist_mat, self.neighb_n)
        twin_len_idx = np.array(self.twin_len*1000/raw.info['sfreq']).astype(int)
        linspace = np.floor(np.linspace(0,raw.n_times,twin_len_idx)).astype(int)
        twin_part = _twin_builder(linspace)
        self.twin_n = twin_part.shape[0]

        votes = np.zeros((self.chan_n, self.twin_n))
        for twin_idx in range(self.twin_n):
            data = raw[:,twin_part[twin_idx,0]:twin_part[twin_idx,1]][0]
            corrs = _corr_table(self, data)
            for chan_idx in range(self.chan_n):
                if np.max(corrs[chan_idx,:])<self.thresh:
                    votes[chan_idx,twin_idx] = 1
        recs = []
        for chan_idx in range(self.chan_n):
            if (np.sum(votes[chan_idx,:])/self.twin_n)>self.vote_thresh:
                recs.append(raw.ch_names[self.picks[chan_idx]])

        return recs

class Anoar():
    '''
    Class for identifying bad trial/channel points, bad trials, and removing
    or repairing them, ignoring certain sources of noise (EOG) if desired.
    '''
    def __init__(self, eog_picks, erp_trigs=[[0]], chan_thresh=0.1, p_thresh=0.999,
                 eog_sub=1, raw_time=2):
        '''
        eog_picks: indices of eog (or EKG, etc) channels

        erp_trigs: trigger indices to examine. Different ERPs or groups of ERPs
        can be considered separately. E.g. erp_trigs=[[1,2,3],[4,5]] will do two
        processes, one with all trials with the triggers 1,2,3, and another with
        4 and 5. This may be useful if you expect different conditions to have
        very different ERP averages, though keep in mind that the algorithm functions
        better with large numbers of trials.

        chan_thresh: Trials with more than this proportion of bad channels are
        thrown out entirely.

        p_thresh: Z-scores are calculated for all values of the NxC noise matrix.
        Values which are larger than this on the CDF of the Z distribution (i.e.
        in the xth percentile of noisyness) are marked as bad.

        eog_sub: Whether or not to subtract the variance correlated with the channels
        specified in eog_picks

        raw_time: in the case of dealing with raw data, how long in seconds are
        the partitions to temporarily divide the data into.

        '''

        self.eog_picks = eog_picks
        self.erp_trigs = erp_trigs
        self.chan_thresh = chan_thresh
        self.erp_n = len(erp_trigs)
        self.p_thresh = p_thresh
        self.eog_sub = eog_sub
        self.raw_time = raw_time
        self.lgr = LinearRegression()
        self.doRaw = 0

    def get_recs(self):
        return self.bad_trials, self.bad_inds

    def viz(self):
        # visualise the results of the recommend process
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        erp_n = len(self.ssq)
        fig = plt.figure()
        for ssq_idx in range(erp_n):
            normo = LogNorm()
            plt.subplot(erp_n, 3, ssq_idx*3+1)
            plt.title('SSD', fontsize=20)
            plt.imshow(self.ssq_pure[ssq_idx], norm=normo,
                       interpolation='none', aspect='auto')
            plt.axis('off')
            plt.subplot(erp_n, 3, ssq_idx*3+2)
            plt.title('SSD adjusted for EOG', fontsize=20)
            plt.imshow(self.ssq[ssq_idx], norm=normo,
                       interpolation='none', aspect='auto')
            plt.axis('off')
            plt.subplot(erp_n, 3, ssq_idx*3+3)
            plt.title('Data marked for removal', fontsize=20)
            plt.imshow(self.disp_mat[ssq_idx], interpolation='none',
                       aspect='auto')
            plt.axis('off')
        fig.text(0.47, 0.04, 'Channels', ha='center', fontsize=20)
        fig.text(0.04, 0.5, 'Trials', va='center', rotation='vertical', fontsize=20)
        fig.tight_layout()
        plt.show()
        return(fig)

    def recommend(self, indata):

        if isinstance(indata, mne.io.BaseRaw):
            doRaw = 1
            self.doRaw = 1
        elif isinstance(indata, mne.BaseEpochs):
            epochs = indata
            doRaw = 0
        else:
            raise ValueError("Data must be either raw or epoched")

        if doRaw:
            t_range = range(0,indata.n_times,int(self.raw_time*indata.info["sfreq"]))
            self.events = np.array(([(x, 0, 0) for x in t_range]))
            epochs = mne.Epochs(indata,self.events,tmin=0, tmax=self.raw_time,
                                baseline=None, picks=None, proj=False)

        p_thresh = self.p_thresh
        erp_trigs = self.erp_trigs
        eog_sub = self.eog_sub
        lgr = self.lgr

        data = epochs.get_data()
        picks_good = mne.pick_types(epochs.info, eeg=True)
        chan_good_n = len(picks_good)
        erp_n = len(erp_trigs)
        chan_n = data.shape[1]
        samp_n = data.shape[2]
        trial_n = np.zeros(erp_n, dtype=int)

        # produce ERPs for noise estimation
        evoks = np.empty((erp_n, chan_n, samp_n))
        trigs = epochs.events[:,2]
        for erp_idx, erp in enumerate(erp_trigs):
            erp_match = np.zeros(len(trigs), dtype=bool)
            for e in erp:
                erp_match += trigs==e
            if not doRaw: # for arbitrary lengths of raw data, expected val is 0
                evoks[erp_idx,:,:] = np.mean(data[erp_match,:,:],0)
            trial_n[erp_idx] = np.sum(erp_match)
        # error sum of squares for each trial
        orig_trial_idx = [np.zeros(trial_n[x], dtype=int) for x in range(erp_n)] # find your way back to original trial idx after data have been segregated into separate matrices
        ssq = [np.zeros((trial_n[x],chan_good_n)) for x in range(erp_n)] # sum of squares controlled for EOG
        ssq_pure = [np.zeros((trial_n[x],chan_good_n)) for x in range(erp_n)] # sum of squares
        disp_mat = [np.zeros((trial_n[x],chan_good_n), dtype=bool) for x in range(erp_n)]
        ssq_t_idx = np.zeros(erp_n, dtype=int)
        for trial_idx in range(np.sum(trial_n)):
            erp_type = np.where([trigs[trial_idx] in x for x in erp_trigs])[0] # identify which ERP group this trial belongs to
            if len(erp_type)!=1:
                raise ValueError('Trial doesn''t match group or matches multiple groups.')
            erp_type = erp_type[0]
            for chan_idx, chan in enumerate(picks_good):
                rsq = 0
                # get r² for EOG (or anything else), assign them to respective SSQ matrix (one for each ERP group)
                if eog_sub:
                    rsq = _eog_var(lgr,data[trial_idx,chan,:].reshape(-1,1), data[trial_idx,self.eog_picks,:].T)
                # calculate sum of squares of ERPs, multiply by eog (1 - r²)
                ssq_pure[erp_type][ssq_t_idx[erp_type],chan_idx] = np.sum(((
                        data[trial_idx,chan,:]-evoks[erp_type,chan,:]))**2)
                ssq[erp_type][ssq_t_idx[erp_type],chan_idx] =  ssq_pure[
                        erp_type][ssq_t_idx[erp_type],chan_idx] * (1-rsq)
            orig_trial_idx[erp_type][ssq_t_idx[erp_type]] = trial_idx
            ssq_t_idx[erp_type] += 1

        # find globally bad trials and bad channels within trials
        bad_trials = []
        bad_inds = [np.zeros(0,dtype=int) for x in range(2)]
        for ssq_idx, ssq_mat in enumerate(ssq):
            # individually bad channels within trials
            ssq_p = norm.cdf(zscore(ssq_mat))
            treff = ssq_p > p_thresh
            inds = np.where(treff)

            # global bad trials
            unique_trials = np.unique(inds[0])
            for trial in unique_trials:
                bad_chs_idx = inds[1][np.where(inds[0]==trial)]
                if len(bad_chs_idx)>(len(picks_good)*self.chan_thresh):
                    bad_trials.append(trial)
                    temp_inds = np.where(inds[0]==trial) # get rid of these individual points now that trial is out
                    inds = (np.delete(inds[0], temp_inds), np.delete(inds[1], temp_inds))
                    disp_mat[ssq_idx][trial, :] = 1

            bad_inds[0] = np.concatenate([bad_inds[0], orig_trial_idx[ssq_idx][inds[0]]])
            bad_inds[1] = np.concatenate([bad_inds[1], picks_good[inds[1]]])
            disp_mat[ssq_idx][inds[0],[inds[1]]] = 1

        self.bad_trials = bad_trials
        self.bad_inds = bad_inds
        self.ssq = ssq
        self.ssq_pure = ssq_pure
        self.disp_mat = disp_mat

    def clean(self, indata):

        bad_inds = self.bad_inds
        bad_trials = self.bad_trials
        unique_trials = np.unique(bad_inds[0])

        if self.doRaw:
            if not isinstance(indata, mne.io.BaseRaw):
                raise ValueError("Data must be raw.")
            raw = indata
            events = self.events
            raw_time = self.raw_time
            sfreq = raw.info["sfreq"]

            for trial in unique_trials:
                bad_chs_idx = bad_inds[1][np.where(bad_inds[0]==trial)]
                bad_chs = [raw.ch_names[x] for x in bad_chs_idx]
                tmin = int(events[trial,0])
                tmax = int(events[trial,0]+(raw_time*sfreq))
                raw_crop = raw.copy().crop(tmin=tmin/sfreq,tmax=tmax/sfreq)
                raw_crop.info["bads"] = bad_chs
                raw_crop.interpolate_bads()
                raw._data[:,tmin:tmax+1] = raw_crop._data
            bad_trial_starts = events[bad_trials,0]//sfreq
            if bad_trial_starts.size:
                duration = np.repeat(raw_time,len(bad_trial_starts))
                annotations = mne.Annotations(bad_trial_starts,duration,"bad")
                raw.annotations = annotations

            return raw


        else:
            if not isinstance(indata, mne.BaseEpochs):
                raise ValueError("Data must be epoched.")
            epochs = indata
            epochs_clean = epochs.copy()
            if not epochs_clean.preload: epochs_clean.load_data()

            for trial in unique_trials:
                bad_chs_idx = bad_inds[1][np.where(bad_inds[0]==trial)]
                bad_chs = [epochs.ch_names[x] for x in bad_chs_idx]
                epoch = epochs_clean[trial]
                epoch.info['bads'] = bad_chs
                epoch.interpolate_bads()
                epochs_clean._data[trial,:,:] = epoch._data
            epochs_clean.drop(bad_trials)

            return epochs_clean
