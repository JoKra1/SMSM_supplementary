import numpy as np
import scipy as scp
import pandas as pd
from itertools import repeat
import copy
from collections.abc import Callable
from mssm.models import (
    Gamma,
    GAMLSSFamily,
    GSMMFamily,
    Link,
    Formula,
    HSMMFamily,
    lhs,
    Gaussian,
    GAMM,
    GAMMLSS,
    GAMMALS,
    GSMM,
    Identity,
    DifferencePenalty,
    f,
    LOG,
    MULNOMLSS,
    i,
)
from mssm.src.python.repara import reparam
from mssm.src.python.custom_types import DerivOrder, SamplerResult
from mssm.src.python.matrix_solvers import map_csc_to_eigen, cpp_solve_coef
from mssm.src.python.hsmm import (
    _split_matrices,
    _compute_series_probs,
)
from mssm.src.python.discrete import DiscreteModelMatrix

from mssmViz.plot import plot, plot_fitted
import multiprocessing as mp
import networkx as nx
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import mne
from mne.datasets import sample
import math
from tqdm import tqdm
from mssm.src.python.utils import estimateVp
from mssm.src.python.formula import build_model_matrix, build_penalties
from mssm.src.python.matrix_solvers import translate_sparse
from mssm.src.python.mcmc import check_convergence
from mssm.src.python.gamm_solvers import compute_S_emb_pinv_det
from mssmViz.sim import sim18
import warnings
import os.path as op
import os


def event_shape(sfreq, event_width):
    """
    Computes the template of a half-sine (event) with given frequency f and sampling frequency.

    Taken and adapted from patterns.py in the hmp code.

    References:
     - hmp code, see:\
        https://github.com/GWeindel/hmp/blob/devel/hmp/patterns.py#L74 and\
        https://github.com/GWeindel/hmp/blob/devel/hmp/patterns.py#L45
    """
    steps = 1000 / sfreq
    event_width_samples = int(np.round(event_width / steps))
    event_idx = np.arange(event_width_samples) * steps + steps / 2
    event_frequency = 1000 / (
        event_width * 2
    )  # gives event frequency given that events are defined as half-sines
    template = np.sin(
        2 * np.pi * event_idx / 1000 * event_frequency
    )  # event morph based on a half sine with given event width and sampling frequency

    return template


def get_hmp_event_probs_series(
    fam: HSMMFamily,
    series: int,
    coef: np.ndarray,
    coef_split_idx: list[int],
    ys: list[np.ndarray | None],
    Xs: list[scp.sparse.csc_array | np.ndarray | DiscreteModelMatrix | None],
) -> np.ndarray:
    """Computes the probability of each event at every time-point for trial ``series``.

    :param fam: Family of the model
    :type fam: HSMMFamily
    :param coef: The current coefficient estimate (as np.array of shape (-1,1) - so it must not
        be flattened!).
    :type coef: np.ndarray
    :param coef_split_idx: A list used to split (via :func:`np.split`) the ``coef`` into the
        sub-sets associated with each paramter of the llk.
    :type coef_split_idx: list[int]
    :param ys: List containing the vectors of observations (each of shape (-1,1)) passed as
        ``lhs.variable`` to the formulas. **Note**: by convention ``mssm`` expectes that the
        actual observed data is passed along via the first formula (so it is stored in
        ``ys[0]``). If multiple formulas have the same ``lhs.variable`` as this first formula,
        then ``ys`` contains ``None`` at their indices to save memory.
    :type ys: [np.ndarray or None]
    :param Xs: A list of sparse model matrices per likelihood parameter. Can contain None for
        indices for which model matrix should not be formed explicitly.
    :type Xs: list[scp.sparse.csc_array | np.ndarray | DiscreteModelMatrix | None]
    """
    if fam.fast_hmp is False:
        raise ValueError(
            "Event probabilities can only be extracted for fast HMP models."
        )
    # Extract families, transition matrices, etc.
    n_S = fam.llkargs[0]
    obs_fams = fam.llkargs[1]
    d_fams = fam.llkargs[2]
    D = fam.llkargs[5]
    M = fam.llkargs[6]
    starts_with_first = fam.llkargs[7]
    build_mat_idx = fam.llkargs[11]
    shared_pars = fam.llkargs[12]
    shared_m = fam.llkargs[13]
    T = fam.llkargs[14]
    pi = fam.llkargs[15]
    scale = fam.llkargs[17]
    event_template = fam.llkargs[18]
    hmp_fam = fam.llkargs[19]
    is_hmp = fam.is_hmp
    fix_T_pi = fam.fix_T_pi
    tvdtpi = fam.tvdtpi

    flats = None
    rho = None

    # Extract some extra info and define storage
    n_T = len(ys[0])  # Number of time-points in this series
    event_width = None if event_template is None else len(event_template)

    # We can pre-compute the bs and ds - i.e., observation probabilities and
    # duration probabilities
    cross_corr, mus, _, _, _, bs, ds, Ts, pis = _compute_series_probs(
        coef,
        coef_split_idx,
        shared_pars,
        fam.shared_j,
        shared_m,
        ys,
        Xs,
        n_S,
        obs_fams,
        d_fams,
        D,
        M,
        event_width,
        event_template,
        scale,
        False,
        False if fix_T_pi else True,
        False if fix_T_pi else True,
        starts_with_first,
        build_mat_idx,
        is_hmp,
        hmp_fam,
        rho,
        tvdtpi,
        fam.fast_hmp,
        False,
        flats,
    )

    ds[np.isnan(ds) | np.isinf(ds)] = 0

    # The following computes the forward and backward variables of a HMP model as described by
    # Weindel, van Maanen and Borst (2024) and then uses these to compute the event probabilities.
    # Code taken & modified from:
    # https://github.com/GWeindel/hmp/blob/devel/hmp/models/event.py

    n_events = (n_S - 1) // 2
    n_events_full = n_events
    n_flats = n_events + 1
    gains = np.zeros((n_T, n_events), dtype=np.float64)

    for i in range(cross_corr.shape[1]):
        # computes the gains, i.e. the log of the likelihood ratio between
        # event present and absent.
        gains = gains + cross_corr[:, [i]] * mus[:, i, :] - mus[:, i, :] ** 2 / scale
    gains = np.exp(gains)
    gains_b = gains[::-1, ::-1]

    # pmf for each flat with appropriate zeroing to prevent overlap between events
    pmf = np.zeros([n_T, n_flats], dtype=np.float64)
    end = min(n_T, D - 1)

    # Figure out which transitions are actually happening on this trial.
    # Needs translation from bump/flat world to event/stage world.
    n_events, events, stages, bumps, flats = fam._group_hmp_events(series)

    # Handle first stage here, rest in loop over transitions below
    pmf[:end, stages[0]] = ds[:end, stages[0]]
    pmf[: fam.hmp_location[stages[0]], stages[0]] = 0

    # Loop over transitions
    for stage in stages[1:]:
        pmf[:end, stage] = ds[:end, stage]
        pmf[: fam.hmp_location[stage], stage] = 0
    pmf_b = pmf[:, ::-1]

    # Now modified forward pass
    forward = np.zeros((n_T, n_events), dtype=np.float64)
    backward = np.zeros((n_T, n_events), dtype=np.float64)
    forward[:, 0] = pmf[:, stages[0]] * gains[:, events[0]]
    backward[:, 0] = pmf_b[:, stages[0]]

    for evidx in np.arange(1, n_events):
        # convolution between pmf and previous forward multiplied by gain
        forward[:, evidx] = (
            np.convolve(forward[:, evidx - 1], pmf[:, stages[evidx]])[:n_T]
            * gains[:, events[evidx]]
        )
        backward[:, evidx] = np.convolve(
            backward[:, evidx - 1] * gains_b[:, events[evidx - 1]],
            pmf_b[:, stages[evidx]],
        )[:n_T]

    # Revert backward variable
    backward = backward[:, ::-1]
    backward = backward[::-1, :]

    # Compute event_probs
    event_probs = forward * backward
    event_probs = np.clip(event_probs, 0, None)
    event_probs = event_probs / event_probs.sum(axis=0)
    event_probs[np.isnan(event_probs)] = 0

    # And embed in n_events_full columns
    event_probs_full = np.zeros((n_T, n_events_full), dtype=np.float64)
    event_probs_full[:, events] = event_probs

    return event_probs_full


def get_hmp_event_probs(
    fam: HSMMFamily,
    coef: np.ndarray,
    coef_split_idx: list[int],
    ys: list[np.ndarray | None],
    Xs: list[scp.sparse.csc_array | np.ndarray | DiscreteModelMatrix | None],
) -> np.ndarray:
    """Computes the probability of each event at every time-point for every trial.

    :param fam: Family of the model
    :type fam: HSMMFamily
    :param coef: The current coefficient estimate (as np.array of shape (-1,1) - so it must not
        be flattened!).
    :type coef: np.ndarray
    :param coef_split_idx: A list used to split (via :func:`np.split`) the ``coef`` into the
        sub-sets associated with each paramter of the llk.
    :type coef_split_idx: list[int]
    :param ys: List containing the vectors of observations (each of shape (-1,1)) passed as
        ``lhs.variable`` to the formulas. **Note**: by convention ``mssm`` expectes that the
        actual observed data is passed along via the first formula (so it is stored in
        ``ys[0]``). If multiple formulas have the same ``lhs.variable`` as this first formula,
        then ``ys`` contains ``None`` at their indices to save memory.
    :type ys: [np.ndarray or None]
    :param Xs: A list of sparse model matrices per likelihood parameter. Can contain None for
        indices for which model matrix should not be formed explicitly.
    :type Xs: list[scp.sparse.csc_array | np.ndarray | DiscreteModelMatrix | None]
    """
    if fam.fast_hmp is False:
        raise ValueError(
            "Event probabilities can only be extracted for fast HMP models."
        )
    # Extract families, transition matrices, etc.
    n_S = fam.llkargs[0]
    obs_fams = fam.llkargs[1]
    d_fams = fam.llkargs[2]
    sid = fam.llkargs[3]
    tid = fam.llkargs[4]
    M = fam.llkargs[6]
    starts_with_first = fam.llkargs[7]
    shared_pars = fam.llkargs[12]
    shared_m = fam.llkargs[13]
    Lrhoi = fam.llkargs[16]

    n_series = len(sid)
    fix_T_pi = fam.fix_T_pi

    # Initialize potential None arguments
    if tid is None:
        tid = sid

    # Split up ys, Xs
    split_Ys, split_Xs, _ = _split_matrices(
        fam.cross_cor,
        Xs,
        shared_pars,
        shared_m,
        obs_fams,
        d_fams,
        sid,
        tid,
        n_S,
        M,
        False if fix_T_pi else True,
        False if fix_T_pi else True,
        starts_with_first,
        None if fam.fast_hmp else Lrhoi,
    )

    max_len = np.max([len(split_Ys[series][0]) for series in range(n_series)])

    # Now compute event probabilities
    ev_probs = []
    for series in range(n_series):
        with warnings.catch_warnings():  # Supress warnings
            warnings.simplefilter("ignore")
            ev_probss = get_hmp_event_probs_series(
                fam,
                series,
                coef,
                coef_split_idx,
                split_Ys[series],
                split_Xs[series],
            )

            # Embed into max len
            ev_probss = np.vstack(
                (
                    ev_probss,
                    np.zeros((max_len - ev_probss.shape[0], ev_probss.shape[1])),
                )
            )

            ev_probs.append(ev_probss)

    return np.array(ev_probs)


def simulateSources(
    sources,
    n_trials,
    n_jobs,
    file,
    relations=None,
    data_type="eeg",
    n_subj=1,
    path="./",
    overwrite=False,
    verbose=False,
    noise=True,
    times=None,
    seed=None,
    sfreq=100,
    save_snr=False,
    save_noiseless=False,
    event_length_samples=None,
    proportions=None,
):
    """
    Simulates n_trials of EEG and/or MEG using MNE's tools based on the specified sources

    Taken from the hmp code and adapted slightly to permit simulating non-linear covariate +
    random effects on the amplitude of each event and the expected flat durations.

    References:
     - hmp code, see https://github.com/GWeindel/hmp/blob/devel/hmp/simulations.py#L51 (might have\
        been updated)

    Parameters
    ----------
    sources : list
        2D or 3D list with dimensions (n_subjects *) sources * source_parameters
        Source parameters should contain :
        - the name of the source (see the output of available_source())
        - the duration of the event (in frequency, usually 10Hz)
        - the amplitude or strength of the signal from the source, expressed in volt (e.g. 1e-8 V)
        - the duration of the preceding stage as a scipy.stats distribution (e.g.\
            scipy.stats.gamma(a, scale))
    n_trials: int
        Number of trials
    n_jobs: int
        Number of jobs to use with MNE's function (multithreading)
    file: str
        Name of the file to be saved (number of the subject will be added)
    relations: list
        list of int describing to which previous event is each event connected, 1 means stimulus, 2
        means one event after stimulus, ... One event cannot be connected to an upcoming one
    data_type: str
        Whether to simulate "eeg" or "meg"
    n_subj: int
        How many subjects to simulate
    path: str
        path where to save the data
    overwrite: bool
        Whether to overwrite existing file
    verbose: bool
        Whether to display MNE's output
    noise: bool
        Adding noise to the simulated sources
    times: ndarray
        Deterministic simulation of event transitions times. Format is n_sources X n_trials
    location: float
        value in ms to add after a simulated event and before another one
    save_snr: bool
        Save the signal at peak value and electrode noise
    Returns
    -------
    generating_events: ndarray
        Times of the simulated events used to test for accurate recovery compared to estimation
    files: list
        list of file names (file + number of subject)
    """
    if not verbose:
        mne.set_log_level("warning")
    else:
        mne.set_log_level(True)
    if seed is not None:
        random_state = np.random.RandomState(seed)
    else:
        random_state = np.random.RandomState(np.random.randint(low=0, high=3000))
    sources = np.array(sources, dtype=object)
    if len(np.shape(sources)) == 2:
        sources = [sources]  # If only one subject
    if relations is None:
        relations = np.arange(len(sources[0]) + 1) + 1
    if proportions is None:
        proportions = np.repeat(1, len(sources[0]) + 1)
    if np.shape(sources)[0] != n_subj:
        raise ValueError("Number of subject is not coherent with the sources provided")

    # Following code and comments largely comes from MNE examples (e.g. \
    # https://mne.tools/stable/auto_examples/simulation/simulated_raw_data_using_subject_anatomy.html)
    # It loads the data, info structure and forward solution for one example subject,
    # Note that all 'subject' will use this forward solution
    data_path = sample.data_path()
    subjects_dir = op.join(data_path, "subjects")
    subject = "sample"
    # First, we get an info structure from the test subject.
    evoked_fname = op.join(data_path, "MEG", subject, "sample_audvis-ave.fif")
    # mne read raw
    info = mne.io.read_info(evoked_fname, verbose=verbose)
    # To simulate sources, we also need a source space. It can be obtained from the
    # forward solution of the sample subject.
    fwd_fname = op.join(
        data_path, "MEG", subject, "sample_audvis-meg-eeg-oct-6-fwd.fif"
    )
    fwd = mne.read_forward_solution(fwd_fname, verbose=verbose)
    with info._unlock():
        info["sfreq"] = sfreq
    if data_type == "eeg":
        picked_type = mne.pick_types(info, meg=False, eeg=True)
        fwd = mne.pick_types_forward(fwd, meg=False, eeg=True)
    elif data_type == "meg":
        picked_type = mne.pick_types(info, meg=True, eeg=False)
        fwd = mne.pick_types_forward(fwd, meg=True, eeg=False)
    elif data_type == "eeg/meg":
        picked_type = mne.pick_types(info, meg=True, eeg=True)
        fwd = mne.pick_types_forward(fwd, meg=True, eeg=True)
    else:
        raise ValueError(
            f'Invalid data type {data_type}, expected "eeg", "meg" or "eeg/meg"'
        )
    src = fwd["src"]
    info = mne.pick_info(info, picked_type)
    tstep = 1.0 / info["sfreq"]  # sample duration

    # For each subject, simulate associated sources
    files = []
    for subj in range(n_subj):
        sources_subj = sources[subj]
        # Pre-allocate the random times to avoid generating too long 'recordings'

        rand_times = np.zeros((len(sources_subj), n_trials))
        random_indices_list = []
        for s, source in enumerate(sources_subj):
            # How many trials wil have the event, default is all
            props_trial = int(np.round(n_trials * proportions[s]))
            if proportions[s] < 1:
                random_indices_list.append(
                    random_state.choice(
                        np.arange(n_trials), size=props_trial, replace=False
                    )
                )
            else:
                random_indices_list.append(np.arange(n_trials))
            # print(random_indices_list[-1],source[-1])
            rand_times[s, random_indices_list[-1]] = source[
                -1
            ]  # .rvs(size=len(random_indices_list[-1]), random_state=random_state)
        seq_index = np.diff(relations, prepend=0) > 0
        if seq_index.all():  # If all events are sequential
            trial_time = np.sum(rand_times, axis=0)
        else:
            trial_time_seq = np.sum(rand_times[seq_index], axis=0)
            trial_time_nonseq = np.sum(rand_times[~seq_index], axis=0)
            trial_time = np.maximum(trial_time_seq, trial_time_nonseq)
        trial_time /= 1000  # to seconds
        trial_time[1:] += trial_time[:-1] + 0.5
        trial_time = np.cumsum(trial_time)
        # Build simulator
        files_subj = []
        source_simulator = mne.simulation.SourceSimulator(
            src, tstep=tstep, first_samp=0, duration=trial_time[-1] + 10
        )  # add 10sec to the end of the last trial
        if n_subj == 1:
            subj_file = file + "_raw.fif"
        else:
            subj_file = file + f"_{subj}_raw.fif"
        if subj_file in os.listdir(path) and not overwrite:
            subj_file = path + subj_file
            warnings.warn(
                f"{subj_file} exists no new simulation performed", UserWarning
            )
            files_subj.append(subj_file)
            files_subj.append(subj_file.split(".fif")[0] + "_generating_events.npy")
            if save_snr:
                files_subj.append(subj_file.split(".fif")[0] + "_snr.npy")
            if save_noiseless:
                files_subj.append(file + "_noiseless_raw.fif")
            files.append(files_subj)
        else:
            subj_file = op.join(path, subj_file)
            print(f"Simulating {subj_file}")
            # stim_onset occurs every x samples.
            events = np.zeros((n_trials, 3), int)
            stim_onsets = 1 + trial_time  # offset of first stim is 1 s
            events[:, 0] = stim_onsets / tstep  # last event
            events[:, 2] = 1  # trigger 1 = stimulus

            # Fake source, actually stimulus onset
            selected_label = mne.read_labels_from_annot(
                subject,
                regexp=(
                    sources_subj[0][0][0]
                    if isinstance(sources_subj[0][0], list)
                    else sources_subj[0][0]
                ),
                subjects_dir=subjects_dir,
                verbose=verbose,
            )[0]
            label = mne.label.select_sources(
                subject,
                selected_label,
                subjects_dir=subjects_dir,
                random_state=random_state,
            )
            source_time_series = np.array([1e-20])  # stim trigger
            source_simulator.add_data(label, source_time_series, events)

            trigger = 2
            generating_events = events
            for s, source in enumerate(sources_subj):

                if event_length_samples is None:
                    event_duration = int(((1 / source[1]) / 2) * info["sfreq"])
                    # Shift the trigger for the simulations
                    shift = event_duration // 2
                else:
                    event_duration = event_length_samples[s]
                    # Assumes the shortest event duration is a half-sine
                    shift = min(event_length_samples) // 2

                # adding source event, take as previous time the event defined by relation (default
                # is previous event)
                events = generating_events[
                    generating_events[:, -1] == relations[s]
                ].copy()
                random_indices = random_indices_list[s]
                if times is None:
                    # Always at least one sample later
                    rand_i = np.maximum(1, rand_times[s] / (tstep * 1000))
                    rand_i = np.round(rand_i, decimals=0)
                else:
                    rand_i = times[:, trigger - 2] / (tstep * 1000)
                if len(rand_i[rand_i < 0]) > 0:
                    warnings.warn(
                        (
                            "Negative stage duration were found, 1 is imputed "
                            f"for the {len(rand_i[rand_i<0])} trial(s)"
                        ),
                        UserWarning,
                    )
                    rand_i[rand_i < 0] = 1
                events[random_indices, 0] = (
                    events[random_indices, 0] + rand_i[random_indices]
                )  # Events sample.
                events[:, 2] = trigger  # All events have the sample id.

                generating_events = np.concatenate([generating_events, events.copy()])
                # Shift event to onset when simulating pattern
                events[random_indices, 0] = events[random_indices, 0] - shift
                # add these events

                source_it = (
                    range(len(source[0])) if isinstance(source[0], list) else range(1)
                )
                for source_idx in source_it:
                    target_source = (
                        source[0][source_idx]
                        if isinstance(source[0], list)
                        else source[0]
                    )
                    # print(s, target_source)
                    selected_label = mne.read_labels_from_annot(
                        subject,
                        regexp=target_source,
                        subjects_dir=subjects_dir,
                        verbose=verbose,
                    )[0]
                    label = mne.label.select_sources(
                        subject,
                        selected_label,
                        subjects_dir=subjects_dir,
                        location=0,
                        grow_outside=False,
                        random_state=random_state,
                    )
                    # last two parameters ensure sources that are different enough
                    # Define the time course of the activity for each source of the region to
                    # activate

                    amplitude = (
                        source[2][source_idx]
                        if isinstance(source[0], list)
                        else source[2]
                    )

                    if trigger == len(sources_subj) + 1:
                        amplitude = (
                            1e-20  # Last source defines RT and is not an event per se
                        )

                    if isinstance(amplitude, np.ndarray):
                        if len(amplitude.shape) == 1:
                            amplitude = amplitude.reshape(-1, 1)
                        source_time_series = (
                            event_shape(
                                info["sfreq"], event_duration * 1000 / info["sfreq"]
                            ).reshape(1, -1)
                            * amplitude
                        )
                    else:
                        source_time_series = (
                            event_shape(
                                info["sfreq"], event_duration * 1000 / info["sfreq"]
                            )
                            * amplitude
                        )

                    source_simulator.add_data(
                        label, source_time_series, events[random_indices]
                    )
                trigger += 1

            generating_events = generating_events[generating_events[:, 0].argsort()]
            # Project the source time series to sensor space and add some noise. The source
            # simulator can be given directly to the simulate_raw function.
            raw = mne.simulation.simulate_raw(
                info, source_simulator, forward=fwd, n_jobs=n_jobs, verbose=verbose
            )

            n_events = len(sources_subj) - 1
            if save_noiseless:
                raw.save(file + "_noiseless_raw.fif", overwrite=True)
            if data_type == "eeg":
                raw = raw.pick_types(meg=False, eeg=True, stim=True)
            elif data_type == "meg":
                raw = raw.pick_types(meg=True, eeg=False, stim=True)
            elif data_type == "eeg/meg":
                raw = raw.pick_types(meg=True, eeg=True, stim=True)
            if save_snr:
                snr = np.zeros((len(info["ch_names"]), n_events))
                data = copy.deepcopy(raw.get_data())
                for event in range(n_events):
                    times_out = generating_events[
                        generating_events[:, 2] == event + 2, 0
                    ]
                    snr[:, event] = np.mean((data[:, times_out]) ** 2, axis=-1)
            if noise:
                cov = mne.make_ad_hoc_cov(raw.info, verbose=verbose)
                mne.simulation.add_noise(
                    raw,
                    cov,
                    verbose=verbose,
                    iir_filter=[0.2, -0.2, 0.04],
                    random_state=random_state,
                )

            raw.save(subj_file, overwrite=True)
            files_subj.append(subj_file)
            np.save(
                subj_file.split(".fif")[0] + "_generating_events.npy", generating_events
            )
            files_subj.append(subj_file.split(".fif")[0] + "_generating_events.npy")
            if save_snr:
                data = raw.get_data()
                for event in range(n_events):
                    times_out = generating_events[
                        generating_events[:, 2] == event + 2, 0
                    ]
                    snr[:, event] /= np.var(data[:, times_out], axis=-1)
                np.save(subj_file.split(".fif")[0] + "_snr.npy", snr)
                files_subj.append(subj_file.split(".fif")[0] + "_snr.npy")
            files.append(files_subj)
            if save_noiseless:
                files_subj.append(file + "_noiseless_raw.fif")
            print(f"{subj_file} simulated")

    if n_subj == 1:
        files = files[0]
    return files


def sim6(
    n_events: int,
    n_participants: int,
    n_trials: int,
    source_names: list[str],
    mus: list[float],  # Means of the gamma distribution
    scale: float = 0.5,
    s_scale: float = 0.5,
    frequency: float = 10.0,  # Frequency of the event defining its duration, half-sine of 10Hz
    base_amplitude: float = 0.2e-7,
    seed: int | None = 2026,
    sfreq: int = 100,
    cpus: int = 1,
    file: str = "dataset_raw",
):
    # Generate sources
    sources = []
    xs = []
    true_durs = np.zeros((n_events + 1, n_participants, n_trials))

    # Simulate effect of x on expected duration of each stage
    np_gen = np.random.default_rng(seed)

    source_seed = seed
    for participant in range(n_participants):
        # Generate covariates influencing events
        x = np.sort(np_gen.random(size=n_trials))

        xs.append(x)
        sources.append([])

        # sources are mixtured
        for si, names in enumerate(source_names):

            if si < n_events:

                n_classes = len(names)
                sim_dat = sim18(5000, 2, c=0, seed=10, correlate=False, n_ranef=2)

                sim_dat.loc[:, "y"] = np_gen.integers(
                    0, high=n_classes, size=sim_dat.shape[0]
                )

                sim_formulas = []
                for k in range(n_classes - 1):
                    terms = [i(), f(["x0"])]
                    sim_formulas.append(Formula(lhs("y"), terms, data=sim_dat))

                model = GAMMLSS(sim_formulas, MULNOMLSS(n_classes - 1))
                model.fit(method="Chol", progress_bar=False)

                for lam in model.overall_penalties:
                    lam.lam = s_scale

                S_emb, _, _, _ = compute_S_emb_pinv_det(
                    model.coef.shape[0], model.overall_penalties, "svd", root=False
                )

                eigV, U = scp.linalg.eigh(S_emb.toarray())

                # Enforce PSD
                eigV[eigV < 1e-7] = 0
                ieigV = np.zeros_like(eigV)
                ieigV[eigV > 0] = np.sqrt(1 / eigV[eigV > 0])

                # Get root
                lvi = np.diag(ieigV) @ U.T
                model.lvi = lvi

                coefs = []
                for par in range(n_classes - 1):
                    coef_par = model.sample_post(
                        1, par=par, deviations=True, seed=source_seed
                    )
                    coefs.append(coef_par)
                coefs = np.concatenate(coefs, axis=0)

                model.coef = coefs
                pred_dat = pd.DataFrame({"x0": x, "x1": x, "x2": x, "x3": x})
                pmus = []
                for k in range(n_classes - 1):

                    eta, _, _ = model.predict([0, 1], pred_dat, ci=False, par=k)
                    mu = model.family.links[k].fi(eta)
                    pmus.append(mu.reshape(-1, 1))

                probs = model.family.get_probs(*pmus)

                amplitudes = []
                for k in range(n_classes):
                    plt.plot(probs[:, k])
                    sign = np_gen.random()
                    if sign < 0.5:
                        sign = -1
                    else:
                        sign = 1
                    amplitudes.append(sign * base_amplitude * probs[:, k])
                plt.show()
            else:
                amplitudes = 1e-7

            # Now stage durations
            alpha = 1 / scale
            beta = alpha / mus[si]

            # duration samples
            dur = scp.stats.gamma.rvs(
                a=alpha,
                scale=(1 / beta),
                size=n_trials,
                random_state=seed + participant,
            )

            dur[dur < 50] = 70

            true_durs[si, participant, :] = dur

            sources[-1].append([names, frequency, amplitudes, dur])
            source_seed += 1

    # Function used to generate the data
    file = simulateSources(
        sources,
        n_trials,
        cpus,
        file,
        n_subj=n_participants,
        overwrite=True,
        sfreq=sfreq,
        seed=seed,
        path="./",
    )
    return true_durs, xs, file


def sim5(
    n_events: int,
    n_participants: int,
    n_trials: int,
    source_names: list[str],
    mus: list[float],  # Means of the gamma distribution
    scale: float = 0.5,
    fs_scale: float = 0,
    frequency: float = 10.0,  # Frequency of the event defining its duration, half-sine of 10Hz
    base_amplitude: float = 0.2e-7,
    seed: int | None = 2026,
    sfreq: int = 100,
    cpus: int = 1,
    force_group_level: bool = False,
    prevent_sign_flip: bool = False,
    file: str = "dataset_raw",
):
    # Generate sources
    sources = []
    xs = []
    true_durs = np.zeros((n_events + 1, n_participants, n_trials))

    # Simulate effect of x on expected duration of each stage
    np_gen = np.random.default_rng(seed)

    # Set up random smooth sampler
    c = max(1e-7, fs_scale)

    # Set up random smooth sampler for smooth of x0
    # Based on prior assumption discussed by Wood (2017)
    x = np_gen.random(size=1000)
    fs_dat = pd.DataFrame({"x": x, "y": np_gen.random(1000)})

    fs_formula = Formula(
        lhs=lhs("y"),
        terms=[
            f(
                ["x"],
                identifiable=False,
                nk=10,
                penalty=[DifferencePenalty()],
                pen_kwargs=[{"m": 1}],
            )
        ],
        data=fs_dat,
    )

    fs_model = GAMM(fs_formula, Gaussian())
    fs_pen = build_penalties(fs_formula)

    mmat = fs_model.get_mmat()
    fs_model.coef = np_gen.random(mmat.shape[1]).reshape(-1, 1)
    cov = fs_formula.cov_flat
    S = fs_pen[0].S_J

    C, Srp, Drp, IRrp, rms1, rms2, rp_rank = reparam(
        mmat, S, cov, identity=True, scale=False, QR=True
    )

    pred_dat = pd.DataFrame({"x": np.linspace(0, 1, 30)})
    _, pred_mat, _ = fs_model.predict(None, n_dat=pred_dat)
    pred_mat = pred_mat @ C
    pred_mat[:, [-1]] = 1
    etas = np.zeros((n_participants, n_events, 30))

    V = (Srp / (c * 1e7)).toarray()
    source_seed = seed
    for participant in range(n_participants):
        # Generate covariates influencing events
        x = np_gen.random(size=n_trials)

        # Get prediction matrix for this participant
        pred_dat = pd.DataFrame({"x": x})
        _, mmat, _ = fs_model.predict(None, n_dat=pred_dat)
        mmat_RP = mmat @ C
        mmat_RP[:, [-1]] = 1

        xs.append(x)
        sources.append([])

        # per source we have multiple amplitudes (one for each trial)
        for si, name in enumerate(source_names):

            # Sample random smooth for this participant and source
            sample = scp.stats.multivariate_normal.rvs(
                mean=[0 for _ in range(Srp.shape[1])],
                cov=V,
                size=1,
                random_state=source_seed,
            )
            eta = mmat_RP @ sample

            if si < n_events:
                etas[participant, si, :] = pred_mat @ sample

            if force_group_level:
                eta = 0.9
                etas[participant, :, :] = 0.9
            elif si < n_events and prevent_sign_flip:
                direction = np_gen.random()
                step = 0.1
                if direction < 0.5:
                    while np.any((0.1 + eta) >= 0):
                        eta -= step
                        etas[participant, si, :] -= step
                else:
                    while np.any((0.1 + eta) <= 0):
                        eta += step
                        etas[participant, si, :] += step
                # plt.plot(etas[participant, si, :] + 0.1)
            amplitude = base_amplitude * (0.1 + eta)

            # Now stage durations
            alpha = 1 / scale
            beta = alpha / mus[si]

            # duration samples
            dur = scp.stats.gamma.rvs(
                a=alpha,
                scale=(1 / beta),
                size=n_trials,
                random_state=seed + participant,
            )

            dur[dur < 50] = 70

            true_durs[si, participant, :] = dur

            sources[-1].append([name, frequency, amplitude, dur])
            source_seed += 1
        # plt.show()

    # Function used to generate the data
    file = simulateSources(
        sources,
        n_trials,
        cpus,
        file,
        n_subj=n_participants,
        overwrite=True,
        sfreq=sfreq,
        seed=seed,
        path="./",
    )
    return true_durs, etas, xs, file


def sim4(
    n_events: int,
    n_participants: int,
    n_trials: int,
    source_names: list[str],
    mus: list[float],  # Means of the gamma distribution
    scale: float = 0.5,
    fs_scale: float = 0,
    frequency: float = 10.0,  # Frequency of the event defining its duration, half-sine of 10Hz
    base_amplitude: float = 0.2e-7,
    seed: int | None = 2026,
    sfreq: int = 100,
    cpus: int = 1,
    force_group_level: bool = False,
    file: str = "dataset_raw",
):
    # Generate sources
    sources = []
    xs = []
    true_durs = np.zeros((n_events + 1, n_participants, n_trials))

    # Simulate effect of x on expected duration of each stage
    np_gen = np.random.default_rng(seed)

    # Set up random smooth sampler
    c = max(1e-7, fs_scale)

    # Set up random smooth sampler for smooth of x0
    # Based on prior assumption discussed by Wood (2017)
    x = np_gen.random(size=1000)
    fs_dat = pd.DataFrame({"x": x, "y": np_gen.random(1000)})

    fs_formula = Formula(
        lhs=lhs("y"),
        terms=[
            f(
                ["x"],
                identifiable=False,
                nk=10,
                penalty=[DifferencePenalty()],
                pen_kwargs=[{"m": 1}],
            )
        ],
        data=fs_dat,
    )

    fs_model = GAMM(fs_formula, Gaussian())
    fs_pen = build_penalties(fs_formula)

    mmat = fs_model.get_mmat()
    fs_model.coef = np_gen.random(mmat.shape[1]).reshape(-1, 1)
    cov = fs_formula.cov_flat
    S = fs_pen[0].S_J

    C, Srp, Drp, IRrp, rms1, rms2, rp_rank = reparam(
        mmat, S, cov, identity=True, scale=False, QR=True
    )

    V = (Srp / (c * 1e7)).toarray()

    for participant in range(n_participants):
        # Generate covariates influencing events
        x = np_gen.random(size=n_trials)

        # Get prediction matrix for this participant
        pred_dat = pd.DataFrame({"x": x})
        _, mmat, _ = fs_model.predict(None, n_dat=pred_dat)
        mmat_RP = mmat @ C
        mmat_RP[:, [-1]] = 1

        # Sample random smooth for this participant
        sample = scp.stats.multivariate_normal.rvs(
            mean=[0 for _ in range(Srp.shape[1])],
            cov=V,
            size=1,
            random_state=seed + participant,
        )
        eta = mmat_RP @ sample

        if force_group_level:
            eta = 0.9

        xs.append(x)
        sources.append([])

        # per source we have multiple amplitudes (one for each trial)
        for si, name in enumerate(source_names):

            amplitude = base_amplitude * (0.1 + eta)

            # Now stage durations
            alpha = 1 / scale
            beta = alpha / mus[si]

            # duration samples
            dur = scp.stats.gamma.rvs(
                a=alpha,
                scale=(1 / beta),
                size=n_trials,
                random_state=seed + participant,
            )

            dur[dur < 50] = 70

            true_durs[si, participant, :] = dur

            sources[-1].append([name, frequency, amplitude, dur])

    # Function used to generate the data
    file = simulateSources(
        sources,
        n_trials,
        cpus,
        file,
        n_subj=n_participants,
        overwrite=True,
        sfreq=sfreq,
        seed=seed,
        path="./",
    )
    return true_durs, xs, file


def sim3(
    n_events: int,
    n_participants: int,
    n_trials: int,
    source_names: list[str],
    mus: list[float],  # Means of the gamma distribution
    scale: float = 0.5,
    rint_scale: float = 1,
    frequency: float = 10.0,  # Frequency of the event defining its duration, half-sine of 10Hz = 50ms
    base_amplitude: float = 0.2e-7,
    seed: int | None = 2026,
    sfreq: int = 100,
    cpus: int = 1,
    force_group_level: bool = False,
    file: str = "dataset_raw",
):
    # Generate sources
    sources = []
    xs = []
    true_durs = np.zeros((n_events + 1, n_participants, n_trials))

    # Simulate effect of x on expected duration of each stage
    np_gen = np.random.default_rng(seed)

    # Simulate random intercepts
    rind = scp.stats.norm.rvs(size=n_participants, scale=rint_scale, random_state=seed)

    for participant in range(n_participants):
        # Generate covariates influencing events
        x = np_gen.random(size=n_trials)
        xs.append(x)
        sources.append([])

        # per source we have multiple amplitudes (one for each trial)
        for si, name in enumerate(source_names):

            # Get event specific amplitude for each trial. Functions from Gu & Whaba (1991)
            # "Minimizing GCV/GML Scores with Multiple Smoothing Parameters via the Newton Method"
            if si == 0:
                fv = 2 * np.sin(np.pi * x)
                fv -= 1.5
            elif si == 1:
                fv = np.exp(2 * x)
                fv -= 3.5
            elif si == 2:
                fv = 0.2 * np.power(x, 11) * np.power(10 * (1 - x), 6) + 10 * np.power(
                    10 * x, 3
                ) * np.power(1 - x, 10)
                fv -= 3.5
            else:
                fv = np.zeros_like(x)

            eta = fv + rind[participant]
            if force_group_level:
                eta = 0.9
            amplitude = base_amplitude * (0.1 + eta)

            # Now stage durations
            alpha = 1 / scale
            beta = alpha / mus[si]

            # duration samples
            dur = scp.stats.gamma.rvs(
                a=alpha,
                scale=(1 / beta),
                size=n_trials,
                random_state=seed + participant,
            )

            dur[dur < 50] = 70

            true_durs[si, participant, :] = dur

            sources[-1].append([name, frequency, amplitude, dur])

    # Function used to generate the data
    file = simulateSources(
        sources,
        n_trials,
        cpus,
        file,
        n_subj=n_participants,
        overwrite=True,
        sfreq=sfreq,
        seed=seed,
        path="./",
    )
    return true_durs, xs, file


def sim2(
    n_events: int,
    n_participants: int,
    n_trials: int,
    source_names: list[str],
    mus: list[float],  # Means of the gamma distribution
    scale: float = 0.5,
    rint_scale: float = 1,
    frequency: float = 10.0,  # Frequency of the event defining its duration, half-sine of 10Hz = 50ms
    base_amplitude: float = 0.2e-7,
    seed: int | None = 2026,
    sfreq: int = 100,
    cpus: int = 1,
    force_group_level: bool = False,
    file: str = "dataset_raw",
):
    # Generate sources
    sources = []
    xs = []
    true_durs = np.zeros((n_events + 1, n_participants, n_trials))

    # Simulate effect of x on expected duration of each stage
    np_gen = np.random.default_rng(seed)

    # Simulate random intercepts
    rind = scp.stats.norm.rvs(size=n_participants, scale=rint_scale, random_state=seed)

    for participant in range(n_participants):
        # Generate covariates influencing events
        x = np_gen.random(size=n_trials)
        xs.append(x)
        sources.append([])

        # per source we have multiple amplitudes (one for each trial)
        for si, name in enumerate(source_names):

            # Get event shared amplitude for each trial. Function is from Gu & Whaba (1991)
            # "Minimizing GCV/GML Scores with Multiple Smoothing Parameters via the Newton Method"

            fv = 2 * np.sin(np.pi * x)
            fv -= 1.5

            eta = fv + rind[participant]
            if force_group_level:
                eta = 0.9
            amplitude = base_amplitude * (0.1 + eta)

            # Now stage durations
            alpha = 1 / scale
            beta = alpha / mus[si]

            # duration samples
            dur = scp.stats.gamma.rvs(
                a=alpha,
                scale=(1 / beta),
                size=n_trials,
                random_state=seed + participant,
            )

            dur[dur < 50] = 70

            true_durs[si, participant, :] = dur

            sources[-1].append([name, frequency, amplitude, dur])

    # Function used to generate the data
    file = simulateSources(
        sources,
        n_trials,
        cpus,
        file,
        n_subj=n_participants,
        overwrite=True,
        sfreq=sfreq,
        seed=seed,
        path="./",
    )
    return true_durs, xs, file


def sim1(
    n_events: int,
    n_trials: int,
    source_names: list[str],
    scale: float = 0.5,
    frequency: float = 10.0,  # Frequency of the event defining its duration, half-sine of 10Hz = 50ms
    amplitude: float = 0.2e-7,
    seed: int | None = 2026,
    sfreq: int = 100,
    cpus: int = 1,
    file: str = "dataset_raw",
):
    # Generate sources
    sources = []
    true_durs = np.zeros((n_events + 1, n_trials))

    # Simulate effect of x on expected duration of each stage
    np_gen = np.random.default_rng(seed)

    x = np_gen.random(size=n_trials)
    gam_dat = pd.DataFrame({"x": x})

    # One source = one frequency, one amplitude and a given by-trial variability distribution
    for si, name in enumerate(source_names):

        # Get mean of stage-specific Gamma distribution. Functions from Gu & Whaba (1991)
        # "Minimizing GCV/GML Scores with Multiple Smoothing Parameters via the Newton Method"
        if si == 0:
            fv = 0.5 * np.sin(np.pi * x) + 5
        elif si == 1:
            fv = 0.05 * np.exp(2 * x) + 5
        elif si == 2:
            fv = (
                0.2 * np.power(x, 11) * np.power(10 * (1 - x), 6)
                + 10 * np.power(10 * x, 3) * np.power(1 - x, 10)
            ) * 0.2 + 5
        else:
            fv = np.zeros_like(x) + 5

        mu = Gamma().link.fi(fv)

        # Convert to parameters scipy expects
        alpha = 1 / scale
        beta = alpha / mu

        # duration samples
        dur = scp.stats.gamma.rvs(
            a=alpha, scale=(1 / beta), size=n_trials, random_state=seed
        )

        dur[dur < 50] = 70

        # Collect durations
        true_durs[si, :] = dur
        gam_dat[f"durs_{si}"] = dur

        sources.append([name, frequency, amplitude, dur])

    # Function used to generate the data
    file = simulateSources(
        sources,
        n_trials,
        cpus,
        file,
        overwrite=True,
        sfreq=sfreq,
        seed=seed,
        path="./",
    )
    return true_durs, x, file


class GAMMA2(GAMLSSFamily):
    """Family for a GAMMA 2 model with an :math:`\\alpha` parameter of 2.

    This Family follows the :class:`Gamma` family, in that we assume:
    :math:`Y_i \\sim \\Gamma(\\mu_i,\\phi=0.5)`.

    References:
     - Anderson, J. R., Zhang, Q., Borst, J. P., & Walsh, M. M. (2016). The discovery of processing\
        stages: Extension of Sternberg’s method.
     - Rigby, R. A., & Stasinopoulos, D. M. (2005). Generalized Additive Models for Location,\
        Scale and Shape.
     - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth\
        Models.

    :param links: Link functions for the mean. Standard would be ``links=[LOG()]``.
    :type links: [Link]
    """

    def __init__(self, links: list[Link]) -> None:
        super().__init__(1, links)
        # All derivatives based on gamlss.dist: https://github.com/gamlss-dev/gamlss.dist, but
        # adjusted so that \\phi (the scale) is equal to 0.5
        # see also: Rigby, R. A., & Stasinopoulos, D. M. (2005). Generalized Additive Models for
        # Location, Scale and Shape.

    def dpars(
        self, y: np.ndarray, *mus: np.ndarray, index: int, order: DerivOrder
    ) -> np.ndarray:
        """Returns partial derivatives of the log-likelihood with respect to the mean and scale
        or a combination indexed by ``index`` of ``order`` (first order, pure second,
        mixed second).

        All derivatives taken from gamlss.dist: https://github.com/gamlss-dev/gamlss.dist see
        also: Rigby, R. A., & Stasinopoulos, D. M. (2005).

        References:
         - Rigby, R. A., & Stasinopoulos, D. M. (2005). Generalized Additive Models for \
            Location, Scale and Shape.

        :param y: A numpy array of shape (-1,1) containing each observed value.
        :type y: np.ndarray
        :param mus: 1 np array for the mean corresponding to each of N observations. Array is of
            shape (N,1).
        :type mus: np.ndarray
        :param index: Index for specific derivative vector to return.
        :type index: int
        :param order: Order of partial derivative.
        :type order: DerivOrder
        :return: a N-dimensional vector of shape (-1,1) containing the desired derivative evaluated
            for every observation in ``y``.
        :rtype: np.ndarray
        """
        mu = mus[0]

        if order == DerivOrder.d1:
            if index == 0:
                return (y - mu) / (0.5 * np.power(mu, 2))
            else:
                raise ValueError("No Derivative of order d1 exists at index > 0.")

        elif order == DerivOrder.d2:
            if index == 0:
                return -1 / (0.5 * np.power(mu, 2))
            else:
                raise ValueError("No Derivative of order d2 exists at index > 0.")

        elif order == DerivOrder.d2m:
            raise ValueError("No Derivative > order d2m exists.")

    def lp(self, y: np.ndarray, *mus: np.ndarray) -> np.ndarray:
        """Log-probability of observing every proportion in :math:`\\mathbf{y}` under their
        respective Gamma with mean = :math:`\\boldsymbol{\\mu}` and scale = 0.5.

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition\
            (2nd ed.).

        :param y: A numpy array containing each observed value.
        :type y: np.ndarray
        :param mus: 1 np array for the mean corresponding to each of N observations. Array is of
            shape (N,1).
        :type mus: np.ndarray
        :return: a N-dimensional vector containing the log-probability of observing each data-point
            under the current model.
        :rtype: np.ndarray
        """
        # Need to transform from mean and scale to \alpha & \beta
        # From Wood (2017), we have that
        # \\phi = 1/\alpha
        # so \alpha = 1/\\phi
        # From https://en.wikipedia.org/wiki/Gamma_distribution, we have that:
        # \\mu = \alpha/\beta
        # \\mu = 1/\\phi/\beta
        # \beta = 1/\\phi/\\mu
        # scipy docs, say to set scale to 1/\beta.
        # see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html
        mu = mus[0]
        alpha = 1 / 0.5
        beta = alpha / mu
        return scp.stats.gamma.logpdf(y, a=alpha, scale=(1 / beta))

    def lcp(self, y: np.ndarray, *mus: np.ndarray) -> np.ndarray:
        """Log of the cumulative probability of observing every value in y under their respective
        Gamma with mean = :math:`\\boldsymbol{\\mu}` and scale = 0.5.

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition\
            (2nd ed.).

        :param y: A numpy array containing each observed value.
        :type y: np.ndarray
        :param mus: 1 np array for the mean corresponding to each of N observations. Array is of
            shape (N,1).
        :type mus: np.ndarray
        :return: a N-dimensional vector containing the log of the cumulative probability of
            observing each data-point under the current model.
        :rtype: np.ndarray
        """
        mu = mus[0]
        alpha = 1 / 0.5
        beta = alpha / mu
        return scp.stats.gamma.logcdf(y, a=alpha, scale=(1 / beta))

    def llk(self, y: np.ndarray, *mus: np.ndarray) -> float:
        """log-probability of data under given model. Essentially sum over all elements in the
        vector returned by the :func:`lp` method.

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition\
            (2nd ed.).

        :param y: A numpy array containing each observation.
        :type y: np.ndarray
        :param mus: 1 np array for the mean corresponding to each of N observations. Array is of
            shape (N,1).
        :type mus: np.ndarray
        :return: The log-probability of observing all data under the current model.
        :rtype: float
        """
        mu = mus[0]
        return sum(self.lp(y, mu))[0]

    def get_resid(self, y: np.ndarray, *mus: np.ndarray) -> np.ndarray:
        """Get standardized residuals for a Gamma GAMMLSS model (Rigby & Stasinopoulos, 2005).

        Essentially, to get a standaridzed residual vector we first have to account for the
        mean-variance relationship of our RVs (which we also have to do for the :class:`Gamma`
        family) - for this we can simply compute deviance residuals again (see Wood, 2017).
        These should be :math:`\\sim N(0,\\phi_i)` (where :math:`\\phi_i` is the element in
        ``scale`` for a specific observation) - so if we divide each of those by the
        observation-specific scale we can expect the resulting standardized residuals to be
        :math:` \\sim N(0,1)` if the model is correct.

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).


        :param y: A numpy array containing each observation.
        :type y: np.ndarray
        :param mus: 1 np array for the mean corresponding to each of N observations. Array is of
            shape (N,1).
        :type mu: np.ndarray
        :return: A list of standardized residuals that should be ~ N(0,1) if the model is correct.
        :rtype: np.ndarray
        """
        mu = mus[0]
        res = np.sign(y - mu) * np.sqrt(Gamma().D(y, mu) / 0.5)
        return res

    def rvs(self, *mus: np.ndarray, size: int = 1, seed: int | None = 0) -> np.ndarray:
        """Returns ``size`` random samples for each of the distributions parameterized by ``mu``
        and ``scale``.

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param mus: 1 np array for the mean corresponding to each of N observations. Array is of
            shape (N,1).
        :type mus: np.ndarray
        :param size: Number of random samples to return per distribution. Defaults to 1.
        :type size: int, optional
        :param seed: Seed to use for random number generation. Defaults to 0.
        :type seed: int, optional
        :return: a numpy array of shape ``(size, mus[0].shape[0])`` containing random
            samples from every distribution parameterized by ``mu`` and ``scale``.
        :rtype: np.ndarray
        """
        mu = mus[0]
        scale = 0.5
        alpha = 1 / scale
        beta = alpha / mu
        return scp.stats.gamma.rvs(
            size=(size, mu.shape[0]),
            a=alpha.flatten(),
            scale=(1 / beta.flatten()),
            random_state=seed,
        )

    def init_coef(self, models: list[Callable]) -> np.ndarray:
        """Function to initialize the coefficients of the model.

        Fits a GAMM for the mean and initializes all coef.

        :param models: A list of :class:`mssm.models.GAMM`'s, - each based on one of the formulas
            provided to a model.
        :type models: [mssm.models.GAMM]
        :return: A numpy array of shape (-1,1), holding initial values for all model coefficients.
        :rtype: np.ndarray
        """
        return None
        mean_model = models[0]
        mean_model.family = Gamma(self.links[0], scale=0.5)
        mean_model.fit(progress_bar=False, max_inner=1)

        m_coef, _ = mean_model.get_pars()
        return m_coef


def init_hmp(
    dat: pd.DataFrame,
    tdat: pd.DataFrame,
    obs_formulas: list[Formula],
    d_formulas: list[Formula],
    init_states: list[np.ndarray] | None,
    fam: HSMMFamily,
    Xs: list[scp.sparse.csc_array | None],
    ys: list[np.ndarray | None],
    rand_init: bool,
    seed: int | None,
    offset_d: int = 5,
    should_plot: bool = False,
    should_init_durs: bool = True,
    axs: list[matplotlib.axes._axes.Axes] | None = None,
) -> tuple[np.ndarray, list[float]]:
    """Initializes a HMP-like HSMM based on (random) state estimates.

    :return: Initial estimates for coefficients and lambda parameters.
    :rtype: tuple[np.ndarray, list[float]]
    """

    def rand_init_series(
        ys: list[np.ndarray] | None,
        n_S: int,
        D: int,
        rand_init: bool,
        gen: np.random.Generator,
    ):

        # Extract info
        y = ys[0]
        n_T = len(y)
        n_events = int((n_S - 1) / 2)
        n_stages = n_events + 1

        # Min event dur subtracted -> n_T is max. total flat dur
        n_T -= int(5 * n_events)

        # Split total dur equally or randomly into n_stages tiles
        if rand_init:
            stage_durs = gen.choice(int(n_T / n_stages), size=n_stages) + 1
        else:
            stage_durs = [int(n_T / n_stages) for _ in range(n_stages)]

        # Collect splits
        durs = []
        sis = []
        si = 0
        for j in range(n_S):
            if j % 2 == 1:
                durs.append(5)
            else:
                durs.append(stage_durs[si])
                sis.append(j)
                si += 1

        # Make sure sum of total durs matches len of trial
        while (sum(durs)) < len(y):
            j = gen.choice(sis)

            if durs[j] < D:
                durs[j] += 1

        # Create corresponding state sequence
        states = []
        for j in range(n_S):
            states.extend([j for _ in range(durs[j])])

        return np.array(states), np.array(durs).reshape(1, -1)

    def compute_durs_series(states: np.ndarray, n_S: int):

        durs = [[] for _ in range(n_S)]
        d = 0
        j = states[0]

        for s in states:
            if s == j:
                d += 1
            else:
                durs[j].append(d)
                j = s
                d = 1

        # Catch dur from last state
        durs[j].append(d)

        durs = [np.mean(d) for d in durs]

        return np.array(durs).reshape(1, -1)

    # Extract families, transition matrices, etc.
    n_S = fam.llkargs[0]
    obs_fams = fam.llkargs[1]
    d_fams = fam.llkargs[2]
    sid = fam.llkargs[3]
    tid = fam.llkargs[4]
    D = fam.llkargs[5]
    M = fam.llkargs[6]
    starts_with_first = fam.llkargs[7]
    n_cores = fam.llkargs[10]
    shared_pars = fam.llkargs[12]
    shared_m = fam.llkargs[13]
    Lrhoi = fam.llkargs[16]
    scale = fam.llkargs[17]
    n_series = len(sid)
    event_template = fam.llkargs[18]
    event_width = len(event_template)

    # Split up ys, Xs
    split_Ys, _, _ = _split_matrices(
        ys if fam.fast_hmp is False else fam.cross_cor,
        Xs,
        shared_pars,
        shared_m,
        obs_fams,
        d_fams,
        sid,
        tid,
        n_S,
        M,
        False,
        False,
        starts_with_first,
        None if fam.fast_hmp else Lrhoi,
    )

    np_gen = np.random.default_rng(seed)

    if init_states is None:

        init_states = []
        init_durs = []
        if n_cores == 1:

            for series in range(n_series):
                statess, durss = rand_init_series(
                    split_Ys[series],
                    n_S,
                    D,
                    rand_init,
                    np_gen,
                )

                init_states.append(statess)
                init_durs.append(durss)

        else:  # Compute in parallel

            args = zip(
                split_Ys,
                repeat(n_S),
                repeat(D),
                repeat(rand_init),
                repeat(np_gen),
            )

            with mp.Pool(processes=n_cores) as pool:

                init_states, init_durs = zip(*pool.starmap(rand_init_series, args))

    else:
        init_durs = []
        if n_cores == 1:

            for series in range(n_series):
                durss = compute_durs_series(init_states[series], n_S)

                init_durs.append(durss)

        else:  # Compute in parallel

            args = zip(init_states, repeat(n_S))

            with mp.Pool(processes=n_cores) as pool:

                init_durs = pool.starmap(compute_durs_series, args)

    for series in range(n_series):
        for j in range(n_S):
            if j % 2 == 1 and j in init_states[series]:
                onset = np.argmin(np.abs(init_states[series] - j))
                init_states[series][init_states[series] == j] = -1
                init_states[series][onset + event_width // 2] = j

    init_states = np.concatenate(init_states)
    init_durs = np.concatenate(init_durs, axis=0)

    # Can now initialize coef
    init_coef = []
    init_lambda = []
    ofidx = 0
    axi = 0
    for j in range(n_S):

        for m in range(M):

            if j % 2 == 1:
                # Model of mean of obs model in state j and of signal m
                obs_form = obs_formulas[ofidx]
                init_j_dat = dat.loc[init_states == j, :]
                init_j_form = Formula(
                    lhs(obs_form.lhs.variable), terms=obs_form.terms, data=init_j_dat
                )

                init_j_model = GAMM(init_j_form, Gaussian(scale=scale))
                init_j_model.fit(progress_bar=False)

                if should_plot and len(obs_form.smooth_terms) > 0:
                    plot(init_j_model)

                init_coef.extend(init_j_model.coef.flatten())
                init_lambda.extend(
                    [
                        pen.lam / init_j_model.scale
                        for pen in init_j_model.overall_penalties
                    ]
                )
                ofidx += 1

    if should_init_durs is True:
        dfidx = 0
        for j in range(n_S):

            if j % 2 == 0:

                init_j_dat = copy.deepcopy(tdat)
                init_j_dat["d_init"] = init_durs[:, j] + (offset_d if j > 0 else 0)
                # print(init_j_dat)
                d_fam = d_fams[j]
                d_forms = []

                for _ in range(d_fam.n_par):
                    # Model of mean of obs model in state j and of signal m
                    dform = d_formulas[dfidx]
                    init_j_form = Formula(
                        lhs("d_init"), terms=dform.terms, data=init_j_dat
                    )
                    d_forms.append(init_j_form)
                    dfidx += 1

                try:
                    init_j_model = GAMMLSS(d_forms, d_fams[j])
                    init_j_model.fit(progress_bar=False)

                    init_coef.extend(init_j_model.coef.flatten())
                    init_lambda.extend(
                        [pen.lam for pen in init_j_model.overall_penalties]
                    )

                    if should_plot:
                        for par in range(d_fam.n_par):
                            plot(
                                init_j_model,
                                dist_par=par,
                                axs=([axs[axi]] if axs is not None else None),
                            )

                except:  # noqa
                    GAMM_Fam = (
                        Gamma(link=d_fam.links[0])
                        if isinstance(d_fam, GAMMALS)
                        else Gaussian(link=d_fam.links[0])
                    )
                    # print(GAMM_Fam)
                    init_j_model = GAMM(d_forms[0], GAMM_Fam)
                    init_j_model.fit(progress_bar=False)
                    if should_plot:
                        plot(
                            init_j_model, axs=([axs[axi]] if axs is not None else None)
                        )

                    # print(init_j_model.coef,init_j_model.scale)
                    init_coef.extend(init_j_model.coef.flatten())
                    init_lambda.extend(
                        [
                            pen.lam / init_j_model.scale
                            for pen in init_j_model.overall_penalties
                        ]
                    )

                    init_coef.extend(
                        [
                            d_fam.links[1].f(init_j_model.scale)
                            for _ in range(d_forms[1].n_coef)
                        ]
                    )

                    init_lambda.extend(
                        [1 for _ in range(len(build_penalties(d_forms[1])))]
                    )
                axi += 1

    return np.array(init_coef).reshape(-1, 1), init_lambda


def compute_mean_durs_series(states: np.ndarray, n_S: int):
    """Computes expected duration of each state for a time-series.

    :return: A numpy array of shape (1,-1) holding the mean duration of each state for
        the time series.
    :rtype: np.array
    """

    durs = [[] for _ in range(n_S)]
    d = 0
    j = states[0]

    for s in states:
        if s == j:
            d += 1
        else:
            durs[j].append(d)
            j = s
            d = 1

    # Catch dur from last state
    durs[j].append(d)

    durs = [np.mean(d) for d in durs]

    return np.array(durs).reshape(1, -1)


def compute_durs_transitions_series(states: np.ndarray, excessD: int, n_S: int):
    """Computes all state durations, state transitions, the total number of transitions
    and the initial state of a given time-series.

    :return: A (nested) list holding the durations of each state, a (nested) list holding
        the transitions away from every state, the total number of transitions, and
        the initial state (index)
    :rtype: tuple[list,list,int,int]
    """

    durs = [[] for _ in range(n_S)]
    transitions = [[] for _ in range(n_S)]
    total_transitions = 0
    d = 0
    j = states[0]
    j_init = j

    t = 0
    for s in states:
        if s == j:
            d += 1
        else:
            durs[j].append([t, d])
            transitions[j].append([t, s])
            total_transitions += 1
            j = s
            d = 1

        t += 1

    # Catch dur from last state
    durs[j].append([t, d + excessD])

    return (durs, transitions, total_transitions, j_init)


def advance_chain_hmp(
    chain_id,
    return_dict,
    d_fams: list,
    fam: HSMMFamily,
    coef: np.ndarray,
    init_lam: np.ndarray,
    coef_split_idx: list[int],
    formulas: list[Formula],
    Xs: list[scp.sparse.csc_array | None],
    ys: list[np.ndarray | None],
    n_series: int,
    n_S: int,
    M: int,
    scale: float,
    starts_with_first: bool,
    event_template: np.ndarray,
    n_cores_td: int = 1,
    n_cores_gammlss: int = 1,
    seed: int | None = 0,
    steps: int = 1,
    sample_map: bool = False,
    avg_e_terms: list = [],
):
    """Advances the chain of the Laplace approximate posterior sampler for a HMP model."""
    T = fam.llkargs[14]
    pi = fam.llkargs[15]
    event_width = len(event_template)
    np_gen = np.random.default_rng(seed)
    # tiled_template = np.tile(event_template, n_series).reshape(-1, 1)

    res_coef = np.zeros((len(coef), steps))
    res_lam = np.zeros((len(init_lam), steps))
    res_lp = np.zeros((1, steps))

    # Figure out dimension of shared coef
    avg_j_dim = 0
    if len(avg_e_terms) != 0:
        for tidx in avg_e_terms:
            avg_j_dim += len(formulas[0].coef_idx_per_term[tidx])

    for step in range(steps):
        # print(step)

        # Sample state sequences given data and coefficients
        state_samples = fam.sample_posterior_states(
            coef,
            coef_split_idx,
            ys,
            Xs,
            n_samples=1,
            seed=seed,
        )

        # Get corresponding model llk
        state_lp = 0

        # Collect states and durs
        states_flat = []
        durs = []
        transitions = []
        total_transitions = []
        init_js = []
        state_args = []
        ed_args = []
        flats = []
        for series in range(n_series):
            states_series = state_samples[series][1][:, 0]
            ed_series = state_samples[series][0][0]

            if n_cores_td <= 1:
                (
                    durs_series,
                    transitions_series,
                    total_transitions_series,
                    j_init_series,
                ) = compute_durs_transitions_series(states_series, ed_series, n_S)

                durs.append(durs_series)
                transitions.append(transitions_series)
                total_transitions.append(total_transitions_series)
                init_js.append(j_init_series)
            else:
                state_args.append(states_series)
                ed_args.append(ed_series)

            if fam.hmp_d_offset:
                _, _, _, _, flatss = fam._group_hmp_events(series)
                flats.append(flatss)

            for j in range(n_S):
                if j % 2 == 1 and j in states_series:
                    onset = np.argmin(np.abs(states_series - j))
                    states_series[states_series == j] = -1
                    states_series[onset + event_width // 2] = j

            states_flat.extend(states_series)

        # Compute durs and transitions in parallel
        if n_cores_td > 1:
            args = zip(state_args, ed_args, repeat(n_S))

            with mp.Pool(processes=n_cores_td) as pool:
                durs, transitions, total_transitions, init_js = zip(
                    *pool.starmap(compute_durs_transitions_series, args)
                )

        states_flat = np.array(states_flat)

        # Now estimate coef given states
        next_coef = []
        next_lam = []
        idx = 0  # Keep track of indices for ys and Xs

        # Emission coef
        for j in range(n_S):

            if j % 2 == 1:
                avg_j_indices = []
                avg_j_coef = np.zeros(avg_j_dim)
                avg_j_cov = np.zeros((avg_j_dim, avg_j_dim))
                jcoef = []

                for m in range(M):
                    # print(step, j, m)
                    # Since we assume HMP/HSMMMVPA pattern for observation model family will be None
                    # but we still need to compute models for bump states.

                    fJM = copy.deepcopy(formulas[idx])

                    # print(chain_id, step, j, m, np.sum(states_flat == j) / 5)
                    # print(j, np.sum(states_flat == j), tiled_template.shape[0])

                    if np.sum(states_flat == j) == 0:
                        jcoef.extend([np.nan for _ in range(formulas[idx].n_coef)])

                    else:

                        # Update data of formula
                        tdat = fJM.data

                        # Only keep obs/rows related to current state
                        tdat = tdat.loc[states_flat == j, :]

                        # Now update formula
                        y_flat, cov_flat, NAs_flat, y, cov, NAs, sid, _ = (
                            fJM.encode_data(tdat)
                        )

                        # Store encoding
                        fJM.y_flat = y_flat
                        fJM.cov_flat = cov_flat
                        fJM.NOT_NA_flat = NAs_flat
                        fJM.y = y
                        fJM.cov = cov
                        fJM.NOT_NA = NAs
                        fJM.sid = sid

                        # And fit!
                        g = GAMM(fJM, Gaussian(Identity(), scale=scale))

                        g.fit(
                            progress_bar=False,
                            method="Chol",
                            check_cond=0,
                            n_cores=n_cores_gammlss,
                        )
                        state_lp -= 0.5 * g.penalty

                        if sample_map:
                            jfmcoef = g.coef.flatten()
                            if len(g.overall_penalties) != 0:
                                next_lam.extend(
                                    [lam.lam / scale for lam in g.overall_penalties]
                                )

                        elif len(g.overall_penalties) == 0:
                            jfmcoef = g.sample_post(n_ps=1, seed=seed)[:, 0]

                        else:
                            # Get Vp
                            _, Vpr, _, _, ep, dBetadRhos = estimateVp(g)

                            # Sample lam
                            lam = np.exp(
                                scp.stats.multivariate_normal.rvs(
                                    mean=ep.flatten(),
                                    cov=Vpr,
                                    size=1,
                                    random_state=seed,
                                )
                            )

                            if len(ep) == 1:
                                lam = np.array([lam])

                            # Sample coef given lam
                            for lami in range(len(lam)):
                                # Enforce working infty limits
                                lam[lami] = max(1e-7, lam[lami])
                                lam[lami] = min(1e7, lam[lami])

                                g.overall_penalties[lami].lam = lam[lami]
                                g.overall_penalties[lami].frozen = True

                            next_lam.extend(lam / scale)

                            g.fit(
                                progress_bar=False,
                                method="Chol",
                                check_cond=0,
                                restart=True,
                                n_cores=n_cores_gammlss,
                            )

                            jfmcoef = g.sample_post(n_ps=1, seed=seed)[:, 0]

                        # Collect coefficients and corresponding indices over which to average
                        if len(avg_e_terms) != 0:
                            avg_jm_coef = []
                            avg_jm_indices = []
                            avg_jm_indices_r = []
                            for tidx in avg_e_terms:
                                cidx = fJM.coef_idx_per_term[tidx]
                                avg_jm_indices_r.extend(cidx)
                                avg_jm_indices.extend(cidx + len(jcoef))
                                avg_jm_coef.extend(g.coef[cidx].flatten())

                            avg_j_coef += avg_jm_coef
                            avg_j_cov += (g.lvi.T @ g.lvi * g.scale)[
                                np.ix_(avg_jm_indices_r, avg_jm_indices_r)
                            ]
                            avg_j_indices.append(avg_jm_indices)

                        jcoef.extend(jfmcoef)

                    idx += 1

                # Average over channels then sample
                if len(avg_e_terms) != 0 and len(avg_j_indices) != 0:
                    jcoef = np.array(jcoef)
                    avg_j_coef /= M
                    avg_j_cov /= M**2
                    avg_j_cov = scp.stats.multivariate_normal.rvs(
                        mean=avg_j_coef,
                        cov=avg_j_cov,
                        size=1,
                        random_state=seed,
                    )
                    for cidx in avg_j_indices:
                        jcoef[cidx] = avg_j_coef

                next_coef.extend(jcoef)

        # Duration coef
        fidx = 0
        for j in range(n_S * (1 if starts_with_first else 2)):
            # print("d", step, j)

            jdFam = d_fams[j]
            if jdFam is not None:

                # Extract formulas associated with duration model of this state
                fJM = copy.deepcopy(formulas[idx : idx + jdFam.n_par])  # noqa: E203

                ddat = fJM[0].data
                ddat_idx = []
                jdurs = []

                for series in range(n_series):
                    sdur = durs[series]

                    for td in sdur[j]:
                        offset = 0
                        if fam.hmp_d_offset:
                            offset = (
                                event_width // 2
                                if (fidx in [flats[series][0], flats[series][-1]])
                                else event_width
                            )

                        jdurs.append(td[1] + offset)
                        ddat_idx.append(series)

                if len(jdurs) == 0:
                    for par in range(jdFam.n_par):
                        next_coef.extend([np.nan for _ in range(formulas[idx].n_coef)])
                        idx += 1
                else:

                    ddat = ddat.iloc[ddat_idx, :]
                    ddat[fJM[0].lhs.variable] = jdurs

                    for par in range(jdFam.n_par):

                        # Update data of formula
                        y_flat, cov_flat, NAs_flat, y, cov, NAs, sid, _ = fJM[
                            par
                        ].encode_data(ddat)

                        # Store encoding
                        fJM[par].y_flat = y_flat
                        fJM[par].cov_flat = cov_flat
                        fJM[par].NOT_NA_flat = NAs_flat
                        fJM[par].y = y
                        fJM[par].cov = cov
                        fJM[par].NOT_NA = NAs
                        fJM[par].sid = sid
                        idx += 1

                    # Fit model
                    g = GAMMLSS(fJM, jdFam)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        g.fit(
                            progress_bar=False,
                            method="Chol",
                            n_cores=n_cores_gammlss,
                            check_cond=0,
                        )
                    state_lp -= 0.5 * g.penalty

                    if sample_map:
                        jfmcoef = g.coef.flatten()
                        if len(g.overall_penalties) != 0:
                            next_lam.extend([lam.lam for lam in g.overall_penalties])

                    else:
                        if len(g.overall_penalties) != 0:
                            # Get Vp
                            _, Vpr, _, _, ep, _ = estimateVp(g)

                            # Sample lam
                            lam = np.exp(
                                scp.stats.multivariate_normal.rvs(
                                    mean=ep.flatten(),
                                    cov=Vpr,
                                    size=1,
                                    random_state=seed,
                                )
                            )

                            if len(ep) == 1:
                                lam = np.array([lam])

                            # Sample coef given lam
                            for lami in range(len(lam)):
                                # Enforce working infty limits
                                lam[lami] = max(1e-7, lam[lami])
                                lam[lami] = min(1e7, lam[lami])

                                g.overall_penalties[lami].lam = lam[lami]
                                g.overall_penalties[lami].frozen = True

                            next_lam.extend(lam)

                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                g.fit(
                                    progress_bar=False,
                                    method="Chol",
                                    check_cond=0,
                                    n_cores=n_cores_gammlss,
                                    restart=True,
                                )

                        jfmcoef = []
                        for par in range(jdFam.n_par):
                            jfmcoef.extend(
                                g.sample_post(n_ps=1, seed=seed, par=par)[
                                    :, 0
                                ].flatten()
                            )

                    # print(jfmcoef)
                    next_coef.extend(jfmcoef)

                fidx += 2

        res_coef[:, step] = next_coef
        res_coef[np.isnan(next_coef), step] = coef[np.isnan(next_coef), 0]
        coef = res_coef[:, [step]].copy()
        # Update llk
        state_lp += fam.llk(coef, coef_split_idx, ys, Xs)
        res_lam[:, step] = np.array(next_lam)
        res_lp[0, step] = state_lp
        seed += step

    return_dict[chain_id] = [res_lp, res_coef, res_lam]


def sample_hmp(
    fam: HSMMFamily,
    init_coefs: list[np.ndarray],
    coef_split_idx: list[int],
    formulas: list[Formula],
    Xs: list[scp.sparse.csc_array | None],
    ys: list[np.ndarray | None],
    is_hmp: bool = False,
    n_iter=500,
    n_burnin=500,
    seed=0,
    n_cores_td=1,
    callback=None,
    n_chains=2,
    parallelize_chains=True,
    n_steps=100,
    sample_map: bool = False,
    avg_e_terms: list = [],
) -> tuple[np.ndarray, np.ndarray, GSMM]:
    """Laplace approximate beam sampler for a HMP-like HsMM as discussed by by Krause, van Rij, and
    Borst (submitted).

    :return: Samples for the coefficients, regularization parameters, and a GSMM initialized with
        the average coefficients.
    :rtype: tuple[np.ndarray, np.ndarray, GSMM]
    """
    np_gen = np.random.default_rng(seed)

    # Extract families, transition matrices, etc.
    n_S = fam.llkargs[0]
    d_fams = fam.llkargs[2]
    sid = fam.llkargs[3]
    M = fam.llkargs[6]
    starts_with_first = fam.llkargs[7]
    n_cores = fam.llkargs[10]
    scale = fam.llkargs[17]
    event_template = fam.llkargs[18]
    hmp_fam = fam.llkargs[19]
    is_hmp = fam.is_hmp
    tvdtpi = fam.tvdtpi

    if is_hmp is False:
        raise ValueError("Sampler only supports HMP-like HsMMs")

    if tvdtpi:
        raise ValueError(
            "Sampler does not support time-varying model of the state durations currently."
        )

    if hmp_fam != "Gaussian":
        raise ValueError("Only Gaussian HMP-like HsMMs are supported by the sampler.")

    n_series = len(sid)
    coefs = copy.deepcopy(init_coefs)
    lams = []

    n_lam = 0
    for form in formulas:
        pen = build_penalties(form)
        n_lam += len(pen)

    for chain in range(n_chains):
        lams.append(np.ones(n_lam))

    lp_samples = np.zeros((n_chains, n_iter, 1))
    coef_samples = np.zeros((n_chains, n_iter, len(coefs[0])))

    lam_samples = None
    if n_lam > 0:
        lam_samples = np.zeros((n_chains, n_iter, n_lam))

    manager = None
    if parallelize_chains:
        manager = mp.Manager()

    hsmm = GSMM(formulas, family=fam)
    hsmm.coef = np.ones(len(coefs[0])).reshape(-1, 1)
    hsmm.coef_split_idx = coef_split_idx
    hsmm.lvi = np.identity(len(coefs[0]))

    pbar = tqdm(total=n_iter + n_burnin, desc="Warming up...", leave=True)
    iter = 0
    while iter < (n_iter + n_burnin):

        # Calculate steps to take
        if iter < n_burnin:
            if iter + n_steps <= n_burnin:
                steps = n_steps
            else:
                steps = n_burnin - iter
        else:
            if iter + n_steps <= (n_iter + n_burnin):
                steps = n_steps
            else:
                steps = (n_iter + n_burnin) - iter

        return_dict = manager.dict() if parallelize_chains else dict()

        chains = []

        for chain in range(n_chains):

            chain_seed = int(np_gen.random() * 10000)

            args = (
                chain,
                return_dict,
                d_fams,
                fam,
                coefs[chain],
                lams[chain],
                coef_split_idx,
                formulas,
                Xs,
                fam.cross_cor,
                n_series,
                n_S,
                M,
                scale,
                starts_with_first,
                event_template,
                n_cores_td,
                n_cores,
                chain_seed,
                steps,
                sample_map,
                avg_e_terms,
            )

            if parallelize_chains:
                chains.append(mp.Process(target=advance_chain_hmp, args=args))
                chains[chain].start()
            else:
                advance_chain_hmp(*args)

        # Collect from chains
        for chain in range(n_chains):

            if parallelize_chains:
                chains[chain].join()

            coefs[chain] = return_dict[chain][1][:, [-1]]
            lams[chain] = return_dict[chain][2][:, -1]

            if iter >= n_burnin:
                sidx = iter - n_burnin
                eidx = sidx + steps
                lp_samples[chain, sidx:eidx, 0] = return_dict[chain][0][0, :]
                coef_samples[chain, sidx:eidx, :] = return_dict[chain][1].T
                if n_lam > 0:
                    lam_samples[chain, sidx:eidx, :] = return_dict[chain][2].T

        iter += steps

        pbar.update(steps)

        if (iter - n_burnin) == 0:
            pbar.set_description_str(desc="Sampling...", refresh=True)

        # Convergence:
        if iter > n_burnin:

            if ((iter - n_burnin) // 4) > (2 * n_chains):

                conv_samples_lp = lp_samples[:, : (iter - n_burnin), :]
                conv_samples_coef = coef_samples[:, : (iter - n_burnin), :]
                conv_samples_lam = None
                if n_lam > 0:
                    conv_samples_lam = np.log(lam_samples[:, : (iter - n_burnin), :])

                ess, rhat, mcse = check_convergence(
                    conv_samples_coef,
                    None,
                    None,
                    conv_samples_lp,
                    conv_samples_lam,
                    hsmm,
                    type=0,
                )

                desc = (
                    f"Sampling... Iter: {iter}, Min. ESS: {np.round(np.min(ess), decimals=2)}, "
                    f"Max. Rhat: {np.round(np.max(rhat), decimals=2)}, "
                    f"Max. MCSE: {np.round(np.max(mcse), decimals=2)}"
                )

                pbar.set_description_str(desc=desc, refresh=True)

            # Callback (optional)
            if callback is not None:
                callback(
                    iter - n_burnin,
                    coef_samples[:, : (iter - n_burnin), :],
                    lam_samples[:, : (iter - n_burnin), :] if n_lam > 0 else None,
                )

    # Create GSMM object with mean coef and approx. covariance matrix of samples
    mean_coef = np.mean(coef_samples, axis=(0, 1)).reshape(-1, 1)

    # Get approximate covariance matrix
    cov_samples = np.concatenate(
        [coef_samples[chain, :, :].T for chain in range(n_chains)], axis=1
    )

    try:
        V = np.cov(cov_samples)
        eigV, U = scp.linalg.eigh(V)

        # Enforce PD
        eps = np.power(np.finfo(float).eps, 0.5)
        thresh = eps * np.max(np.abs(eigV))
        eigV[eigV <= thresh] = thresh

        # Get root
        lvi = np.diag(np.sqrt(eigV)) @ U.T

    except:  # noqa
        warnings.warn(
            (
                "Failed to compute Root of regularized Covariance matrix "
                "of samples. Setting ``lvi`` attribute of returned "
                "hsmm to identity."
            )
        )
        lvi = np.identity(len(coefs[0]))

    hsmm.coef = mean_coef
    hsmm.lvi = lvi

    res = SamplerResult(
        lp_samples,
        coef_samples,
        rhos=None if lam_samples is None else np.log(lam_samples),
    )

    return res, hsmm


def plot_hsmm(
    model: (
        GSMM
        | list[
            HSMMFamily,
            np.ndarray,
            list[int],
            list[scp.sparse.csc_array | None],
        ]
    ),
    ys: list[np.ndarray | None],
    t_dat: pd.DataFrame,
    series: int = 0,
    build_matrix: list[bool] | None = None,
    max_emissions: int = 3,
    max_time: int = 200,
    emissions: list[int] | None = None,
    emission_colors: list[float] | None = None,
    color_map_emissions: str = "RdYlBu_r",
    state_colors: list[float] | None = None,
    color_map_states: str = "Paired",
    fig_size: tuple[float, float] = (19 / 2.54, 6 / 2.54),
    wspace: float = 0.1,
    max_ncol_leg: int = 6,
    seed: int = 0,
    bar_height: float = 0.6,
    d_dens_cutoff: float = 1e-3,
    avg_dur_label: str = "Average \n Duration",
    emission_label: str = "Signal",
    split_dur_by: str | None = None,
    split_dur_levels: list[str] | None = None,
    split_dur_levels_map: dict | None = None,
    dur_graph_type: int = 2,
    plot_state_labels: bool = True,
    network_graph: bool = True,
    n_samples: int = 10000,
    magnitudes: list[np.ndarray] | None = None,
    pc_weights: np.ndarray | None = None,
    means: np.ndarray | float | None = 0,
    scale: float | None = None,
    topo_width: int = 10,
    topo_height_rel: float = 2,
    positions: np.ndarray | None = None,
    highlight_state_1: int | None = None,
    highlight_state_2: int | None = None,
    highlight_d: int | None = None,
    max_emission_time: None | int = None,
    hide_densities: bool = False,
    fig: matplotlib.figure.Figure | None = None,
    axs: list[matplotlib.axes._axes.Axes] | None = None,
):
    """Plot an overview for an estimated HsMM model."""

    # Extract model quantities of interest for plotting
    if isinstance(model, GSMM):
        coef = model.coef
        coef_split_idx = model.coef_split_idx
        Xs = model.get_mmat()
        if build_matrix is not None:
            Xs = [X if build_matrix[xi] else None for xi, X in enumerate(Xs)]
        fam = model.family
    else:
        coef = model[1]
        coef_split_idx = model[2]
        Xs = model[3]
        fam = model[0]

    # Extract families, transition matrices, etc.
    n_S = fam.llkargs[0]
    sid = fam.llkargs[3]
    tid = fam.llkargs[4]
    M = fam.llkargs[6]
    starts_with_first = fam.llkargs[7]
    T = fam.llkargs[14]
    pi = fam.llkargs[15]
    event_template = fam.llkargs[18]
    is_hmp = fam.is_hmp
    n_series = len(sid)
    fix_T_pi = fam.fix_T_pi

    # Initialize potential None arguments
    if tid is None:
        tid = sid

    if isinstance(T, list):
        T = T[fam.T_index[series]]

    if isinstance(pi, list):
        pi = pi[fam.pi_index[series]]

    if fix_T_pi is False:
        ps = fam.compute_Tpi(coef, coef_split_idx, ys, Xs)

        T = ps[series][0]
        pi = ps[series][1]

    # Setup default arguments
    emission_iterator = emissions
    if emission_iterator is None:
        emission_iterator = np.arange(min(M, max_emissions))

    if emission_colors is None:
        emission_colors = np.linspace(0.1, 0.9, min(M, max_emissions))

    if state_colors is None:
        state_colors = np.arange(n_S)

    if is_hmp:
        flats = np.arange(0, n_S, 2)

    state_cmp = matplotlib.colormaps[color_map_states]
    emissions_cmp = matplotlib.colormaps[color_map_emissions]

    # Set up figure
    call_plot = False
    if fig is None and axs is None:
        fig = plt.figure(figsize=fig_size, layout="constrained")
        call_plot = True
        # subfigs = fig.subfigures(1,3, wspace=wspace)

    if axs is None:
        axs = fig.subplots(1, 3, gridspec_kw=dict(wspace=wspace))

    # Network graph or average state probability plot
    ax = axs[0]

    if network_graph:

        G = nx.MultiDiGraph()
        edgecolors = [
            (
                "red"
                if highlight_state_1 is not None and 0 == highlight_state_1
                else "black"
            )
        ]
        for j in range(n_S):

            if highlight_state_1 is not None and (j + 1) == highlight_state_1:
                edgecolors.append("red")
            elif highlight_state_2 is not None and (j + 1) == highlight_state_2:
                edgecolors.append("red")
            else:
                edgecolors.append("black")

            jrad = 0.15
            if j == int(n_S / 2) and n_S % 2 != 0:
                jrad = 0
            elif j >= n_S / 2:
                jrad = -0.15

            G.add_edge(
                "$\\boldsymbol{\\pi}$",
                j + 1,
                p=np.round(pi[j], decimals=2),
                rad=jrad,
                symbol="$\\boldsymbol{\\pi}_{" + f"{j+1}" + "}$",
            )

            for i in range(n_S):

                if i == j:
                    continue

                G.add_edge(
                    j + 1,
                    i + 1,
                    p=np.round(T[j, i], decimals=2),
                    rad=0.15,
                    symbol="$\\Delta_{" + f"{j+1}{i+1}" + "}$",
                )

        pos = nx.shell_layout(G)
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_size=200,
            node_color=["white", *state_cmp(state_colors)],
            edgecolors=edgecolors,
        )
        nx.draw_networkx_labels(
            G,
            pos,
            ax=ax,
            font_size=plt.rcParams["axes.titlesize"],
            font_family=plt.rcParams["font.family"],
            font_weight="bold",
        )

        for *edge, attrs in G.edges(keys=True, data=True):
            # print(edge,attrs)

            label = {tuple(edge): f"{attrs['p']}"}
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=[(edge[0], edge[1])],
                connectionstyle=f'arc3, rad = {attrs["rad"]}',
                ax=ax,
                alpha=1,
            )

            nx.draw_networkx_edge_labels(
                G,
                pos,
                label,
                connectionstyle=f'arc3, rad = {attrs["rad"]}',
                label_pos=0.2 if attrs["rad"] == 0 else 0.5,
                font_color=emissions_cmp(emission_colors[-1]),
                font_size=plt.rcParams["font.size"],
                font_family=plt.rcParams["font.family"],
                font_weight="semibold",
                bbox={
                    "alpha": 1,
                    "facecolor": "white",
                    "edgecolor": "white",
                    "pad": 0.1,
                },
                verticalalignment="center",
                horizontalalignment="center",
                ax=ax,
            )

        ax.set_title("State Transitions", fontweight="bold")
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.set_xlabel(
            "$\\boldsymbol{\\pi}=$ Initial State Distribution", fontweight="bold"
        )

    # Average state probability/duration graph
    else:

        if split_dur_by is None or is_hmp is False:
            state_samples = fam.sample_posterior_states(
                coef, coef_split_idx, ys, Xs, n_samples=n_samples
            )
            max_T = np.max(
                [state_samples[ser][1].shape[0] for ser in range(len(state_samples))]
            )
            time = np.arange(max_T)
            state_post_probs = np.zeros((max_T, n_S))

            norm = 0
            for ser in range(len(state_samples)):
                for seqi in range(n_samples):
                    seq = state_samples[ser][1][:, seqi]
                    seq_idx = np.arange(len(seq))

                    for j in range(n_S):
                        state_post_probs[seq_idx[seq == j], j] += 1

                    norm += 1

            state_post_probs /= norm

            # Now plot
            for j in range(n_S):
                ax.plot(
                    time,
                    state_post_probs[:, j],
                    label=f"State: {j+1}",
                    color=state_cmp(j),
                )

            ax.set_title("State Distribution", fontweight="bold")
            ax.spines["left"].set_visible(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(True)
            ax.set_xlabel("Time", fontweight="bold")
            ax.set_ylabel("Probability", fontweight="bold")
            # ax.set_ylim(0, 1.5)
            ax.set_yticks(np.linspace(0, 1, 5))

        # ax.legend(loc="upper right", ncol=1, frameon=False)
        if is_hmp:

            # Plot average state duration as well
            if split_dur_by is None:
                divider = make_axes_locatable(ax)
                ax2 = divider.append_axes("bottom", size="10%", pad=0.4)
            else:
                ax2 = ax
                ax2.set_title("State Distribution", fontweight="bold")

            viterbi = fam.decode_viterbi(coef, coef_split_idx, ys, Xs)

            est_durs = [
                compute_mean_durs_series(viterbi[s][1], n_S)
                for s in range(len(viterbi))
            ]
            est_durs = np.concatenate(est_durs, axis=0)

            lefts = []
            if split_dur_by is None:
                avg_state_durs = np.array(
                    [
                        (est_durs.mean(axis=0))[j]
                        + (len(event_template) if j > 0 else 0)
                        for j in flats
                    ]
                )

                left = np.cumsum([0, *avg_state_durs[:-1]])
                ax2.barh(
                    avg_dur_label,
                    avg_state_durs,
                    left=left,
                    color=state_cmp(state_colors[flats]),
                )
                ax2.set_xlim(0, round(np.sum(avg_state_durs)))

            else:
                split_levels = (
                    np.unique(t_dat[split_dur_by].values)
                    if split_dur_levels is None
                    else split_dur_levels
                )

                for lvl in split_levels:
                    state_durs = np.concatenate(
                        [
                            (
                                est_durs[t_dat[split_dur_by] == lvl, j].reshape(-1, 1)
                                + (len(event_template) if j > 0 else 0)
                            )
                            for j in flats
                        ],
                        axis=1,
                    )

                    # Average duration of states per level
                    avg_state_durs = state_durs.mean(axis=0)

                    left = np.cumsum([0, *avg_state_durs[:-1]])
                    lefts.append(left)

                    ax2.barh(
                        (
                            lvl
                            if split_dur_levels_map is None
                            else split_dur_levels_map[lvl]
                        ),
                        avg_state_durs,
                        left=left,
                        color=state_cmp(state_colors[np.arange(len(flats))]),
                        height=bar_height,
                    )

            # Plot bumps
            if magnitudes is not None and pc_weights is not None:

                lvls = ax2.get_yticks()

                if len(lvls) > 1:
                    # Make sure top bumps don't reach over axis lim
                    y_lim = list(ax2.get_ylim())
                    y_lim[-1] += (topo_height_rel * np.diff(ax2.get_ylim())[0]) / 2
                    ax2.set_ylim(y_lim)

                y_norm = np.diff(ax2.get_ylim())[0]

                adj_bump_loc = (
                    lvls + np.abs(ax2.get_ylim()[0])  # - (bar_height / 3)
                ) / y_norm

                x_norm = np.diff(ax2.get_xbound())[0]
                topo_width_rel = topo_width / x_norm
                for bumpi in range(len(magnitudes)):

                    bump_axis = ax2.inset_axes(
                        [
                            (left[bumpi + 1] - (topo_width / 2)) / x_norm,
                            adj_bump_loc[-1],  # 1,
                            topo_width_rel,
                            topo_height_rel,
                        ],
                        transform=ax2.transAxes,
                    )

                    bump = (
                        magnitudes[bumpi][-1]
                        if isinstance(magnitudes[bumpi], list)
                        else magnitudes[bumpi]
                    )
                    mne.viz.plot_topomap(
                        np.ndarray.flatten(bump.T @ pc_weights.T),
                        positions,
                        cmap="Spectral_r",
                        axes=bump_axis,
                        show=False,
                        sensors=False,
                    )

                    if isinstance(magnitudes[bumpi], list):

                        # print(lvls, y_norm, adj_bump_loc)
                        for bumpi2 in range(len(lvls) - 1):
                            # print(bumpi2)
                            bump_axis2 = ax2.inset_axes(
                                [
                                    (lefts[bumpi2][bumpi + 1] - (topo_width / 2))
                                    / x_norm,
                                    adj_bump_loc[bumpi2],
                                    topo_width_rel,
                                    topo_height_rel,
                                ],
                                transform=ax2.transAxes,
                            )

                            bump2 = magnitudes[bumpi][bumpi2]
                            mne.viz.plot_topomap(
                                np.ndarray.flatten(bump2.T @ pc_weights.T),
                                positions,
                                cmap="Spectral_r",
                                axes=bump_axis2,
                                show=False,
                                sensors=False,
                            )

            if split_dur_by is None:
                # Fix first axis
                ax.tick_params(bottom=False, labelbottom=False)
                ax.spines["bottom"].set_visible(False)
                ax.set_xlim(0, round(np.sum(avg_state_durs)))

            ax2.tick_params(left=False, labelleft=True)
            ax2.spines["left"].set_visible(False)
            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_visible(False)
            ax2.set_xlabel("Time", fontweight="bold")
            # ax2.tick_params(axis="y", labelrotation=90)

    """
    ax.text(
        0,
        1.05,
        "a)",
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=plt.rcParams["axes.titlesize"],
    )
    """

    # Network graph end

    # Durations density graph
    ax = axs[1]
    handles = []

    edps = fam.compute_od_probs(coef, coef_split_idx, ys, Xs, log=False)
    avg_d_dens = np.mean([edps[t][1] for t in range(n_series)], axis=0)

    plot_D = np.sum(avg_d_dens.max(axis=1) > d_dens_cutoff)
    flat_idx = 0
    for state in range(n_S):

        if hide_densities:
            continue

        if (
            highlight_state_2 is not None
            and highlight_d is not None
            and (state + 1) == highlight_state_2
        ):
            ax.axvline(highlight_d, color=state_cmp(state), linestyle="dashed")

        if is_hmp and state % 2 == 1:
            continue

        ax.fill(
            [*np.arange(1, plot_D + 1), *np.flip(np.arange(1, plot_D + 1))],
            [*avg_d_dens[:plot_D, state], *np.zeros(plot_D)],
            alpha=0.5,
            color=state_cmp(flat_idx if is_hmp else state),
        )
        hd = ax.plot(
            np.arange(1, plot_D + 1),
            avg_d_dens[:plot_D, state],
            label=(
                f"{('Flat' if is_hmp else 'State')}: {(flat_idx if is_hmp else state)+1}"
                if starts_with_first
                else None
            ),
            linestyle=("solid" if starts_with_first else "dashed"),
            color=state_cmp(flat_idx if is_hmp else state),
        )
        flat_idx += 1
        if starts_with_first:
            handles.append(hd[0])

    if starts_with_first is False:
        for state in range(n_S, 2 * n_S):
            ax.fill(
                [*np.arange(1, plot_D + 1), *np.flip(np.arange(1, plot_D + 1))],
                [*avg_d_dens[:plot_D, state], *np.zeros(plot_D)],
                alpha=0.5,
                color=state_cmp(state - n_S),
            )
            hd = ax.plot(
                np.arange(1, plot_D + 1),
                avg_d_dens[:plot_D, state],
                label=f"State: {(state - n_S)+1}",
                color=state_cmp(state - n_S),
            )
            handles.append(hd[0])

    ax.set_xticks(np.linspace(1, plot_D, 5))
    ax.set_xlim(1, plot_D)
    ax.set_xlabel("State Duration", fontweight="bold")
    ax.set_ylabel("Probability", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("State Durations", fontweight="bold")

    # Durations density graph end

    # Emission plot
    ax = axs[2]
    # print(ax, axs)

    if is_hmp:
        pred = fam.predict(
            coef, coef_split_idx, ys, Xs, seed=seed, n_samples=1, scale=scale
        )
    else:
        pred = fam.predict(coef, coef_split_idx, ys, Xs, seed=seed, n_samples=1)
    sample = 0

    emission_pred = pred[series][0][:, :, sample]
    state_pred = pred[series][1][:, sample]

    if emission_pred.shape[0] > max_time:
        emission_pred = emission_pred[:max_time, :]
        state_pred = state_pred[:max_time]

    time = np.arange(emission_pred.shape[0])

    if max_emission_time is None:
        max_emission_time = emission_pred.shape[0]

    m = 0
    for m in emission_iterator:
        ax.plot(
            time[:max_emission_time],
            emission_pred[:max_emission_time, m],
            label=emission_label + f": {m+1}",
            color=emissions_cmp(emission_colors[m]),
            alpha=1,
        )
        m += 1

    dat_max = np.max(emission_pred)
    dat_min = np.min(emission_pred)
    dat_sd = np.std(emission_pred)
    y_ticks = [*np.linspace(dat_min - 0.1 * dat_sd, dat_max + 3.5 * dat_sd, 5)]

    ax.set_yticks(y_ticks)
    ax.set_xlim(0, emission_pred.shape[0])
    ax.set_ylabel("Emissions", fontweight="bold")
    ax.tick_params(bottom=False, labelbottom=False)
    ax.spines["left"].set_visible(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_title("State Emissions", fontweight="bold")

    divider = make_axes_locatable(ax)
    ax3 = divider.append_axes("bottom", size="20%", pad=0.2)
    # print(ax2, ax3)
    ax3.plot(
        time[:max_emission_time], state_pred[:max_emission_time] + 1, color="black"
    )

    if plot_state_labels:
        ax3.set_yticks(np.arange(n_S) + 1)
    else:
        ax3.tick_params(labelleft=False)
    ax3.set_ylabel("States", fontweight="bold")
    # ax3.spines["left"].set_visible(False)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.set_xlabel("Time", fontweight="bold")
    ax3.set_xlim(0, emission_pred.shape[0])

    ax.legend(loc="upper right", ncol=1, frameon=False)

    if fig is not None:
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=min(n_S, max_ncol_leg),
            bbox_transform=fig.transFigure,
            bbox_to_anchor=(0.5, -0.1),
            frameon=False,
        )

        fig.align_xlabels()
        fig.align_titles()

    if call_plot:
        plt.plot()


def plot_val_hsmm(
    model: (
        GSMM
        | list[
            HSMMFamily,
            np.ndarray,
            list[int],
            list[scp.sparse.csc_array | None],
        ]
    ),
    ys: list[np.ndarray | None],
    dat: pd.DataFrame,
    t_dat: pd.DataFrame,
    which: int,
    time_var: str = "samples",
    trials_var: str = "epochs",
    type_3_res: str = "posterior_dur",
    pred_vars: list[str] = [],
    resid_terms: list[Formula] | None = None,
    resid_states: list[int] | None = None,
    which_pred: list[int] | None = None,
    process_dur_res_func: Callable | None = None,
    add_resid: bool = True,
    plot_partial: int | None = None,
    partial_title: str = "Partial plot",
    emission_label: str = "Signal",
    ar_lag: int = 50,
    per_component: bool = False,
    select_component: int | None = None,
    predictive_examples: int = 10,
    quantile_label_pos: float = 0.75,
    build_matrix: list[bool] | None = None,
    state_colors: list[float] | None = None,
    color_map_states: str = "Paired",
    color_map_resids: str = "RdYlBu_r",
    color_resids_by: str | None = None,
    fig_size: tuple[float, float] = (19 / 2.54, 6 / 2.54),
    wspace: float = 0.1,
    hspace: float = 0.1,
    seed: int = 0,
    n_samples: int = 10000,
    is_hmp: bool = False,
    alpha: float = 1,
    fig: matplotlib.figure.Figure | None = None,
):
    """Generate different validation plots for an estimated HsMM model."""

    np_gen = np.random.default_rng(seed)
    # Extract model quantities of interest for plotting
    if isinstance(model, GSMM):
        coef = model.coef
        coef_split_idx = model.coef_split_idx
        Xs = model.get_mmat()
        if build_matrix is not None:
            Xs = [X if build_matrix[xi] else None for xi, X in enumerate(Xs)]
        fam = model.family
    else:
        coef = model[1]
        coef_split_idx = model[2]
        Xs = model[3]
        fam = model[0]

    # Extract families, transition matrices, etc.
    n_S = fam.llkargs[0]
    sid = fam.llkargs[3]
    tid = fam.llkargs[4]
    M = fam.llkargs[6]
    is_hmp = fam.is_hmp
    n_series = len(sid)

    # Initialize potential None arguments
    if tid is None:
        tid = sid

    if state_colors is None:
        state_colors = np.arange(n_S)

    resid_cmp = matplotlib.colormaps[color_map_resids]

    if color_resids_by is not None:
        # Get color predictor
        color_cluster = np.unique(dat[color_resids_by])

        resid_cmap = {
            c: v
            for c, v in zip(
                color_cluster,
                np_gen.choice(
                    np.linspace(0.1, 0.9, len(color_cluster)),
                    size=len(color_cluster),
                    replace=False,
                ),
            )
        }

    if resid_states is None:
        resid_states = np.arange(n_S)

    # Set up figure
    call_plot = False
    if fig is None:
        fig = plt.figure(figsize=fig_size, layout="constrained")
        call_plot = True

    # Forward resid: qq, against time, order, and AR
    if which in [0, 2]:
        if per_component:
            axs = fig.subplots(
                4 if which == 0 else 5,
                M,
                gridspec_kw=dict(wspace=wspace, hspace=hspace),
            )
        else:
            axs = fig.subplots(
                1, 4 if which == 0 else 5, gridspec_kw=dict(wspace=wspace)
            )

        if which == 0:
            if select_component is None:
                fig.suptitle("Forward Quantile Residual Plots", fontweight="bold")
            else:
                fig.suptitle(
                    "Forward Quantile Residual Plots for "
                    + emission_label
                    + f": {select_component + 1}",
                    fontweight="bold",
                )
        else:
            if select_component is None:
                fig.suptitle("Forward Residual Plots", fontweight="bold")
            else:
                fig.suptitle(
                    "Forward Residual Plots for "
                    + emission_label
                    + f": {select_component + 1}",
                    fontweight="bold",
                )

        forward_res = fam.get_resid(
            coef, coef_split_idx, ys, Xs, transform_to_normal=True, resid_type="forward"
        )

        if select_component is None:
            forward_res_sum = np.sum(forward_res, axis=1) / forward_res.shape[1]
        else:
            forward_res_sum = forward_res[:, select_component]

        for m in range(M if per_component else 1):

            # QQ plot
            ax = axs[0, m] if per_component else axs[0]
            m_res = forward_res[:, m] if per_component else forward_res_sum

            qs = np.linspace(
                5 / (len(m_res) * 10), 1 - (5 / (len(m_res) * 10)), len(m_res)
            )
            tq = scp.stats.norm.ppf(qs, scale=math.sqrt(1))

            # Get empirical quantiles
            eq = np.sort(m_res, axis=0)

            ax.scatter(tq, eq, color="black", facecolor="none")

            # Add theoretical truth line
            eqs2 = np.quantile(m_res, [0.25, 0.75])
            tq2 = scp.stats.norm.ppf([0.25, 0.75], scale=math.sqrt(1))

            slope = (eqs2[1] - eqs2[0]) / (tq2[1] - tq2[0])
            offset = eqs2[0] - slope * tq2[0]
            ax.axline([0, offset], slope=slope, color="black")

            ax.set_title(
                (
                    f"Residual QQ-plot Signal: {m + 1}"
                    if per_component
                    else "Residual QQ-plot"
                ),
                fontweight="bold",
            )
            ax.set_xlabel("Theoretical Quantiles", fontweight="bold")
            ax.set_ylabel("Quantile Residuals", fontweight="bold")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if m > 0:
                ax.set_ylabel("")

            # Time plot
            ax = axs[1, m] if per_component else axs[1]

            if color_resids_by is None:
                ax.scatter(
                    dat[time_var], m_res, color="black", facecolor="none", alpha=alpha
                )
            else:
                ax.scatter(
                    dat[time_var],
                    m_res,
                    color=[resid_cmp(resid_cmap[c]) for c in dat[color_resids_by]],
                    facecolor="none",
                    alpha=alpha,
                )

            ax.set_title(
                (
                    f"Residual Time-course Signal: {m + 1}"
                    if per_component
                    else "Residual Time-course"
                ),
                fontweight="bold",
            )
            ax.set_xlabel("Time", fontweight="bold")
            ax.set_ylabel("Quantile Residuals", fontweight="bold")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if m > 0:
                ax.set_ylabel("")

            # Order
            ax = axs[2, m] if per_component else axs[2]

            if color_resids_by is None:
                ax.scatter(
                    np.arange(len(m_res)),
                    m_res,
                    color="black",
                    facecolor="none",
                    alpha=alpha,
                )
            else:

                ax.scatter(
                    np.arange(len(m_res)),
                    m_res,
                    color=[resid_cmp(resid_cmap[c]) for c in dat[color_resids_by]],
                    facecolor="none",
                    alpha=alpha,
                )

            ax.set_title(
                (
                    f"Ordered Residuals Signal: {m + 1}"
                    if per_component
                    else "Ordered Residuals"
                ),
                fontweight="bold",
            )

            ax.set_xlabel("Index", fontweight="bold")
            ax.set_ylabel("Quantile Residuals", fontweight="bold")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if m > 0:
                ax.set_ylabel("")

            # Acf
            ax = axs[3, m] if per_component else axs[3]

            # Auto-correlation check
            acf = np.array(
                [
                    1,
                    *[
                        np.corrcoef(m_res[:-lag], m_res[lag:])[0, 1]
                        for lag in range(1, ar_lag)
                    ],
                ]
            )

            for lg in range(ar_lag):
                ax.plot([lg, lg], [0, acf[lg]], color="black", linewidth=0.5)

            ax.set_title(
                (
                    f"Quantile Residual ACF Signal: {m + 1}"
                    if per_component
                    else "Quantile Residual ACF"
                ),
                fontweight="bold",
            )
            ax.axhline(0, color="red")
            ax.set_xlabel("Lag", fontweight="bold")
            ax.set_ylabel("ACF", fontweight="bold")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_ylim(-1, 1)

            if m > 0:
                ax.set_ylabel("")

    # Predictive resid against time, and qq
    if which in [1, 2]:

        if which == 1:
            if per_component:
                axs = fig.subplots(2, M, gridspec_kw=dict(wspace=wspace, hspace=hspace))
            else:
                axs = fig.subplots(1, 2, gridspec_kw=dict(wspace=wspace))

        if which == 1:
            if select_component is None:
                fig.suptitle("Forward Predictive Residual Plots", fontweight="bold")
            else:
                fig.suptitle(
                    "Forward Predictive Residual Plots for "
                    + emission_label
                    + f": {select_component + 1}",
                    fontweight="bold",
                )

        predictive_res = fam.get_resid(
            coef, coef_split_idx, ys, Xs, resid_type="predictive"
        )

        if select_component is None:
            predictive_res_sum = (
                np.sum(predictive_res, axis=1) / predictive_res.shape[1]
            )
        else:
            predictive_res_sum = predictive_res[:, select_component]

        for m in range(M if per_component else 1):
            # QQ plot
            m_res = predictive_res[:, m] if per_component else predictive_res_sum

            if which == 1:
                ax = axs[0, m] if per_component else axs[0]

                qs = np.linspace(
                    5 / (len(m_res) * 10), 1 - (5 / (len(m_res) * 10)), len(m_res)
                )
                tq = scp.stats.norm.ppf(qs, scale=math.sqrt(1))

                # Get empirical quantiles
                eq = np.sort(m_res, axis=0)

                ax.scatter(tq, eq, color="black", facecolor="none")

                # Add theoretical truth line
                eqs2 = np.quantile(m_res, [0.25, 0.75])
                tq2 = scp.stats.norm.ppf([0.25, 0.75], scale=math.sqrt(1))

                slope = (eqs2[1] - eqs2[0]) / (tq2[1] - tq2[0])
                offset = eqs2[0] - slope * tq2[0]
                ax.axline([0, offset], slope=slope, color="black")

                ax.set_title(
                    (
                        f"Residual QQ-plot Signal: {m + 1}"
                        if per_component
                        else "Residual QQ-plot"
                    ),
                    fontweight="bold",
                )
                ax.set_xlabel("Theoretical Quantiles", fontweight="bold")
                ax.set_ylabel("Predictive Residuals", fontweight="bold")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

                if m > 0:
                    ax.set_ylabel("")

            # Time-course
            ax = (
                axs[1 if which == 1 else 4, m]
                if per_component
                else axs[1 if which == 1 else 4]
            )

            res_dat = copy.deepcopy(dat)
            res_dat["predictive_res"] = m_res

            ax.axhline(scp.stats.norm.ppf(0.99), linestyle="dotted", color="red")
            ax.text(
                x=np.quantile(np.unique(res_dat[time_var].values), quantile_label_pos),
                y=scp.stats.norm.ppf(0.99) + 0.2,
                s="$\\Phi^{-1}(0.99)$",
                color="red",
            )
            ax.axhline(scp.stats.norm.ppf(0.95), linestyle="dashed", color="red")
            ax.axhline(0, linestyle="solid", color="red")

            ax.text(
                x=np.quantile(np.unique(res_dat[time_var].values), quantile_label_pos),
                y=scp.stats.norm.ppf(0.05) + 0.2,
                s="$\\Phi^{-1}(0.05))$",
                color="red",
            )
            ax.axhline(scp.stats.norm.ppf(0.05), linestyle="dashed", color="red")
            ax.axhline(scp.stats.norm.ppf(0.01), linestyle="dotted", color="red")

            if n_series > 1:

                ax.fill(
                    [
                        *np.unique(res_dat[time_var].values),
                        *np.flip(np.unique(res_dat[time_var].values)),
                    ],
                    [
                        *res_dat.loc[:, [time_var, "predictive_res"]]
                        .groupby([time_var])
                        .quantile(0.99)
                        .values,
                        *np.flip(
                            res_dat.loc[:, [time_var, "predictive_res"]]
                            .groupby([time_var])
                            .quantile(0.01)
                            .values
                        ),
                    ],
                    alpha=0.25,
                    color="black",
                )

                ax.fill(
                    [
                        *np.unique(res_dat[time_var].values),
                        *np.flip(np.unique(res_dat[time_var].values)),
                    ],
                    [
                        *res_dat.loc[:, [time_var, "predictive_res"]]
                        .groupby([time_var])
                        .quantile(0.95)
                        .values,
                        *np.flip(
                            res_dat.loc[:, [time_var, "predictive_res"]]
                            .groupby([time_var])
                            .quantile(0.05)
                            .values
                        ),
                    ],
                    alpha=0.5,
                    color="black",
                )

                for ep in np_gen.choice(
                    np.unique(res_dat[trials_var].values),
                    size=predictive_examples,
                    replace=False,
                ):
                    ax.plot(
                        res_dat[time_var].values[res_dat[trials_var] == ep],
                        m_res[res_dat[trials_var] == ep],
                        color="black",
                        linewidth=1,
                    )
            else:
                ax.plot(res_dat[time_var].values, m_res, color="black")

            ax.set_title(
                (
                    f"Residual Time-course Signal: {m + 1}"
                    if per_component
                    else "Residual Time-course"
                ),
                fontweight="bold",
            )

            # Rug plot
            unq_t, unq_counts = np.unique(res_dat[time_var].values, return_counts=True)
            ax.scatter(
                unq_t,
                [ax.get_ylim()[0]] * len(unq_t),
                marker="|",
                color="black",
                linewidths=1,
                alpha=np.array(unq_counts) / np.max(unq_counts),
            )

            ax.set_xlabel("Time", fontweight="bold")
            ax.set_ylabel("Predictive Residuals", fontweight="bold")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if m > 0:
                ax.set_ylabel("")

    if which in [3, 4]:
        axs = fig.subplots(
            (4 if plot_partial else 3) if which == 4 else 1,
            len(resid_states),
            gridspec_kw=dict(wspace=wspace, hspace=hspace),
        )

        dur_res = fam.get_resid(
            coef,
            coef_split_idx,
            ys,
            Xs,
            resid_type=type_3_res,
            n_samples=n_samples,
            seed=seed,
        )

        if type_3_res == "posterior_dur":
            dur_res = dur_res.mean(axis=2)
            fig.suptitle("Posterior Duration Residual Plots", fontweight="bold")
        else:
            fig.suptitle("Viterbi Duration Residual Plots", fontweight="bold")

        ax_idx = 0
        for st in resid_states:

            # Covariate plot
            ax = axs[0, ax_idx] if which == 4 else axs[ax_idx]

            state_res = dur_res[:, st]

            if len(pred_vars) == 1 and add_resid:
                ax.scatter(
                    t_dat[pred_vars[0]].values,
                    state_res,
                    color="black",
                    facecolor="none",
                )
                ax.axhline(0, color="red")

            res_dat = copy.deepcopy(t_dat)
            res_dat["res"] = state_res

            res_dat = res_dat.sort_values(by=pred_vars).reset_index(drop=True)

            g = GAMM(Formula(lhs("res"), resid_terms, data=res_dat), Gaussian())
            g.fit(progress_bar=False)

            if process_dur_res_func is None:
                pred_res_dats = [res_dat]
                pred_res_cols = [state_colors[((st // 2) if is_hmp else st)]]
                color_map_pred = color_map_states
            else:
                color_map_pred, pred_res_dats, pred_res_cols = process_dur_res_func(
                    res_dat
                )

            pred_ylim = [None, None]
            for pred_res_dat, pred_res_col in zip(pred_res_dats, pred_res_cols):

                plot_fitted(
                    pred_res_dat,
                    pred_vars,
                    g,
                    use=which_pred,
                    ax=ax,
                    ylim=ax.get_ylim() if add_resid else None,
                    col=pred_res_col,
                    cmp=color_map_pred,
                )

                new_pred_ylim = list(ax.get_ylim())

                if pred_ylim[0] is None:
                    pred_ylim[0] = new_pred_ylim[0]
                    pred_ylim[1] = new_pred_ylim[1]
                else:
                    if new_pred_ylim[0] < pred_ylim[0]:
                        pred_ylim[0] = new_pred_ylim[0]
                    if new_pred_ylim[1] > pred_ylim[1]:
                        pred_ylim[1] = new_pred_ylim[1]

            if pred_ylim[0] is not None:
                ax.set_yticks(
                    np.linspace(*pred_ylim, 5),
                    np.round(np.linspace(*pred_ylim, 5), decimals=2),
                )
                ax.set_ylim(pred_ylim)

            if len(pred_vars) == 1:
                ax.set_ylabel("Duration Residuals", fontweight="bold")
            if ax_idx > 0:
                ax.set_ylabel("")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_title(
                f"{'Flat' if is_hmp else 'State'}: {(st // 2 if is_hmp else st) + 1}\nResiduals",
                fontweight="bold",
            )

            if which == 4:
                row_idx = 1

                # Partial plot (for example of random smooth terms)
                if plot_partial is not None:
                    ax = axs[row_idx, ax_idx]

                    plot(g, which=[plot_partial], axs=[ax])

                    ax.set_title(
                        partial_title,
                        fontweight="bold",
                    )
                    ax.set_ylabel("Duration Residuals", fontweight="bold")

                    if ax_idx > 0:
                        ax.set_ylabel("")

                    row_idx += 1

                # QQ plot
                ax = axs[row_idx, ax_idx]

                qs = np.linspace(
                    5 / (len(state_res) * 10),
                    1 - (5 / (len(state_res) * 10)),
                    len(state_res),
                )
                tq = scp.stats.norm.ppf(qs, scale=math.sqrt(1))

                # Get empirical quantiles
                eq = np.sort(state_res, axis=0)

                ax.scatter(tq, eq, color="black", facecolor="none")

                # Add theoretical truth line
                eqs2 = np.quantile(state_res, [0.25, 0.75])
                tq2 = scp.stats.norm.ppf([0.25, 0.75], scale=math.sqrt(1))

                slope = (eqs2[1] - eqs2[0]) / (tq2[1] - tq2[0])
                offset = eqs2[0] - slope * tq2[0]
                ax.axline([0, offset], slope=slope, color="black")

                ax.set_title(
                    "Residual QQ-plot",
                    fontweight="bold",
                )
                ax.set_xlabel("Theoretical Quantiles", fontweight="bold")
                ax.set_ylabel("Duration Residuals", fontweight="bold")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

                if ax_idx > 0:
                    ax.set_ylabel("")
                row_idx += 1

                # ACF plot
                ax = axs[row_idx, ax_idx]

                acf = np.array(
                    [
                        1,
                        *[
                            np.corrcoef(state_res[:-lag], state_res[lag:])[0, 1]
                            for lag in range(1, ar_lag)
                        ],
                    ]
                )

                for lg in range(ar_lag):
                    ax.plot([lg, lg], [0, acf[lg]], color="black", linewidth=0.5)

                ax.set_title(
                    "Residual ACF",
                    fontweight="bold",
                )
                ax.axhline(0, color="red")
                ax.set_xlabel("Lag", fontweight="bold")
                ax.set_ylabel("ACF", fontweight="bold")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.set_ylim(-1, 1)

                if ax_idx > 0:
                    ax.set_ylabel("")

            ax_idx += 1

    fig.align_labels(axs)

    if call_plot:
        plt.plot()
