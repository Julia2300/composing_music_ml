from helper_functions import get_file_and_dirnames
from mido import MidiFile, tempo2bpm
import pandas as pd
from tqdm import tqdm

# analysis on midi file level

def get_midi_analysis(mid):
    """
    Get MIDI file analysis on a file level.

    :param mid: Mido MidiFile object
    :return: Dictionary with analysis information including type, length in seconds, ticks per beat, and track count
    """
    return {
        "type": mid.type,
        "length_sec": mid.length.__round__(2),
        "ticks_per_beat": mid.ticks_per_beat,
        "track_count": len(mid.tracks)
    }

# convert midi file to dataframe

def midi2pandas(mid):
    """
    Convert MIDI file to a pandas DataFrame.

    :param mid: Mido MidiFile object
    :return: Two DataFrames: one for song messages and one for meta messages
    """
    song=[]
    meta=[]
    for j in range(len(mid.tracks)):
        for i in mid.tracks[j] :
            params=i.__dict__
            if str(type(i))!="<class 'mido.midifiles.meta.MetaMessage'>":
                song.append(params)
            else:
                meta.append(params)

    df=pd.DataFrame(song).fillna(0)
    df["time_abs"]=0
    for i in df["channel"].unique():
        filtre=(df["channel"]==i)
        df.loc[filtre,'time_abs']=df.loc[filtre,"time"].cumsum()
    df=pd.concat([df,df['type'].str.get_dummies()], axis=1)
    df=df.sort_values(by=['time_abs','channel'])

    meta_df=pd.DataFrame(meta).fillna(0)
    meta_df=pd.concat([meta_df,meta_df['type'].str.get_dummies()], axis=1)

    return df, meta_df

# analysis on dataframe level

def get_meta_analysis(df):
    """
    Analyze meta information from DataFrame.

    :param df: DataFrame containing MIDI meta messages
    :return: Dictionary with metrics related to time signature, key signature, and tempo
    """
    # time signature metrics
    try:
        num = df[df["type"]=="time_signature"]["numerator"].iloc[0]
        denom = df[df["type"]=="time_signature"]["denominator"].iloc[0]
        clocks = df[df["type"]=="time_signature"]["clocks_per_click"].iloc[0]
        notated = df[df["type"]=="time_signature"]["notated_32nd_notes_per_beat"].iloc[0]
        time_sig_count = len(df[df["type"]=="time_signature"].index)
    except:
        num = 0
        denom = 0
        clocks = 0
        notated = 0
        time_sig_count = 0
    # key metrics
    try:
        key = df[df["type"]=="key_signature"]["key"].iloc[0]
    except:
        key = 0
    # tempo metrics
    try:
        tempo = df[df["type"]=="set_tempo"]["tempo"].iloc[0]
        bpm = tempo2bpm(tempo)
        tempo_count = len(df[df["type"]=="set_tempo"].index)
    except:
        tempo = 0
        bpm = 0
        tempo_count = 0

    return {
        "numerator": num,
        "denominator": denom,
        "clocks_per_click": clocks,
        "notated_32nd_notes_per_beat": notated,
        "time_sig_count": time_sig_count,
        "key": key,
        "tempo": tempo,
        "bpm": bpm.__round__(),
        "tempo_count": tempo_count,

    }

def get_note_analysis(df, ticks_per_beat):
    """
    Analyze note information from a DataFrame.

    :param df: DataFrame containing MIDI song messages
    :param ticks_per_beat: Number of ticks per beat in the MIDI file
    :return: Dictionary with metrics related to pitch, velocity, and duration of notes
    """

    # pitch metrics
    note_df = df[df["note"] > 0]
    note_series = note_df["note"]
    note_lowest = note_series.min()
    note_highest = note_series.max()
    note_avg = note_series.mean().round()
    note_variation_count = len(note_series.unique())

    # velocity metrics
    vel_series = df[df["velocity"] > 0]["velocity"]
    vel_lowest = vel_series.min()
    vel_highest = vel_series.max()
    vel_avg = vel_series.mean().round()
    vel_same = len(vel_series.unique()) == 1

    # duration metrics
    duration_series = note_df[note_df["time"] > 0]["time"]
    duration_lowest = duration_series.min()/ticks_per_beat
    duration_highest = duration_series.max()/ticks_per_beat
    duration_avg = duration_series.mean()/ticks_per_beat

    return {
        "note_lowest": note_lowest,
        "note_highest": note_highest,
        "note_avg": note_avg,
        "note_variation_count": note_variation_count,
        "vel_lowest": vel_lowest,
        "vel_highest": vel_highest,
        "vel_avg": vel_avg,
        "vel_same": vel_same,
        "duration_lowest": duration_lowest,
        "duration_highest": duration_highest,
        "duration_avg": duration_avg
    }

def monophonic_overlap_analysis(df):
    """
    Analyze monophonic and overlap information from a DataFrame.

    :param df: DataFrame containing MIDI song messages
    :return: Dictionary with monophonic and overlap information
    """
    channel_polyphonic = {channel:False for channel in df["channel"].unique()}
    channel_overlap = {channel:False for channel in df["channel"].unique()}
    for channel in channel_polyphonic:
        channel_df = df[df["channel"] == channel]
        notes_playing = 0
        for time_abs in channel_df["time_abs"].unique():
            time_df = channel_df[channel_df["time_abs"] == time_abs].sort_values("velocity")
            # check whether 2 notes start at the same time
            if len(time_df[time_df["velocity"] > 0]) > 1:
                channel_polyphonic[channel] = True
            # check whether 2 notes play at the same time
            for i,row in time_df.iterrows():
                if row["type"] == "note_on" and row["velocity"] > 0:
                    notes_playing = notes_playing + 1
                    if notes_playing > 1:
                        channel_overlap[channel] = True
                elif row["type"] == "note_off" or (row["type"] == "note_on" and row["velocity"] == 0):
                    notes_playing = notes_playing - 1
    return {
        "monophonic": False in channel_polyphonic.values(),
        "overlap": False not in channel_overlap.values()
    }

# create analysis

def compute_metrics_over_dataframe(df):
    """
    Compute metrics over a DataFrame.

    :param df: DataFrame to compute metrics on
    :return: DataFrame with mean, min, max values for each column
    """
    mean_row = {col: "" for col in df.columns}
    mean_row["name"] = "mean"
    min_row = {col: "" for col in df.columns}
    min_row["name"] = "min"
    max_row = {col: "" for col in df.columns}
    max_row["name"] = "max"


    for col in df.columns:
        if col in ["name","key","error"] :
            continue
        mean_row[col] = df[ df["error"] == False][col].mean()
        min_row[col] =  df[df["error"] == False][col].min()
        max_row[col] = df[ df["error"] == False][col].max()

    df = df.append(mean_row, ignore_index=True)
    df = df.append(min_row, ignore_index=True)
    df = df.append(max_row, ignore_index=True)
    return df

def analyse_data_folder(path, compute_metrics = True):
    """
    Analyze MIDI files in a given directory.

    :param path: Path of the directory containing MIDI files
    :param compute_metrics: Whether to compute mean, min, max values (default is True)
    :return: DataFrame with analysis results
    """
    files,_ = get_file_and_dirnames(path)
    df = pd.DataFrame()
    for file in tqdm(files):
        if file[-4:] != ".mid" and file[-5:] != ".midi":
            continue
        p = f"{path}/{file}"
        analysis = {"name": file}
        try:
            mid = MidiFile(p)
            analysis.update(get_midi_analysis(mid))
            mid_df, meta_df = midi2pandas(mid)
            analysis.update(get_meta_analysis(meta_df))
            analysis.update(get_note_analysis(mid_df, analysis["ticks_per_beat"]))
            analysis.update(monophonic_overlap_analysis(mid_df))
            analysis["error"] = False
        except:
            print("err")
            analysis.update({col:"error" for col in df.columns if col != "name"})
            analysis["error"] = True
        df = df.append(analysis, ignore_index=True)
    if compute_metrics:
        df = compute_metrics_over_dataframe(df)
    return df