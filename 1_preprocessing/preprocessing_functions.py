

from mido import Message, MidiFile, MidiTrack, MetaMessage
from music21 import pitch, converter, interval, midi

def change_to_4_4_metric_and_120_bpm(mid, ticks_per_beat):
    """
    converts time metric of a MIDI file to 4/4, cuts empty bars at beginning and changes tempo to 120 bpm

    :param mid: MIDI file in mido format
    :param ticks_per_beat: ticks per beat of MIDI file
    :return: MIDI file in mido format
    """
    mid_new = MidiFile()
    time_added = False
    tempo_added = False
    first_note = True
    for j in range(len(mid.tracks)):
        track = MidiTrack()
        for m in mid.tracks[j]:
            params=m.__dict__
            # replace time measure with 4/4
            if m.is_meta and params['type']=="time_signature":
                if not time_added:
                    track.append(MetaMessage(
                        'time_signature',
                        numerator=4,
                        denominator=4,
                        clocks_per_click=params["clocks_per_click"],
                        notated_32nd_notes_per_beat=params["notated_32nd_notes_per_beat"],
                        time=params["time"]
                    ))
                    time_added = True
                else:
                    continue
            # replace tempo with 120 bpm
            elif m.is_meta and params['type']=="set_tempo":
                if not tempo_added:
                    track.append(MetaMessage(
                        'set_tempo',
                        tempo=500000,
                        time=params["time"]
                    ))
                    tempo_added = True
                else:
                    continue
            # delete silent measures in the beginning
            elif params["type"]=="note_on" and first_note:
                track.append(Message(
                    'note_on',
                    channel=params["channel"],
                    note=params["note"],
                    velocity=params["velocity"],
                    time=int(params["time"] % (4*(ticks_per_beat)))
                ))
                first_note = False
            # keep other messages
            else:
                track.append(m)
        mid_new.tracks.append(track)
    return mid_new


def transpose_to_Cmaj_Amin(stream):
    """
    transposes music stream to Cmajor or Aminor

    :param stream: MIDI file in music21 stream format
    :return: transposed MIDI file in music21 stream format
    """
    music_analysis = stream.analyze('key')
    if music_analysis.mode == "major":
        i = interval.Interval(music_analysis.tonic, pitch.Pitch('C'))
    elif music_analysis.mode == "minor":
        i = interval.Interval(music_analysis.tonic, pitch.Pitch('A'))
    else:
        return "unknown scale"
    transposed_stream = stream.transpose(i)
    return transposed_stream


def transpose_to_octave_4_to_6(stream):
    """
    transposes music stream to start in octave 4

    :param stream: MIDI file in music21 stream format
    :return: transposed MIDI file in music21 stream format
    """
    lowest = stream.parts[0].pitches[0]
    for part in stream.parts:
        for note in part.pitches:
            if note < lowest:
                lowest = note

    transposing_octaves = 4 - lowest.octave
    for part in stream.parts:
        for note in part.pitches:
            note.transpose(transposing_octaves*12, inPlace=True)

    return stream