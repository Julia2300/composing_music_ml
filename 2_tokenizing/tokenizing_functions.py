from os import walk
import miditoolkit

def get_file_and_dirnames(p):
    f = []
    d = []
    for (dirpath, dirnames, filenames) in walk(p):
        f.extend(filenames)
        d.extend(dirnames)
        break
    return f,d

def convert_to_note_items(path):
    midi_obj = miditoolkit.midi.parser.MidiFile(path)
    notes = midi_obj.instruments[0].notes
    notes.sort(key=lambda x: (x.start, x.pitch))
    note_items = []
    for note in notes:
        note_items.append({
            "name": "Note",
            "start": note.start,
            "end": note.end,
            "pitch": note.pitch
        })   
    note_items.sort(key=lambda x: x["start"])
    return note_items