import os
import pickle
from collections import defaultdict

import networkx as nx
from mido import MidiFile


def __note_to_name(note):
    """Convert MIDI note number to note name."""
    notes = 'C C# D D# E F F# G G# A A# B'.split()
    return notes[note % 12] + str(note // 12 - 1)


def midi_to_graph(reconstruct_graph=False):
    if reconstruct_graph or not os.path.exists('midi2graph.pkl'):
        return __midi_to_graph()
    else:
        with open('midi2graph.pkl', 'rb') as f:
            midi2graph = pickle.load(f)
            return midi2graph


def __midi_to_graph(base_dir="./midi"):
    """
    Load the MIDI files from the base directory and convert them into a graph representation.
    This function ready the midi files directly so only thing you have to do is to just execute this function.
    """
    G = nx.Graph()
    directories = os.listdir(base_dir)
    if '.DS_Store' in directories: directories.remove('.DS_Store')  # For macOS

    for directory in directories:
        path = os.path.join(base_dir, directory)
        for file in os.listdir(path):
            if file.endswith('.mid') or file.endswith('.midi'):
                midi_path = os.path.join(path, file)
                __process_midi(midi_path, G)
    with open('midi2graph.pkl', 'wb') as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    return G


def __process_midi(midi_path, G):
    """
    Process a single MIDI file and update the graph G with nodes and edges.
    """
    mid = MidiFile(midi_path)
    current_program = {channel: 0 for channel in range(16)}  # Default program for each channel
    note_starts = defaultdict(list)  # Track start times of notes

    for i, track in enumerate(mid.tracks):
        time = 0  # Track cumulative time in ticks
        for msg in track:
            time += msg.time  # Increment time by delta-time of the message

            # Process tempo changes
            if msg.type == 'set_tempo':
                current_tempo = msg.tempo
                tempo_node = f'tempo-{current_tempo}'
                G.add_node(tempo_node, type='tempo', tempo=current_tempo)

            # Process instrument (program) changes
            if msg.type == 'program_change':
                current_program[msg.channel] = msg.program
                program_node = f'program-{msg.channel}-{msg.program}'
                G.add_node(program_node, type='program', program=msg.program, channel=msg.channel)

            # Process time signature changes (simplified)
            if msg.type == 'time_signature':
                current_time_signature = f'{msg.numerator}/{msg.denominator}'
                ts_node = f'ts-{current_time_signature}'
                G.add_node(ts_node, type='time_signature', time_signature=current_time_signature)

            # Process notes
            if msg.type == 'note_on' and msg.velocity > 0:  # note_on with velocity 0 can be treated as note_off
                note_name = f'note-{msg.note}-{msg.channel}'
                G.add_node(note_name, type='note', note=msg.note, velocity=msg.velocity, channel=msg.channel)
                # Connect note to its program (instrument)
                program_node = f'program-{msg.channel}-{current_program[msg.channel]}'
                G.add_edge(note_name, program_node)
                note_starts[time].append(note_name)
                # Optionally, connect note to current tempo and time signature
                G.add_edge(note_name, tempo_node)
                G.add_edge(note_name, ts_node)

    # Detect and process chords
    for start_time, notes in note_starts.items():
        if len(notes) > 1:  # More than one note at the same start time indicates a chord
            chord_name = '-'.join(sorted([__note_to_name(int(n.split('-')[1])) for n in notes]))
            chord_node = f'chord-{start_time}-{chord_name}'
            G.add_node(chord_node, type='chord', notes=chord_name, time=start_time)
            for note in notes:
                G.add_edge(chord_node, note)


def __get_songs(BASE_DIR, directories):
    for directory in directories:
        path = os.path.join(BASE_DIR, directory)
        for file in os.listdir(path):
            if file.endswith(".mid") or file.endswith(".midi"):
                yield MidiFile(os.path.join(path, file))
