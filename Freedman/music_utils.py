# music_utils.py
# Reusable constants, helper functions, and visualization functions
# for tonality and cadence analysis with CRIM Intervals.
#
# Usage in notebook:
#   import importlib, music_utils
#   importlib.reload(music_utils)
#   from music_utils import *

import numpy as np
import pandas as pd
import re

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =============================================================================
# CONSTANTS
# =============================================================================

pitch_order = ['A#1', 'B1',
    'C2', 'C#2', 'D2', 'D#2','E-2', 'E2', 'E#2', 'F2', 'F#2', 'G-2', 'G2', 'G#2', 'A-2', 'A2', 'A#2','B-2', 'B2', 'B#2',
    'C3', 'C#3', 'D-3','D3', 'D#3', 'E-3','E3', 'E#3', 'F3', 'F#3', 'G-3', 'F##3', 'G3', 'G#3', 'A-3', 'A3', 'A#3', 'B-3','B3', 'B#3',
    'C4', 'C#4', 'D-4','D4', 'D#4','E-4', 'E4', 'F-4', 'E#4', 'F4', 'F#4', 'G-4', 'F##4', 'G4', 'G#4', 'A-4','A4', 'A#4', 'B-4', 'B4', 'B#4',
    'C5', 'C#5','C##5', 'D-5','D5', 'D#5', 'E-5','E5', 'F-5','E#5','F5', 'F#5', 'G-5', 'F##5','G5', 'G#5', 'A-5', 'A5', 'A#5', 'B-5', 'B5',
    'C6']

recta_order = ['D2', 'E-2', 'E2', 'F2', 'F#2', 'G2', 'G#2', 'A2', 'B-2', 'B2',
               'C3', 'C#3','D3', 'D#3', 'E-3','E3', 'F3', 'F#3',  'G3', 'G#3', 'A-3', 'A3', 'B-3','B3',
               'C4', 'C#4', 'D-4','D4', 'D#4','E-4', 'E4', 'F4', 'F#4', 'G-4',  'G4', 'G#4', 'A-4','A4',  'B-4', 'B4',
               'C5', 'C#5','D-5','D5', 'D#5', 'E-5','E5','F5', 'F#5', 'G-5', 'G5', 'G#5', 'A-5', 'A5',  'B-5', 'B5',
               'C6']

pitch_class_order_no_rests = ['C', 'C#', 'D-','D', 'D#', 'E-', 'E', 'F-', 'E#', 'F', 'F#', 'G-', 'F##', 'G', 'G#', 'A-','A', 'A#', 'B-', 'B', 'B#']
pitch_class_order_with_rests = pitch_class_order_no_rests + ['Rest']
pitch_class_order = pitch_class_order_with_rests

category_order = {
    'C': 0, 'C#': 1, 'Db': 2, 'D': 3, 'D#': 4, 'Eb': 5, 'E': 6, 'F': 7, 'F#': 8,
    'G': 9, 'G#': 10, 'A': 11, 'A#': 12, 'Bb': 13, 'B': 14, 'Rest': 15
}

REST_TOKENS = {'r', 'rest', 'Rest', '-', ''}

NOTE_PC_CHROMATIC = ['C', 'C#', 'D-', 'D', 'D#', 'E-', 'E', 'F', 'F#', 'G-', 'G', 'G#', 'A-', 'A', 'A#', 'B-', 'B', 'F-', 'E#', 'F##', 'B#', 'Rest']
NOTE_PC_FIFTHS    = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'A-', 'E-', 'B-', 'F', 'G#', 'D#', 'A#', 'D-', 'G-', 'F-', 'E#', 'B#', 'F##', 'Rest']
NOTE_PC_DIATONIC  = ['C', 'D', 'E', 'F', 'G', 'A', 'B', 'B-', 'E-', 'A-', 'D-', 'C#', 'F#', 'G#', 'D#', 'A#', 'G-', 'F-', 'E#', 'F##', 'B#', 'Rest']
NOTE_PC_ORDERS    = {'Chromatic': NOTE_PC_CHROMATIC, 'Fifths': NOTE_PC_FIFTHS, 'Diatonic': NOTE_PC_DIATONIC}

contrasting_colors = [
    '#636EFA', '#DC267F', '#009E73', '#FFB000', '#977277',
    '#EC4899', '#48BB78', '#ED8936', '#2563EB', '#8338EC',
    '#FF922B', '#06D6A0', '#EF4444', '#F97316', '#84CC16',
    '#3B82F6', '#A855F7', '#22C55E', '#EA580C', '#94A3B8',
]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def standardize_note(note):
    if '-' in note:
        return note.replace('-', 'b')
    return note


def extract_letter(value, include_rests=True):
    """Extract pitch class from a note string."""
    if pd.isna(value):
        return None
    s = str(value).strip()
    if s.lower() in REST_TOKENS:
        return 'Rest' if include_rests else None
    return s.rstrip('0123456789')


def extract_non_numeric(val):
    """Extract non-numeric characters excluding 'Rest'."""
    if not isinstance(val, str):
        return val
    result = ''.join(re.findall(r'[a-zA-Z]+', val)).replace('Rest', '')
    return result if result else val


def combine_columns(x_col, y_col):
    """Combine two columns with underscores, keeping only pairs where both sides have values."""
    x_values = x_col.fillna('')
    y_values = y_col.fillna('')
    combined = []
    indices = []
    for idx, (x, y) in enumerate(zip(x_values, y_values)):
        if x and y:
            combined.append(f"{x}_{y}")
            indices.append(idx)
    return pd.Series(combined, index=indices)


def filter_by_percentage(df, column, threshold_percent):
    """Filter DataFrame to keep only rows where the value appears in at least threshold_percent of total rows."""
    value_counts = df[column].value_counts()
    total_rows = len(df)
    value_percentages = (value_counts / total_rows * 100).round(2)
    values_to_keep = value_percentages[value_percentages >= threshold_percent].index
    return df[df[column].isin(values_to_keep)].copy()


# =============================================================================
# SPECIES ANALYSIS
# =============================================================================

def find_species(piece, limit_to_entries=True, species_intervals=['5', '-5', '4', '-4']):
    notes = piece.notes()
    notes = notes.map(extract_non_numeric)
    notes_numb = piece.numberParts(df=notes)
    mel = piece.melodic(kind='d', end=False)
    mel_numb = piece.numberParts(df=mel)

    if limit_to_entries:
        mel_numb = piece.entries(mel_numb)

    mel_stacked = mel_numb.stack()
    filtered_mel = mel_stacked[mel_stacked.isin(species_intervals)]
    filtered_mel = filtered_mel.unstack().fillna('')
    filtered_nr = notes_numb.where(filtered_mel.notna())

    filtered_nr.columns = filtered_nr.columns.astype(str)
    filtered_mel.columns = filtered_mel.columns.astype(str)

    merged_df = pd.merge(filtered_nr, filtered_mel, left_index=True, right_index=True, how='left')

    result_df = pd.DataFrame(index=merged_df.index)
    for base_col in filtered_nr.columns:
        if f'{base_col}_x' in merged_df.columns:
            result_df[base_col] = combine_columns(merged_df[f'{base_col}_x'], merged_df[f'{base_col}_y'])
        elif base_col in merged_df.columns:
            result_df[base_col] = merged_df[base_col]
        else:
            result_df[base_col] = ''

    result_df_numbered = piece.detailIndex(result_df, measure=True, beat=False, offset=False, progress=True)
    result_df_numbered['composer'] = piece.metadata['composer']
    result_df_numbered['title'] = piece.metadata['title']
    result_df_numbered = result_df_numbered.reset_index()

    species_df = pd.melt(
        result_df_numbered,
        id_vars=['composer', 'title', 'Measure', 'Progress'],
        var_name='Voice',
        value_name='Interval',
        value_vars=[col for col in result_df_numbered.columns if col not in ['Measure', 'Progress', 'composer', 'title']]
    )
    species_df = species_df[species_df['Interval'].astype(bool)]

    four_five_dict = {
        'C_5':'C/G', 'C_-5':'C/F', 'C_4':'C/F', 'C_-4':'C/G',
        'D_5':'D/A', 'D_-5':'D/G', 'D_4':'D/G', 'D_-4':'D/A',
        'E_5':'E/B', 'E_-5':'E/A', 'E_4':'E/A', 'E_-4':'E/B',
        'F_5':'F/C', 'F_-5':'F/B-', 'F_4':'F/B-', 'F_-4':'F/C',
        'G_5':'G/D', 'G_-5':'G/C', 'G_4':'G/C', 'G_-4':'G/D',
        'A_5':'A/E', 'A_-5':'A/D', 'A_4':'A/D', 'A_-4':'A/E',
        'B_5':'B/F#', 'B_-5':'B/E', 'B_4':'B/E', 'B_-4':'B/F#',
        'B-_5':'B-/F', 'B-_-5':'B-/E-', 'B-_4':'B-/E-', 'B-_-4':'B-/F',
    }
    octave_dict = {
        'C/G':'CGC', 'C/F':'FCF', 'D/A':'DAD', 'D/G':'GDG',
        'E/B':'EBE', 'E/A':'AEA', 'F/C':'FCF', 'F/B-':'BbFBb',
        'G/D':'GDG', 'G/C':'CGC', 'A/E':'AEA', 'A/D':'DAD',
        'B/F#':'BF#B', 'B/E':'EBE', 'B-/F':'BbFBb', 'B-/E-':'EbBbEb'
    }

    species_df['Species'] = species_df['Interval'].map(four_five_dict)
    species_df['Octave'] = species_df['Species'].map(octave_dict)
    return species_df.reset_index(drop=True)


# CADENCE CHARTS

def cadence_bar_charts(cadences, cadences_brief, contrasting_colors):

    # Bar Charts for Cadence Types and Tones by Piece (sorted by prevalence)
    cadtype_by_piece = cadences_brief.groupby(['Title', 'CadType']).size().reset_index(name='Count')
    tone_by_piece = cadences_brief.groupby(['Title', 'Tone']).size().reset_index(name='Count')

    # Calculate percentages
    cadtype_totals = cadtype_by_piece.groupby('Title')['Count'].sum()
    cadtype_by_piece_pct = cadtype_by_piece.copy()
    cadtype_by_piece_pct['Percentage'] = cadtype_by_piece_pct.apply(
        lambda row: (row['Count'] / cadtype_totals[row['Title']]) * 100, axis=1
    )

    tone_totals = tone_by_piece.groupby('Title')['Count'].sum()
    tone_by_piece_pct = tone_by_piece.copy()
    tone_by_piece_pct['Percentage'] = tone_by_piece_pct.apply(
        lambda row: (row['Count'] / tone_totals[row['Title']]) * 100, axis=1
    )

    # Get overall prevalence of CadTypes across all pieces
    cadtype_overall = cadences['CadType'].value_counts().sort_values(ascending=False)
    cadtype_order = cadtype_overall.index.tolist()

    # Create bar chart for CadType by piece
    fig_cadtype_bar = px.bar(
        cadtype_by_piece,
        x='CadType',
        y='Count',
        color='Title',
        title='Cadence Types by Piece (sorted by overall prevalence)',
        category_orders={'CadType': cadtype_order},
        color_discrete_sequence=contrasting_colors[:len(cadtype_by_piece['Title'].unique())],
        barmode='group'
    )

    fig_cadtype_bar.update_layout(
        xaxis_title='Cadence Type',
        yaxis_title='Count',
        legend_title='Piece',
        width=1000,
        height=600
    )

    fig_cadtype_bar.show()

    # Now create bar chart for Tone by piece
    tone_overall = cadences['Tone'].value_counts().sort_values(ascending=False)
    tone_order = tone_overall.index.tolist()

    fig_tone_bar = px.bar(
        tone_by_piece,
        x='Tone',
        y='Count',
        color='Title',
        title='Cadence Tones by Piece (sorted by overall prevalence)',
        category_orders={'Tone': tone_order},
        color_discrete_sequence=contrasting_colors[:len(tone_by_piece['Title'].unique())],
        barmode='group'
    )

    fig_tone_bar.update_layout(
        xaxis_title='Tone',
        yaxis_title='Count',
        legend_title='Piece',
        width=1000,
        height=600
    )

    fig_tone_bar.show()

    # Also create percentage versions
    print("\nCreating percentage versions...")

    # CadType percentages
    fig_cadtype_pct = px.bar(
        cadtype_by_piece_pct,
        x='CadType',
        y='Percentage',
        color='Title',
        title='Cadence Types by Piece (Percentages, sorted by overall prevalence)',
        category_orders={'CadType': cadtype_order},
        color_discrete_sequence=contrasting_colors[:len(cadtype_by_piece_pct['Title'].unique())],
        barmode='group'
    )

    fig_cadtype_pct.update_layout(
        xaxis_title='Cadence Type',
        yaxis_title='Percentage',
        legend_title='Piece',
        width=1000,
        height=600
    )

    fig_cadtype_pct.show()

    # Tone percentages
    fig_tone_pct = px.bar(
        tone_by_piece_pct,
        x='Tone',
        y='Percentage',
        color='Title',
        title='Cadence Tones by Piece (Percentages, sorted by overall prevalence)',
        category_orders={'Tone': tone_order},
        color_discrete_sequence=contrasting_colors[:len(tone_by_piece_pct['Title'].unique())],
        barmode='group'
    )

    fig_tone_pct.update_layout(
        xaxis_title='Tone',
        yaxis_title='Percentage',
        legend_title='Piece',
        width=1000,
        height=600
    )

    fig_tone_pct.show()



# =============================================================================
# MODE / RANGE ANALYSIS
# =============================================================================

def find_mode_range(df, final_value, voice_value, top_n=7):
    """
    Finds the range of notes in a given voice to help distinguish modal types.

    df:          corpus notes DataFrame (from corpus_note_durs with pitch_class=False)
    final_value: the final tone of each piece
    voice_value: which voice to check
    top_n:       number of highest-percentage notes to use for range calculation
    """
    filtered_df = df[
        (df['Final'] == final_value) &
        (df['Voice'] == voice_value) &
        (df['Notes'] != 'Rest')
    ]

    top_durations = filtered_df.groupby(['Title', 'Voice'], observed=True).apply(
    lambda x: x.nlargest(top_n, 'Percentage'), include_groups=False).reset_index(level=['Title', 'Voice']).reset_index(drop=True)

    # Build note-position mapping
    base_notes = ['C', 'D', 'E', 'F', 'G', 'A', 'B', 'B-']
    octaves = list(range(2, 6))
    note_positions = {}
    position = 0
    for octave in octaves:
        for note in base_notes:
            note_positions[f"{note}{octave}"]  = position
            note_positions[f"{note}-{octave}"] = position - 0.5
            note_positions[f"{note}b{octave}"] = position - 0.5
            note_positions[f"{note}#{octave}"] = position + 0.5
            position += 1

    def get_note_position(note):
        if note in note_positions:
            return note_positions[note]
        if '-' in note:
            parts = note.split('-')
            if len(parts) == 2:
                alt = f"{parts[0]}-{parts[1]}"
                if alt in note_positions:
                    return note_positions[alt]
        if '#' in note:
            parts = note.split('#')
            if len(parts) == 2:
                alt = f"{parts[0]}#{parts[1]}"
                if alt in note_positions:
                    return note_positions[alt]
        print(f"Warning: Could not determine position for note: {note}")
        return -1000

    top_durations['NotePosition'] = top_durations['Notes'].apply(get_note_position)

    result = []
    for (title, voice), group in top_durations.groupby(['Title', 'Voice'], observed=True):
        if len(group) > 0:
            sorted_group = group.sort_values('NotePosition')
            lowest_note  = sorted_group.iloc[0]['Notes']
            highest_note = sorted_group.iloc[-1]['Notes']
            title_rows   = df[df['Title'] == title]
            composer = title_rows['Composer'].iloc[0] if not title_rows.empty else 'Unknown'
            final    = title_rows['Final'].iloc[0]    if not title_rows.empty else 'Unknown'
            result.append({
                'Composer':    composer,
                'Title':       title,
                'Voice':       voice,
                'Final':       final,
                'LowestNote':  lowest_note,
                'HighestNote': highest_note,
                'Range':       f"{lowest_note} to {highest_note}"
            })

    return pd.DataFrame(result)

# Voice Range Chart

def voice_range_chart(corpus_pitch_data, recta_order, top_n=7):

    final_value = corpus_pitch_data['Final'].iloc[0]
    designated_voice = '2'

    # Run find_mode_range for every voice and combine
    voices = corpus_pitch_data['Voice'].unique()
    all_voice_ranges = []
    for voice in voices:
        voice_df = find_mode_range(corpus_pitch_data, final_value=final_value, voice_value=voice, top_n=top_n)
        if not voice_df.empty:
            all_voice_ranges.append(voice_df)

    corpus_voice_ranges = pd.concat(all_voice_ranges, ignore_index=True)

    # Add positional columns needed by the chart
    corpus_voice_ranges['Lowest_pos'] = corpus_voice_ranges['LowestNote'].apply(
        lambda n: recta_order.index(n) if n in recta_order else 0
    )
    corpus_voice_ranges['Highest_pos'] = corpus_voice_ranges['HighestNote'].apply(
        lambda n: recta_order.index(n) if n in recta_order else 0
    )
    corpus_voice_ranges['Range_width'] = (
        corpus_voice_ranges['Highest_pos'] - corpus_voice_ranges['Lowest_pos']
    )

    # Force voice to string for chart labels
    corpus_voice_ranges['Voice'] = corpus_voice_ranges['Voice'].astype(str)

    # Movement order: titles without a movement keyword are chanson models → rank 0
    mass_movements = ['Kyrie', 'Gloria', 'Credo', 'Sanctus', 'Agnus Dei']

    def title_rank(title):
        for i, movement in enumerate(mass_movements):
            if movement in title:
                return i + 1
        return 0  # no movement keyword → chanson model, sorts first

    # Find lowest note of designated voice per piece, for sorting within groups
    designated_voice_notes = {}
    for _, row in corpus_voice_ranges.iterrows():
        if row['Voice'] == designated_voice:
            designated_voice_notes[row['Title']] = row['LowestNote']

    sorted_titles = sorted(
        designated_voice_notes.keys(),
        key=lambda x: (
            title_rank(x),                                # primary: model first, then movements
            recta_order.index(designated_voice_notes[x])  # secondary: lowest note
        )
    )

    # Create combined Title+Voice label, voices sorted numerically within each piece
    corpus_voice_ranges['Title_Voice'] = (
        corpus_voice_ranges['Title'] + '  [V' + corpus_voice_ranges['Voice'] + ']'
    )

    sorted_title_voice = []
    for title in sorted_titles:
        piece_rows = corpus_voice_ranges[corpus_voice_ranges['Title'] == title]
        for voice in sorted(piece_rows['Voice'].unique(), key=lambda v: int(v)):
            label = f"{title}  [V{voice}]"
            if label not in sorted_title_voice:
                sorted_title_voice.append(label)

    # Create figure
    fig = px.bar(
        corpus_voice_ranges,
        x='Range_width',
        y='Title_Voice',
        base='Lowest_pos',
        orientation='h',
        color='Voice',
        color_discrete_map={
            '1': 'rgba(255, 99, 71, 0.5)',
            '2': 'rgba(54, 162, 235, 0.5)',
            '3': 'rgba(100, 20, 100, 0.3)'
        },
        hover_data=['Title', 'Voice', 'Range'],
        category_orders={'Title_Voice': sorted_title_voice},
        labels={'Title_Voice': 'Piece / Voice'}
    )

    _ref_by_final = {
        'D': ['D3', 'G3', 'D4', 'G4', 'D5', 'G5'],
        'E': ['E3', 'C3', 'E4', 'C4', 'E5', 'C5'],
        'F': ['C3', 'F3', 'C4', 'F4', 'C5', 'F5'],
        'G': ['D3', 'G3', 'D4', 'G4', 'D5', 'G5'],
        'A': ['E3', 'A3', 'E4', 'A4', 'E5', 'A5'],
        'C': ['C3', 'G3', 'C4', 'G4', 'C5', 'G5'],
    }
    reference_points = _ref_by_final.get(final_value, ['D3', 'G3', 'D4', 'G4', 'D5', 'G5'])
    shapes = []
    annotations = []
    for point in reference_points:
        if point in recta_order:
            x_position = recta_order.index(point)
            shapes.append(dict(
                type='line',
                x0=x_position, y0=0,
                x1=x_position, y1=len(sorted_title_voice),
                line=dict(dash='dot', color='gray', width=1)
            ))
            annotations.append(dict(
                x=x_position,
                y=len(sorted_title_voice) - 1,
                text=point,
                showarrow=False,
                font=dict(size=12)
            ))

    fig.update_layout(
        title=f'Voice Ranges in Pieces with Final {final_value}',
        xaxis=dict(
            title='Note Range',
            tickmode='array',
            tickvals=list(range(len(recta_order))),
            ticktext=recta_order,
            tickangle=45
        ),
        yaxis=dict(
            title='',
            autorange='reversed',
            categoryorder='array',
            categoryarray=sorted_title_voice
        ),
        height=900,
        width=1000,
        bargap=0.3,
        bargroupgap=0.1,
        margin=dict(l=250, r=100, t=100, b=50),
        shapes=shapes,
        annotations=annotations
    )

    fig.show()
    fig.write_html("charts/corpus_voice_ranges.html")

# =============================================================================
# VISUALIZATION: RADAR PLOTS
# =============================================================================

def radar_note_plot(weighted_notes_df, pitch_class_order='Fifths',
                    exclude_rests=True, limit_to_active=True, color_grouping='title'):
    if exclude_rests:
        weighted_notes_df = weighted_notes_df[weighted_notes_df['pitch_class'] != 'Rest']
    if limit_to_active:
        weighted_notes_df = weighted_notes_df[weighted_notes_df['scaled'] > 0]

    pc_order = NOTE_PC_ORDERS[pitch_class_order]
    active_pcs = [pc for pc in pc_order if pc in weighted_notes_df['pitch_class'].values]

    weighted_notes_df = weighted_notes_df.copy()
    weighted_notes_df['pitch_class'] = pd.Categorical(
        weighted_notes_df['pitch_class'], categories=active_pcs, ordered=True
    )
    weighted_notes_df = weighted_notes_df.sort_values('pitch_class')

    fig = px.line_polar(
        weighted_notes_df,
        r='scaled',
        theta='pitch_class',
        color=color_grouping,
        line_close=True,
        range_r=[0, weighted_notes_df['scaled'].max() * 1.1],
        markers=True,
        category_orders={
            'pitch_class': active_pcs,
            color_grouping: list(weighted_notes_df[color_grouping].unique())
        },
        color_discrete_sequence=contrasting_colors[:len(weighted_notes_df[color_grouping].unique())]
    )
    fig.update_traces(fill='toself', mode='markers+lines', opacity=.7)
    fig.update_layout(
        showlegend=True,
        legend=dict(title=color_grouping, orientation='h', yanchor='bottom', y=-0.5, xanchor='right', x=1),
        height=600, width=600,
        title=f'Weighted Note Distribution in Corpus ({pitch_class_order} order)'
    )
    fig.show()
    fig.write_html("charts/radar_note_plot.html")


def cadence_radar(cadences, tone_ordering='Thirds', limit_to_active=False):
    FIFTHS    = {'C':0,'G':1,'D':2,'A':3,'E':4,'B':5,'F#':6,'C#':7,'A-':8,'E-':9,'B-':10,'F':11}
    THIRDS    = {'C':0,'E':1,'G':2,'B-':3,'D':4,'F':5,'A':6,'C#':7}
    CHROMATIC = {'C':0,'C#':1,'D-':1,'D':2,'E-':3,'E':4,'F':5,'F#':6,'G-':6,'G':7,'A-':8,'A':9,'B-':10,'B':11}
    CATEGORY_ORDER = FIFTHS if tone_ordering == 'Fifths' else THIRDS if tone_ordering == 'Thirds' else CHROMATIC

    grouped = cadences.groupby(['Title', 'Tone']).size().reset_index(name='count')
    title_sums = grouped.groupby('Title')['count'].sum()
    grouped['Percentage'] = (grouped['count'] / grouped['Title'].map(title_sums)) * 100

    tone_order  = sorted(CATEGORY_ORDER.keys(), key=lambda x: CATEGORY_ORDER[x])
    active_tones = [t for t in tone_order if t in grouped['Tone'].values] if limit_to_active else tone_order

    full_index = pd.MultiIndex.from_product([grouped['Title'].unique(), active_tones], names=['Title', 'Tone'])
    grouped = (
        grouped[grouped['count'] > 0]
        .set_index(['Title', 'Tone'])[['count', 'Percentage']]
        .reindex(full_index, fill_value=0)
        .reset_index()
    )
    if limit_to_active:
        grouped = grouped[grouped['count'] > 0]

    tone_rank = {t: i for i, t in enumerate(active_tones)}
    grouped = grouped.assign(_sort=grouped['Tone'].map(tone_rank)).sort_values(['Title', '_sort']).drop(columns='_sort').reset_index(drop=True)
    grouped['Tone'] = pd.Categorical(grouped['Tone'], categories=active_tones, ordered=True)

    fig = px.line_polar(grouped, r='Percentage', theta='Tone', line_close=True,
                        color='Title', markers=True, category_orders={'Tone': active_tones})
    fig.update_traces(fill='toself', line=dict(width=2))
    fig.update_layout(
        width=800, height=600,
        legend=dict(orientation='h', yanchor='bottom', y=-0.4, xanchor='center', x=0.5,
                    title=dict(text='Titles', side='top', font_size=12),
                    itemsizing='constant', itemwidth=30,
                    bordercolor='black', borderwidth=1, bgcolor='rgba(255,255,255,0.8)'),
        title=dict(text=f'Relative Distribution of Cadence Tones in Corpus ordered by {tone_ordering}', x=0.5),
        polar=dict(radialaxis=dict(visible=True, title='Percentage')),
        margin=dict(b=120)
    )
    fig.show()
    fig.write_html("charts/radar_cadence_plot.html")


# =============================================================================
# VISUALIZATION: SPECIES + CADENCES
# =============================================================================

def plot_species_cadences(species_df, cadences_df, composer, title):
    """
    Plot species intervals and cadences by progress in a two-panel chart.

    Args:
        species_df:   output of find_species()
        cadences_df:  piece.cadences() with 'Composer' and 'Title' columns added
        composer:     piece composer string (for chart title)
        title:        piece title string (for chart title)
    """
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=('Species: Octave by Progress', 'Cadences: Tone by Progress')
    )

    # Normalize Progress to 0–1 for both datasets so they share the same x-axis scale
    species_df  = species_df.copy()
    cadences_df = cadences_df.copy()
    for df, col in [(species_df, 'Progress'), (cadences_df, 'Progress')]:
        lo, hi = df[col].min(), df[col].max()
        if hi > lo:
            df[col] = (df[col] - lo) / (hi - lo)

    octave_order = ['EbBbEb', 'BbFBb', 'FCF', 'CGC', 'GDG', 'DAD', 'AEA', 'EBE', 'BF#B']
    fig_sp = px.scatter(
        species_df, x='Progress', y='Octave', color='Voice',
        hover_data=['composer', 'title', 'Measure', 'Interval', 'Species'],
        color_discrete_sequence=contrasting_colors[:len(species_df['Voice'].unique())],
        category_orders={'Octave': octave_order}
    )
    for trace in fig_sp.data:
        trace.showlegend = True
        fig.add_trace(trace, row=1, col=1)
    fig.update_yaxes(title_text='Octave', categoryorder='array', categoryarray=octave_order, row=1, col=1)

    tone_order = ['B-', 'F', 'C', 'G', 'D', 'A', 'E']
    fig_cad = px.scatter(
        cadences_df, x='Progress', y='Tone', color='CadType',
        hover_data=['Composer', 'Title', 'CadType', 'Tone'],
        color_discrete_sequence=contrasting_colors[len(species_df['Voice'].unique()):],
        category_orders={'Tone': tone_order}
    )
    for trace in fig_cad.data:
        trace.marker.symbol = 'diamond'
        trace.marker.size = 10
        trace.marker.opacity = 0.8
        trace.showlegend = True
        fig.add_trace(trace, row=2, col=1)

    fig.update_traces(marker=dict(size=8, opacity=0.7), row=1, col=1)
    fig.update_yaxes(title_text='Tone', categoryorder='array', categoryarray=tone_order, row=2, col=1)
    fig.update_yaxes(title_text='Octave', tickmode='linear', dtick=1, row=1, col=1)

    x_min = min(species_df['Progress'].min(), cadences_df['Progress'].min())
    x_max = max(species_df['Progress'].max(), cadences_df['Progress'].max())
    fig.update_xaxes(range=[x_min, x_max], autorange=False)
    fig.update_xaxes(title_text='Progress', row=2, col=1)
    fig.update_layout(
        height=600, width=800,
        legend=dict(x=1.02, y=1.0, xanchor='left', yanchor='top'),
        margin=dict(r=200),
        title=f'Species and Cadences by Progress in {composer} : {title}',
        legend_title='Legend'
    )
    fig.show()


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

def kl_divergence(p, q):
    """Kullback-Leibler divergence between two distributions."""
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)
    epsilon = 1e-10
    p = np.where(p == 0, epsilon, p)
    q = np.where(q == 0, epsilon, q)
    return np.sum(p * np.log(p / q))


def js_divergence(p, q):
    """Jensen-Shannon divergence between two distributions."""
    m = (p + q) / 2
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def calculate_js_divergence_matrix(df, group_col, value_col, piece_col='Title'):
    """Pairwise Jensen-Shannon divergence matrix for cadence distributions."""
    all_categories = df[group_col].unique()
    pieces = df[piece_col].unique()

    piece_distributions = {}
    for piece in pieces:
        dist = df[df[piece_col] == piece].groupby(group_col)[value_col].sum()
        for cat in all_categories:
            if cat not in dist.index:
                dist[cat] = 0
        total = dist.sum()
        if total > 0:
            dist = dist / total
        piece_distributions[piece] = dist.sort_index()

    js_matrix = pd.DataFrame(index=pieces, columns=pieces, dtype=float)
    for i, piece1 in enumerate(pieces):
        for j, piece2 in enumerate(pieces):
            if i <= j:
                js_div = js_divergence(piece_distributions[piece1].values, piece_distributions[piece2].values)
                js_matrix.loc[piece1, piece2] = js_div
                js_matrix.loc[piece2, piece1] = js_div
    return js_matrix


def calculate_js_notes_matrix(weighted_notes_df, piece_col='title'):
    """Pairwise Jensen-Shannon divergence matrix for pitch class distributions."""
    all_pitch_classes = weighted_notes_df['pitch_class'].unique()
    pieces = weighted_notes_df[piece_col].unique()

    piece_distributions = {}
    for piece in pieces:
        dist = weighted_notes_df[weighted_notes_df[piece_col] == piece].set_index('pitch_class')['scaled']
        for pc in all_pitch_classes:
            if pc not in dist.index:
                dist[pc] = 0
        piece_distributions[piece] = dist.sort_index()

    js_matrix = pd.DataFrame(index=pieces, columns=pieces, dtype=float)
    for i, piece1 in enumerate(pieces):
        for j, piece2 in enumerate(pieces):
            if i <= j:
                js_div = js_divergence(piece_distributions[piece1].values, piece_distributions[piece2].values)
                js_matrix.loc[piece1, piece2] = js_div
                js_matrix.loc[piece2, piece1] = js_div
    return js_matrix


def classical_mds(distance_matrix, n_components=2):
    """Classical multidimensional scaling."""
    D = distance_matrix.values
    n = D.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ (D ** 2) @ H
    eigenvals, eigenvecs = np.linalg.eigh(B)
    idx = np.argsort(eigenvals)[::-1]
    eigenvals, eigenvecs = eigenvals[idx], eigenvecs[:, idx]
    pos_idx = eigenvals > 1e-8
    eigenvals, eigenvecs = eigenvals[pos_idx], eigenvecs[:, pos_idx]
    coords = eigenvecs[:, :n_components] @ np.diag(np.sqrt(eigenvals[:n_components]))
    return coords


def create_mds_plot(distance_matrix, title, piece_abbrev=None):
    """Create MDS scatter plot from a distance matrix.

    piece_abbrev: optional dict mapping full piece names to short labels.
    """
    mds_coords = classical_mds(distance_matrix)
    mds_df = pd.DataFrame({
        'MDS1':  mds_coords[:, 0],
        'MDS2':  mds_coords[:, 1],
        'Piece': distance_matrix.index
    })
    if piece_abbrev:
        mds_df['Abbrev'] = mds_df['Piece'].map(piece_abbrev)
    else:
        mds_df['Abbrev'] = mds_df['Piece']

    fig = px.scatter(mds_df, x='MDS1', y='MDS2', text='Abbrev',
                     title=title, hover_data=['Piece'], color='Abbrev',
                     color_discrete_sequence=contrasting_colors[:len(mds_df)])
    fig.update_traces(textposition='top center', marker=dict(size=12))
    fig.update_layout(xaxis_title='MDS Dimension 1', yaxis_title='MDS Dimension 2',
                      showlegend=False, width=700, height=600)
    return fig

