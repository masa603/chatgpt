"""Draw the two-button circuit from the supplied reference and frozen pin map.

This is a new vector schematic.  The supplied image is a circuit reference,
not a background to trace.  Component values were confirmed by the user.
"""

import hashlib
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle

INK = '#23313f'
WIRE = '#304f69'
MUTED = '#536579'


def build_circuit_figure(folder, source):
    folder, source = Path(folder), Path(source)
    folder.mkdir(parents=True, exist_ok=True)
    program = source.read_text()
    expected = {'SENSOR_PIN': 'A0', 'LED_R_PIN': '8', 'LED_G_PIN': '6',
                'LED_B_PIN': '7', 'BUTTON_CALIB_PIN': '2', 'BUTTON_READ_PIN': '3',
                'USE_PHYSICAL_BUTTONS': '0'}
    for name, value in expected.items():
        assert re.search(r'^#define\s+' + name + r'\s+' + value + r'\b', program, re.M)
    assert 'pinMode(pin, INPUT_PULLUP)' in program
    assert 'button.stableState == LOW' in program
    plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'font.size': 12,
                         'pdf.fonttype': 3, 'svg.fonttype': 'path'})
    fig, ax = plt.subplots(figsize=(11.0, 6.2))
    ax.set(xlim=(.05, 16.35), ylim=(.65, 9.55), aspect='equal')
    ax.axis('off')

    def line(*points, color=WIRE, lw=1.5, **kwargs):
        ax.plot([p[0] for p in points], [p[1] for p in points], color=color,
                lw=lw, solid_capstyle='round', **kwargs)

    def label(x, y, value, size=12, ha='center', va='center', color=INK, **kwargs):
        return ax.text(x, y, value, fontsize=size, ha=ha, va=va, color=color, **kwargs)

    def dot(x, y):
        ax.add_patch(Circle((x, y), .045, color=WIRE, zorder=6))

    def ground(x, y):
        line((x, y), (x, y-.16))
        for i, halfwidth in enumerate([.24, .16, .08]):
            line((x-halfwidth, y-.16-.10*i), (x+halfwidth, y-.16-.10*i))

    def resistor_h(x0, x1, y, ref):
        cx = (x0+x1)/2
        line((x0, y), (cx-.40, y))
        ax.add_patch(Rectangle((cx-.40, y-.14), .80, .28,
                               facecolor='white', edgecolor=INK, lw=1.3, zorder=5))
        line((cx+.40, y), (x1, y))
        label(cx, y+.45, ref+'  330 Ω', size=11)

    def resistor_v(x, top, bottom, ref, value, side='right'):
        cy = (top+bottom)/2
        line((x, top), (x, cy+.35))
        ax.add_patch(Rectangle((x-.14, cy-.35), .28, .70,
                               facecolor='white', edgecolor=INK, lw=1.3, zorder=5))
        line((x, cy-.35), (x, bottom))
        shift, align = (.36, 'left') if side=='right' else (-.36, 'right')
        label(x+shift, cy, ref+'\n'+value, size=11, ha=align)

    def power(x, y):
        line((x, y), (x, y+.47))
        ax.add_patch(Polygon([(x-.11, y+.45), (x+.11, y+.45), (x, y+.64)],
                             closed=True, facecolor='white', edgecolor=INK, lw=1.2))
        label(x, y+.84, '3V3', size=11)

    def switch(x, y, ref):
        # Normally open momentary switch; left is power, right is the input.
        line((x, y), (x+.13, y))
        line((x+.82, y), (x+.95, y))
        line((x+.82, y), (x+.29, y+.29), color=INK)
        for xx in [x+.13, x+.82]:
            ax.add_patch(Circle((xx, y), .045, facecolor='white', edgecolor=INK, lw=1.1, zorder=7))
        line((x+.48, y+.34), (x+.48, y+.57), color=INK, lw=1)
        line((x+.34, y+.57), (x+.62, y+.57), color=INK, lw=1)
        label(x+.48, y-.36, ref, size=11)

    # Board: symbol pins are arranged for wiring clarity, not as a board footprint.
    ax.add_patch(Rectangle((6.40, 2.05), 3.55, 7.15, facecolor='#f2f6fa', edgecolor=INK, lw=1.6))
    label(8.175, 8.63, 'U1', size=11, color=MUTED)
    label(8.175, 8.04, 'Arduino', size=18, weight='bold')
    label(8.175, 7.54, 'UNO R4 WiFi', size=15, weight='bold')
    leftpins = [(8.95, '3V3'), (6.60, 'A0'), (4.85, 'D2'), (3.65, 'D3'), (2.50, 'GND')]
    for y, name in leftpins:
        line((6.0, y), (6.4, y))
        label(6.58, y, name, size=12, ha='left')
    for y, name in [(8.25, 'D8'), (6.45, 'D6'), (4.65, 'D7')]:
        line((9.95, y), (10.35, y))
        label(9.77, y, name, size=12, ha='right')
    line((6.0, 2.5), (5.76, 2.5), (5.76, 1.98))
    ground(5.76, 1.98)

    # Phototransistor emitter follower from supplied reference.
    line((6.0, 8.95), (2.5, 8.95), (2.5, 8.42))
    ax.add_patch(Circle((2.18, 7.85), .58, facecolor='white', edgecolor=INK, lw=1.2))
    line((1.96, 7.53), (1.96, 8.17), color=INK, lw=1.8)
    line((1.96, 8.04), (2.50, 8.42), color=INK)
    line((1.96, 7.65), (2.50, 7.25), color=INK)
    ax.add_patch(FancyArrowPatch((2.12, 7.53), (2.49, 7.26), arrowstyle='-|>',
                                color=INK, mutation_scale=11, lw=1.1))
    line((2.5, 7.25), (2.5, 6.60), (6.0, 6.60))
    for a, b in [((.82, 8.52), (1.48, 8.11)), ((1.03, 8.82), (1.69, 8.41))]:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', color=MUTED, mutation_scale=11, lw=1.15))
    label(3.18, 7.80, 'Q1', size=11, ha='left')
    dot(2.5, 6.60)
    line((2.5, 6.60), (2.5, 6.25))
    ax.add_patch(Rectangle((2.36, 5.56), .28, .69, facecolor='white', edgecolor=INK, lw=1.3))
    label(2.91, 5.90, 'R4\n1 kΩ', size=11, ha='left')
    line((2.5, 5.56), (2.5, 5.34))
    ground(2.5, 5.34)

    # User-confirmed external pull-downs: each input -> 10k -> GND;
    # each normally open button connects its input to 3.3V when pressed.
    power(.48, 4.85)
    line((.48, 4.85), (.70, 4.85))
    switch(.70, 4.85, 'SW1')
    line((1.65, 4.85), (6.0, 4.85))
    dot(1.90, 4.85)
    resistor_v(1.90, 4.85, 2.70, 'R5', '10 kΩ')
    ground(1.90, 2.70)

    power(3.10, 3.65)
    line((3.10, 3.65), (3.35, 3.65))
    switch(3.35, 3.65, 'SW2')
    line((4.30, 3.65), (6.0, 3.65))
    dot(4.80, 3.65)
    resistor_v(4.80, 3.65, 1.30, 'R6', '10 kΩ', side='left')
    ground(4.80, 1.30)

    # Three LED junctions within a common-cathode RGB LED package.
    for y, ref, colorname, color in [(8.25, 'R1', 'R', '#a63639'),
                                     (6.45, 'R2', 'G', '#357650'),
                                     (4.65, 'R3', 'B', '#3569a5')]:
        resistor_h(10.35, 12.20, y, ref)
        line((12.2, y), (12.78, y))
        ax.add_patch(Polygon([(12.78, y-.22), (12.78, y+.22), (13.20, y)],
                             closed=True, facecolor='white', edgecolor=INK, lw=1.3))
        line((13.20, y-.25), (13.20, y+.25), color=INK)
        line((13.20, y), (15.45, y))
        for offset in [0, .25]:
            ax.add_patch(FancyArrowPatch((12.82+offset, y+.31), (13.13+offset, y+.68),
                                        arrowstyle='-|>', mutation_scale=9, color=color, lw=1))
        label(14.12, y+.45, colorname, size=12, color=color)
        dot(15.45, y)
    line((15.45, 8.25), (15.45, 3.47))
    ground(15.45, 3.47)
    ax.add_patch(Rectangle((12.50, 4.12), 3.30, 4.98, facecolor='none',
                           edgecolor='#8593a0', lw=1, ls=(0, (4, 3))))
    fig.subplots_adjust(left=.01, right=.99, top=.99, bottom=.01)
    for extension in ['pdf', 'svg', 'png']:
        kwargs = {'dpi': 240} if extension == 'png' else {}
        if extension == 'pdf':
            kwargs['metadata'] = {'CreationDate': None, 'ModDate': None}
        fig.savefig(folder/f'circuit_two_buttons.{extension}', bbox_inches='tight', pad_inches=.08, **kwargs)
    svg = folder/'circuit_two_buttons.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)

    record = {'role': 'REDRAWN_REFERENCE_CIRCUIT_WITH_TWO_MOMENTARY_BUTTONS',
              'reference': 'User-supplied circuit image, rearranged with attribution',
              'resistors_confirmed_by_user': '2026-11-27; LED:330 ohm each; sensor load:1 kiloohm; each button:10 kiloohm',
              'button_wiring_confirmed_by_user': '2026-11-27; resistors to GND; buttons to3.3V',
              'sensor_type_symbol': 'Phototransistor as in supplied reference; device model unconfirmed',
              'supply': 'Sensor 3.3V as in supplied reference; common ground; USB board power',
              'pin_definitions_checked': expected,
              'button_mode': 'EXTERNAL_10K_PULLDOWN; BUTTON_TO_3V3; PRESSED_HIGH',
              'frozen_firmware_button_mode': 'INPUT_PULLUP; pressed LOW; disabled during actual10 captures',
              'visible_labels': 'BOARD_MODEL_COMPONENT_REFERENCES_RESISTOR_VALUES_PIN_NUMBERS_RGB_AND_POWER_NETS_ONLY',
              'physical10_capture_control': 'SERIAL; PHYSICAL_BUTTONS_DISABLED_IN_FROZEN_FIRMWARE',
              'frozen_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'netlist': {'sensor_collector': ['3V3'], 'sensor_emitter': ['A0', 'R4.1'],
                          'sensor_ground': ['R4.2', 'GND'],
                          'calibration': ['D2', 'SW1.2', 'R5.1'],
                          'read': ['D3', 'SW2.2', 'R6.1'],
                          'button_power': ['SW1.1', 'SW2.1', '3V3'],
                          'button_ground': ['R5.2', 'R6.2', 'GND'],
                          'red': ['D8', 'R1', 'LED_R.anode'], 'green': ['D6', 'R2', 'LED_G.anode'],
                          'blue': ['D7', 'R3', 'LED_B.anode'],
                          'led_ground': ['LED_R.cathode', 'LED_G.cathode', 'LED_B.cathode', 'GND']},
              'resistors_ohms': {'R1':330, 'R2':330, 'R3':330, 'R4':1000, 'R5':10000, 'R6':10000}}
    (folder/'circuit_two_buttons.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    build_circuit_figure(args.output, args.source)
