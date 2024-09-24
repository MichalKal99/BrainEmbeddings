def compensate_eeg_audio_with_markers(
    eeg, eeg_sr, audio, audio_sr, markers_time_stamps
):
    # the sEEg and audio data might slightly differ in length with the markers. We should compensate by reducing to the smallest:
    print(
        "The ORIGINAL length audio {}, sEEG {}, markers {}".format(
            len(eeg) / eeg_sr, len(audio) / audio_sr, markers_time_stamps[-1]
        )
    )
    minimum = min(len(eeg) / eeg_sr, len(audio) / audio_sr, markers_time_stamps[-1])
    eeg = eeg[: int(minimum * eeg_sr)]
    audio = audio[: int(minimum * audio_sr)]
    print(
        "The REDUCED length audio {}, sEEG {}, markers {}".format(
            len(eeg) / eeg_sr, len(audio) / audio_sr, markers_time_stamps[-1]
        )
    )
    return eeg, audio


def clean_markers(markers_time_series, markers_time_stamps):
    # There is a word written in a weird format and it brakes the code, so I'll change it manually
    for i, marker in enumerate(markers_time_series):
        if marker == ["start;`s morgens\r"]:
            markers_time_series[i] = ["start;smorgens\r"]
        elif marker == ["end;`s morgens\r"]:
            markers_time_series[i] = ["end;smorgens\r"]

    # remove from the time_series some useless characters that difficult the parsing
    markers_time_series_stripped = [
        value[0][value[0].find(";") + 1 : value[0].find("\\")]
        for value in markers_time_series
    ]

    # Start the time_stamps in zero
    markers_time_stamps_norm = [
        marker - markers_time_stamps[0] for marker in markers_time_stamps
    ]

    # remove the first and last value because it is always 'experimentStarted' and 'experimentEnded'
    markers_time_stamps_norm_small = markers_time_stamps_norm[1:-1]
    markers_time_series_stripped_small = markers_time_series_stripped[1:-1]

    return markers_time_series_stripped_small, markers_time_stamps_norm_small


def redefine_cutting_points(markers_time_stamps):
    # redefine cutting points
    markers_time_stamps_mod = []
    for i, marker in enumerate(markers_time_stamps):
        if i == 0:
            # handle the first value
            markers_time_stamps_mod.append(0)
        elif i + 1 == len(markers_time_stamps):
            # handle the last value
            markers_time_stamps_mod.append(marker)
        elif i % 2 == 0:
            # handle all the middle values
            value_to_append = (markers_time_stamps[i - 1] + marker) / 2
            markers_time_stamps_mod.append(value_to_append)
            markers_time_stamps_mod.append(value_to_append)
    return markers_time_stamps_mod
