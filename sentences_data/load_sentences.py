from scipy.signal import decimate
from pathlib import Path
import bisect
import numpy as np
import os 

import pyxdf as xdf


def locate_pos(available_freqs, target_freq):
    pos = bisect.bisect_right(available_freqs, target_freq)
    if pos == 0:
        return 0
    if pos == len(available_freqs):
        return len(available_freqs)-1
    if abs(available_freqs[pos]-target_freq) < abs(available_freqs[pos-1]-target_freq):
        return pos
    else:
        return pos-1  

if __name__=="__main__":
    os.chdir("sentences_data")
    path = Path('./sentences/')
    outPath = Path('./eeg/')
    
    for f in path.rglob('*sentences.xdf'):
        streams = xdf.load_xdf(str(f),dejitter_timestamps=False)
        streamToPosMapping = {}
        for pos in range(0,len(streams[0])):
            stream = streams[0][pos]['info']['name']
            streamToPosMapping[stream[0]] = pos

        # Get sEEG
        eeg = streams[0][streamToPosMapping['Micromed']]['time_series']
        offset = float(streams[0][streamToPosMapping['Micromed']]['info']['created_at'][0])
        eeg_ts = streams[0][streamToPosMapping['Micromed']]['time_stamps'].astype('float')#+offset
        eeg_sr = int(float(streams[0][streamToPosMapping['Micromed']]['info']['nominal_srate'][0]))

        if eeg_sr == 2048:
            eeg = decimate(eeg,2,axis=0)
            eeg_ts = eeg_ts[::2]
            eeg_sr = 1024

        #Get channel info
        chNames = []
        for ch in streams[0][streamToPosMapping['Micromed']]['info']['desc'][0]['channels'][0]['channel']:
            chNames.append(ch['label'])

        #Load Audio
        audio = streams[0][streamToPosMapping['AudioCaptureWin']]['time_series']
        offset_audio = float(streams[0][streamToPosMapping['AudioCaptureWin']]['info']['created_at'][0])
        audio_ts = streams[0][streamToPosMapping['AudioCaptureWin']]['time_stamps'].astype('float')#+offset
        audio_sr = int(streams[0][streamToPosMapping['AudioCaptureWin']]['info']['nominal_srate'][0]) 
        
        # Load Marker stream
        markers = streams[0][streamToPosMapping['SentencesMarkerStream']]['time_series']
        offset_marker = float(streams[0][streamToPosMapping['SentencesMarkerStream']]['info']['created_at'][0])
        marker_ts = streams[0][streamToPosMapping['SentencesMarkerStream']]['time_stamps'].astype('float')#-offset

        #Get Experiment time
        i=0
        while markers[i][0]!='experimentStarted':
            i+=1

        eeg_start= locate_pos(eeg_ts, marker_ts[i])
        audio_start = locate_pos(audio_ts, eeg_ts[eeg_start])

        while markers[i][0]!='experimentEnded' and i<len(markers)-1:
            i+=1

        eeg_end= locate_pos(eeg_ts, marker_ts[i])
        audio_end = locate_pos(audio_ts, eeg_ts[eeg_end])
        markers=markers[:i]
        marker_ts=marker_ts[:i]

        eeg = eeg[eeg_start:eeg_end,:]
        eeg_ts = eeg_ts[eeg_start:eeg_end]
        audio = audio[audio_start:audio_end,:]
        audio_ts=audio_ts[audio_start:audio_end]

        # Get sentences
        sentences=['' for a in range(eeg.shape[0])]
        sentencesMask = [m[0].split(';')[0]=='start' for m in markers]
        sentencesStarts = marker_ts[sentencesMask]
        sentencesStarts = np.array([locate_pos(eeg_ts, x) for x in sentencesStarts])
        dispSentences =  [m[0].split(';')[1] for m in markers if m[0].split(';')[0]=='start']
        sentencesEndMask = [m[0].split(';')[0]=='end' for m in markers]
        endSentences = [m[0].split(';')[1] for m in markers if m[0].split(';')[0]=='end']
        sentencesEnds = marker_ts[sentencesEndMask]
        sentencesEnds = np.array([locate_pos(eeg_ts, x) for x in sentencesEnds])

        if len(sentencesStarts)!=len(sentencesEnds):
            print('Problem in labels of %s'  % (str(f.name)))
            foundStarts = [dispSentences[i] in endSentences for i in range(len(dispSentences))]
            remove = np.argwhere(np.array(foundStarts)==False)
            dispSentences = np.delete(dispSentences,remove)
            sentencesStarts = np.delete(sentencesStarts,remove)
            foundEnds = [endSentences[i] in dispSentences  for i in range(len(endSentences))]
            remove = np.argwhere(np.array(foundEnds)==False)
            sentencesEnds = np.delete(sentencesEnds,remove)
            endSentences = np.delete(endSentences,remove)

        for i, start in enumerate(sentencesStarts):
            sentences[start:sentencesEnds[i]]=[dispSentences[i] for rep in range(sentencesEnds[i]-start)]
        
        print('All aligned for %s' % (str(f)))
        
        # Saving
        prefix = str(f.name)[:8]
        np.save(outPath/(prefix + '_sentences_sEEG.npy'), eeg)
        np.save(outPath/(prefix +'_sentences.npy'), np.array(sentences))
        np.save(outPath/(prefix +'_sentences_channelNames.npy'), np.array(chNames))
        np.save(outPath/(prefix +'_sentences_audio.npy'),audio[:,0])
        
        

    print("Done")