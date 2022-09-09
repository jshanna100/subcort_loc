import numpy as np
import mne
import re

zeb2eeg = {"GND":"AFz", "REF":"FCz", "lc1":"LCHK1", "lc2":"LCHK2",
           "lc3":"LCHK3", "rc1":"RCHK1", "rc2":"RCHK2", "rc3":"RCHK3",
           "g1":"Fp1", "g2":"Fp2", "g3":"F7", "g4":"F3", "g5":"Fz", "g6":"F4",
           "g7":"F8", "g8":"FC5", "g9":"FC1", "g10":"FC2", "g11":"FC6",
           "g12":"T7", "g13":"C3", "g14":"Cz", "g15":"C4", "g16":"T8",
           "g17":"TP9", "g18":"CP5", "g19":"CP1", "g20":"CP2", "g21":"CP6",
           "g22":"TP10", "g23":"P7", "g24":"P3", "g25":"Pz", "g26":"P4",
           "g27":"P8", "g28":"PO9", "g29":"O1", "g31":"O2", "g32":"PO10",
           "r1":"Fpz", "r2":"F9", "r3":"AFF5h", "r4":"AFF1h", "r5":"AFF2h",
           "r6":"AFF6h", "r7":"F10", "r8":"FTT9h", "r9":"FTT7h", "r10":"FCC5h",
           "r11":"FCC3h", "r12":"FCC1h", "r13":"FCC2h", "r14":"FCC4h",
           "r15":"FCC6h", "r16":"FTT8h", "r17":"FTT10h", "r19":"TPP7h",
           "r21":"CPP3h", "r22":"CPP1h", "r23":"CPP2h", "r24":"CPP4h",
           "r26":"TPP8h", "r29":"POO1", "r30":"POO2", "r32":"Iz", "w1":"AFp1",
           "w2":"AFp2", "w3":"FFT9h", "w4":"FFT7h", "w5":"FFC5h", "w6":"FFC3h",
           "w7":"FFC1h", "w8":"FFC2h", "w9":"FFC4h", "w10":"FFC6h",
           "w11":"FFT8h", "w12":"FFT10h", "w13":"TTP7h", "w14":"CCP5h",
           "w15":"CCP3h", "w16":"CCP1h", "w17":"CCP2h", "w18":"CCP4h",
           "w19":"CCP6h", "w20":"TTP8h", "w21":"P9", "w22":"PPO9h",
           "w23":"PPO5h", "w24":"PPO1h", "w25":"PPO2h", "w26":"PPO6h",
           "w27":"PPO10h", "w28":"P10", "w29":"I1", "w30":"OI1h", "w31":"OI2h",
           "w32":"I2", "y1":"AF7", "y2":"AF3", "y3":"AF4", "y4":"AF8",
           "y5":"F5", "y6":"F1", "y7":"F2", "y8":"F6", "y9":"FT9", "y10":"FT7",
           "y11":"FC3", "y12":"FC4", "y13":"FT8", "y14":"FT10", "y15":"C5",
           "y16":"C1", "y17":"C2", "y18":"C6", "y19":"TP7", "y20":"CP3",
           "y21":"CPz","y22":"CP4",  "y23":"TP8", "y24":"P5", "y25":"P1",
           "y26":"P2", "y27":"P6", "y28":"PO7", "y29":"PO3", "y30":"POz",
           "y31":"PO4", "y32":"PO8"}

zeb_block_key = {"91":{"AB":1, "CD":2}, "92":{"BA":1, "CD":2},
                 "93":{"AB":1, "DC":2}}

# get the 64-128 channels so we can remove them
last64 = []
for k, v in zeb2eeg.items():
    if re.match("w[0-9]", k) or re.match("r[0-9]", k):
        last64.append(v)

zeb_dir = "/home/jev/deepeeg/zebris/"
proc_dir = "/home/jev/deepeeg/proc/"
out_dir = "/home/jev/subcort_loc/proc/"
subjs = ["91", "92", "93"]
for subj in subjs:
    for zeb_idx, blocks in zip([1, 2], [["A","B"], ["C", "D"]]):
        elec_dict = dict()
        hsp = []
        zebfile = "{}Geschichte_{}_{}_Geschichte.sfp".format(zeb_dir, subj,
                                                             zeb_idx)
        # get and process zebris data
        coords = mne.channels.read_custom_montage(zebfile)
        for ch, ch_pos in coords.get_positions()["ch_pos"].items():
           if ch=="fidt9":
               lpa = ch_pos
           elif ch=="fidt10":
               rpa = ch_pos
           elif ch=="fidnz":
               nas = ch_pos
           elif ch[:3]=="sfl":
               hsp.append(ch_pos)
           elif ch=="REF":
               continue
           else:
               elec_dict[zeb2eeg[ch]]=ch_pos
        hsp = np.array(hsp)

        digmon = mne.channels.make_dig_montage(nasion=nas, lpa=lpa, rpa=rpa,
                                               ch_pos=elec_dict, hsp=hsp)

        for block in blocks:
            raw = mne.io.Raw("{}{}_{}_hand_ica-raw.fif".format(proc_dir, subj,
                                                               block))
            raw.drop_channels(last64)
            if subj == "92" and (block == "C" or block == "D"):
                raw.drop_channels(["PO10"])
            raw.set_montage(digmon)
            raw.save("{}{}_{}_64-raw.fif".format(out_dir, subj, block),
                     overwrite=True)
