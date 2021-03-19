#######################################################
#     Spiros Daskalakis                               #
#     last Revision: 16/03/2021                       #
#     Python Version:  3.9                            #
#     Email: daskalakispiros@gmail.com                #
#     Website: www.daskalakispiros.com                #
#######################################################

import numpy as np
import simpleaudio as sa
import yaml
from box import Box
from matplotlib import pyplot as plt

class RXtag:

    def __init__(self, message_text="", times=1, mode=0):
        self.message_text = message_text
        self.times = times

        with open("config.yml", "r") as ymlfile:
            self.cfg = Box(yaml.safe_load(ymlfile), default_box=True, default_box_attr=None)
        self.print_rx_info()
        preamble_mask = np.array([-1, -1, -1, -1, -1, 1, 1, 1])
        # raw symbols
        symbols = self.text_to_numbers()
        print('Initial Symbols:', symbols)
            # %%%%%%%%%%%%%%%TEST%%%%%%%%%%%%%%%%%%%%%%%
            #signal = self.lora_symbol(3, inverse=0)
            #self.send_packet_to_sound_card(signal)
        if mode == 1:
            # inverse Gray encoding
            inv_gray_symbols = np.zeros(len(symbols), dtype=int)
            for i in range(len(symbols)):
                inv_gray_symbols[i] = self.inverse_gray_encoder(symbols[i])
            symbols=inv_gray_symbols
            print('Gray Symbols:', symbols)

        packet = self.create_packet(preamble_mask, symbols)

        fig = plt.figure(1)
        NFFT = 1024
        # plt.imshow(spectrogram, aspect='auto', cmap='hot_r', origin='lower')
        plt.specgram(packet[:,1], NFFT=NFFT, Fs=self.cfg.TX_SAMPLING_RATE, noverlap=900)
        plt.title('Spectrogram')
        plt.ylabel('Frequency band')
        plt.xlabel('Time window')
        plt.grid(True)
        plt.draw()
        plt.show(block=True)
        #plt.savefig("RX_Packet.pdf")




        #self.send_packet_to_sound_card(packet)

    def print_rx_info(self):
        print("%%%%%%%%%%%% RX Parameters  %%%%%%%%%%%%%%%%")
        print("Sample Rate:", self.cfg.TX_SAMPLING_RATE, "Sps")
        print("Spreading factor:", self.cfg.SF)
        print("Coding rate:", self.cfg.CR)
        print("BandWith:", self.cfg.BW, "Hz")
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    def inverse_gray_encoder(self, n):
        inv = 0
        # Taking xor until
        # n becomes zero
        while n:
            inv = inv ^ n
            n = n >> 1
        return inv

    def gray_encoder(self, n):

        """
        Convert given decimal number into decimal equivalent of its gray code form
        :param n: decimal number
        return: int
        """
        # Right Shift the number by 1 taking xor with original number
        return n ^ (n >> 1)

    def text_to_numbers(self):
        """
        Convert a string message to an integer number according the utf-8 encoding
        return: int numpy array
        """
        text = self.message_text
        # print("The original message is : " + str(text))
        arr = bytes(text, 'utf-8')
        numbers = np.zeros(len(arr), dtype=int)
        i = 0
        for byte in arr:
            numbers[i] = byte
            i = i + 1
        return numbers

    def text_to_binary(self):
        text = self.message_text
        print("The original message is : " + str(text))
        sf = self.cfg.SF
        # using join() + ord() + format()
        # Converting String to binary
        res = ''.join(format(ord(i), '08b') for i in text)
        # printing result

        print("The message after binary conversion : " + str(res))
        # print("The message after dec conversion : " + arr2)
        return res

    def lora_symbol(self, symbol, inverse=0):
        # Initialization
        phase = 0
        Frequency_Offset = self.cfg.CHIRP_F_START
        shift = symbol
        sf = self.cfg.SF
        fs = self.cfg.TX_SAMPLING_RATE
        bw = self.cfg.BW
        num_samples_in = fs * (2 ** sf) / bw
        num_samples = round(num_samples_in)

        signal = np.zeros((num_samples, 2))

        for k in range(num_samples):
            # set output to cosine signal
            signal[k, 0] = np.cos(phase)
            signal[k, 1] = np.sin(phase)

            # ------------------------------------------
            # Frequency from cyclic shift
            f = bw * shift / (2 ** sf)
            if inverse == 1:
                f = bw - f
            # ------------------------------------------
            # apply Frequency offset away from DC
            f = f + Frequency_Offset
            # ------------------------------------------
            # Increase the phase according to frequency
            phase = phase + 2 * np.pi * f / fs
            if phase > np.pi:
                phase = phase - 2 * np.pi
            # ------------------------------------------
            # update cyclic shift
            shift = shift + bw / fs
            if shift >= (2 ** sf):
                shift = shift - 2 ** sf
        return signal

    def create_packet(self, mask, payload_symbols):

        sf = self.cfg.SF
        fs = self.cfg.TX_SAMPLING_RATE
        bw = self.cfg.BW
        left = self.cfg.CHIRP_F_START
        preamble_up_chirp = self.lora_symbol(0, 0)
        preamble_down_chirp = self.lora_symbol(0, 1)

        preamble = np.zeros((1, 2))
        # preamble = np.empty(1, 2)
        for k in range(mask.size):
            if mask[k] == 1:
                preamble = np.concatenate((preamble, preamble_up_chirp), axis=0)
            elif mask[k] == -1:
                preamble = np.concatenate((preamble, preamble_down_chirp), axis=0)
            else:
                preamble_symbol_chirp = self.lora_symbol(mask[k], 1)
                preamble = np.concatenate((preamble, preamble_symbol_chirp), axis=0)
        preamble = preamble[1:, :]
        print("Preamble created, Matrix Dims:", preamble.shape)

        # payload = np.array([])
        payload = np.zeros((1, 2))
        for k in range(payload_symbols.size):
            payload_signal = self.lora_symbol(payload_symbols[k], 0)
            payload = np.concatenate((payload, payload_signal), axis=0)
        payload = payload[1:, :]
        print("Payload created Matrix Dims:", payload.shape)

        packet = np.concatenate((preamble, payload), axis=0)
        print("Packet created Matrix Dims:", packet.shape)

        return packet

    def send_packet_to_sound_card(self, packet):
        # This function uses the sound card in order to produce the chirp signals
        # packet is a array of floats
        sound_fs = self.cfg.TX_SAMPLING_RATE
        sound_res = self.cfg.SOUND_RES

        if sound_res == 16:
            # Ensure that highest value is in 16-bit range
            audio = packet * (2 ** 15 - 1) / np.max(np.abs(packet))
            # Convert to 16-bit data
            audio = audio.astype(np.int16)
            ch = 2
            play_obj = sa.play_buffer(audio, ch, 2, sound_fs)
        elif sound_res == 24:
            # normalize to 24-bit range
            packet *= (2 ** 23 - 1) / np.max(np.abs(packet))
            # convert to 32-bit data
            audio = packet.astype(np.int32)
            # convert from 32-bit to 24-bit by building a new byte buffer, skipping every fourth bit
            # note: this also works for 2-channel audio
            i = 0
            byte_array = []
            for b in audio.tobytes():
                if i % 4 != 3:
                    byte_array.append(b)
                i += 1
            audio = bytearray(byte_array)
            ch = 2
            play_obj = sa.play_buffer(audio, ch, 3, sound_fs)
        # Start playback
        # Wait for playback to finish before exiting
        play_obj.wait_done()
        print("Packet sent to Sound Card")
