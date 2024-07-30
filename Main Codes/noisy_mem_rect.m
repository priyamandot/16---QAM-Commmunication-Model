clc;
clearvars;
close all;


[x,Fs] = loadAudio("project.wav");

% 1. A/D CONVERTER

% x = [0.1;0.2;0.3];
% % Fs = 48000;
audio_normalized = int16(x * 32767); % Normalize
audio_binary = dec2bin(typecast(audio_normalized(:), 'uint16'), 16); % Convert to binary

% audio_normalized = int8(x * 127); % Normalize
% audio_binary = dec2bin(typecast(audio_normalized(:), 'uint8'), 8); % Convert to binary

binary_vector = audio_binary(:)';

binary_vector= str2num(binary_vector(:))'; % don't change to double !! 


% binary_vector = [0, 0, 0, 1 ,1 , 1 , 0 ,0,0, 0, 0, 1 ,1 , 1 , 0 ,0,1,1,1,1,0,1,0,1,1,1,1,1,0,0,0,1];


% 2 . ENCODER
qam_symbols = sixteenqammap_int(binary_vector);
% qam_symbols = sixteenqammap_int(binary_vector);

% 
% figure ;
% plot(real(qam_symbols),imag(qam_symbols),"o");

% 3 . ENCODER LINE - SRRC 
m = 10;
upsampled_16qam = upsample(qam_symbols, m); % adds zeroes in between
% 
% % Plot the original signal (not upsampled)
% figure;
% subplot(2,1,1);
% stem(1:length(qam_symbols), real(qam_symbols), 'r', 'LineWidth', 1.5);
% hold on;
% stem(1:length(qam_symbols), imag(qam_symbols), 'b', 'LineWidth', 1.5);
% title('Original Signal (Not Upsampled)');
% xlabel('Sample Index');
% ylabel('Amplitude');
% legend('Real Part', 'Imaginary Part');
% grid on;
% 
% % Plot the original signal (upsampled)
% subplot(2,1,2);
% stem(1:length(upsampled_16qam), real(upsampled_16qam), 'r', 'LineWidth', 1.5);
% hold on;
% stem(1:length(upsampled_16qam), imag(upsampled_16qam), 'b', 'LineWidth', 1.5);
% title('Original Signal (Upsampled)');
% xlabel('Sample Index');
% ylabel('Amplitude');
% legend('Real Part', 'Imaginary Part');
% grid on;


oversampling_factor = 10; % we need to sample at only one instant 
m = oversampling_factor;
% a = 1; % roll - off 
% length_fil = 5;% (truncated outside [-length*T,length*T])
% %raised cosine transmit filter (time vector set to a dummy variable which is not used)

[~,len] = size(qam_symbols);
% [time_idx, transmit_filter] = rrc_pulse(len,10e6,a,1/1e6);

length_fil = 16;
transmit_filter = get_rectangular(length_fil); % 1/2 1/2 -> 0.0087 

% [transmit_filter,~] = sqrt_raised_cosine(a,m,length_fil);
% transmit_filter = get_rectangular(12);

%NOISELESS MODULATED SIGNAL
tx_output = conv(upsampled_16qam,transmit_filter,"same");
% 
% 
% % Plot the transmit filter
% figure;
% stem(1:length(transmit_filter), transmit_filter, 'b', 'LineWidth', 2);
% xlabel('Time (symbol intervals)');
% ylabel('Amplitude');
% title('Raised Cosine Pulse');
% grid on;
% 
% % Plot the output of the convolution
% figure;
% stem(1:length(tx_output), real(tx_output), 'r', 'LineWidth', 1.5);
% hold on;
% stem(1:length(tx_output), imag(tx_output), 'b', 'LineWidth', 1.5);
% title('Output of Convolution with Transmit Filter');
% xlabel('Sample Index');
% ylabel('Amplitude');
% legend('Real Part', 'Imaginary Part');
% grid on;
% 
% figure ;
% plot(real(tx_output),imag(tx_output),"o");
% title("Constellation after Line encoding");

% 4. MODULATION
m1 = real(tx_output);
m2 = imag(tx_output);

% parameters fc
[~,p] = size(tx_output);
f = 2e6;

fs = 6e6;

t = (0:p-1)/fs;

c1 = cos(2*pi*f*t);
c2 = sin(2*pi*f*t);

% plot(c1)
qam_mod = 2*m1.*c1 + 2*m2.*c2;

% figure ;
% plot(real(qam_mod),imag(qam_mod),"o");
% title("Constellation after Modulation");
% 
% 
% figure;
% stem(qam_mod);
% title("QAM modulation ");


% 5 . CHANNEL NOISE  WITH MEMORY :

qam16_constellation = [-3-3i, -3-1i, -3+3i, -3+1i, ...
                        -1-3i, -1-1i, -1+3i, -1+1i, ...
                         3-3i,  3-1i,  3+3i,  3+1i, ...
                         1-3i,  1-1i,  1+3i,  1+1i
                         ];

% Calculate the sum of squares of the constellation points
sum_of_squares = sum(abs(qam16_constellation).^2);

% Calculate average received energy per symbol
E_B_n_2 = sum_of_squares/16; % 16qam case 


EbN0_dB = 0:0.01:10;

% Convert Eb/N0 from dB to linear scale
EbN0_linear = 10.^(EbN0_dB / 10);

% Define the bit error probability
Pb = qfunc(sqrt(4/5 * EbN0_linear));


% Remove duplicate values from Pb and corresponding Eb/N0 values
[Pb_unique, idx] = unique(Pb);
EbN0_dB_unique = EbN0_dB(idx);

% Find the value of Eb/N0 that corresponds to an error probability of 10^-2 and 10^-5
EbN0_10_2 = interp1(Pb_unique, EbN0_dB_unique, 10^-3);

disp(['Eb/N0 (dB) corresponding to Pb = 10^-2: ', num2str(EbN0_10_2)]);


Es = E_B_n_2*norm(transmit_filter)^2; % 10 is because of mean square values 
Eb = Es/4;
N0 = Eb/(10^(EbN0_10_2 /10));

% sigma_squared = N0/2;
% sigma_squared = 0.2;

b_1 = 1;
Tb = 1/(1e6);
% delta = zeros(3,1);
a_1 = 0.9;
% delta(1) = a;
% delta(b*Tb) = 1-a;
t_1  = (0:20)/fs;

delta = gen_mem_noise(a_1,b_1,Tb,t_1);


% qam_mod_with_memory = conv(qam_mod, delta,"same");
qam_mod_with_memory = filter(delta,1,qam_mod);

% qam_mod_with_memory = qam_mod;
% % Plot QAM symbols before and after convolution
% figure;
% subplot(2,1,1);
% plot(real(qam_mod), imag(qam_mod), 'o', 'MarkerSize', 8);
% title('QAM Symbols Before Convolution');
% xlabel('In-phase');
% ylabel('Quadrature');
% axis square;
% grid on;
% 
% subplot(2,1,2);
% plot(real(qam_mod_with_memory), imag(qam_mod_with_memory), 'o', 'MarkerSize', 8);
% title('QAM Symbols After Convolution with Channel Noise');
% xlabel('In-phase');
% ylabel('Quadrature');
% axis square;
% grid on;

% noise_samples = sqrt(sigma_squared) * (randn(size(qam_mod_with_memory)));

% sig_plus_noise = noise_samples+qam_mod_with_memory;

sig_plus_noise = awgn(qam_mod,25,'measured');

% figure;
% subplot(2,1,1);
% stem(sig_plus_noise);
sig_plus_noise = filter(1,delta,sig_plus_noise);
% subplot(2,1,2);
% stem(sig_plus_noise);



% figure ;
% plot(real(sig_plus_noise),imag(sig_plus_noise),"o");
% title("Constellation after Channel");


% 6 . DEMODULATION 

dem1 = sig_plus_noise.*c1;
dem2 = sig_plus_noise.*c2;


% figure;
% stem(dem1);
% title("Demodulation - m1");
% 
% figure;
% stem(dem2);
% title("Demodulation - m2");

% adding filter to demodulate it 

% Design the Butterworth filter
N = 5; % Filter order
Fc = 1e6; % Cutoff frequency (adjust as needed)
Wn = Fc / (fs / 2); % Normalize cutoff frequency
[b, a] = butter(N, Wn, 'low'); % Design the filter


% Apply the filter to demodulated signals
baseband_signal_1 = filter(b, a, dem1);
baseband_signal_2 = filter(b, a, dem2);


% LPF Block 
% wc_lowpass = 1; % cant be 0 
% bw_lowpass = 1000; 
% order_lowpass = 25;
% 
% filter_coeffs_lowpass = fir1(order_lowpass, wc_lowpass / (fs/2), 'low');
% 
% baseband_signal_1 = filter(filter_coeffs_lowpass, 1, dem1);
% baseband_signal_2 = filter(filter_coeffs_lowpass, 1, dem2);

% % Design the Butterworth filter
% N = 5; % Filter order
% Fc = 1e6; % Cutoff frequency (adjust as needed)
% Wn = f / (fs / 2); % Normalize cutoff frequency
% [b, a] = butter(N, Wn, 'low'); % Design the filter
% 
% 
% % Apply the filter to demodulated signals
% baseband_signal_1 = filter(b, a, dem1);
% baseband_signal_2 = filter(b, a, dem2);
% 

% Define frequency range for visualization
f_range = linspace(0, fs/2, 1000); % Frequency range from 0 to Nyquist frequency

% Compute frequency response of the filter
H = freqz(b, a, f_range, fs);
% 
% % Plot frequency response
% figure;
% plot(f_range, 20*log10(abs(H))); % Plot in dB scale
% title('Frequency Response of Butterworth Low-pass Filter');
% xlabel('Frequency (Hz)');
% ylabel('Magnitude (dB)');
% grid on;
% 
% 
% figure;
% N = length(dem1); % Length of the signal
% frequencies = linspace(-fs/2, fs/2, N); % Frequency axis
% spectrum = fftshift(fft(dem1)); % Shifted FFT
% plot(frequencies, abs(spectrum));
% title('Before Filtering - Frequency Spectrum (dem1)');
% xlabel('Frequency (Hz)');
% ylabel('Magnitude');
% 
% % Plot the FFT of dem2 with positive and negative frequencies
% figure;
% N = length(dem2); % Length of the signal
% frequencies = linspace(-fs/2, fs/2, N); % Frequency axis
% spectrum = fftshift(fft(dem2)); % Shifted FFT
% plot(frequencies, abs(spectrum));
% title('Before Filtering - Frequency Spectrum (dem2)');
% xlabel('Frequency (Hz)');
% ylabel('Magnitude');

% Calculate the frequency response of baseband_signal_1
baseband_spectrum_1 = fftshift(fft(baseband_signal_1));

% Calculate the frequency response of baseband_signal_2
baseband_spectrum_2 = fftshift(fft(baseband_signal_2));

% Define the frequency axis
N = length(baseband_signal_1); % Length of the signal
Fs_baseband = fs; % Since the baseband signals have the same sampling rate as the original signals
frequencies = linspace(-Fs_baseband/2, Fs_baseband/2, N); % Frequency axis
% 
% % Plot the magnitude of the frequency response for baseband_signal_1
% figure;
% plot(frequencies, abs(baseband_spectrum_1));
% title('Frequency Spectrum of Baseband Signal 1');
% xlabel('Frequency (Hz)');
% ylabel('Magnitude');
% 
% % Plot the magnitude of the frequency response for baseband_signal_2
% figure;
% plot(frequencies, abs(baseband_spectrum_2));
% title('Frequency Spectrum of Baseband Signal 2');
% xlabel('Frequency (Hz)');
% ylabel('Magnitude');
% 
% 
% figure;
% stem(baseband_signal_1);
% title("Demodulation - base -1 ");
% 
% figure;
% stem(baseband_signal_2);
% title("Demodulation - base - 2");

 
% figure ;
% plot(real(baseband_signal_1),real(baseband_signal_1),"o");
% title("Constellation before Line ENcoding");


% 7. LINE DECODING
matched_filter = fliplr(transmit_filter);

r1 = conv(baseband_signal_1, matched_filter, "same");
r2 = conv(baseband_signal_2, matched_filter, "same");
% disp(size(r1));
% disp(size(r2));

r1 = (downsample(r1,m));
r2 = (downsample(r2,m));

k = r1 + 1i*r2;
% 
% figure;
% hold on;
% stem(real(k));disp
% stem(imag(k));
% hold off;
% 


figure; 
plot(real(k) , imag(k) , ".");
title("Noise Constellation - Rect with memory");
xlabel("Real part");
ylabel("Imag Part");

% % downsampled_signal_input = downsample(sig_plus_noise, 4);
% 
% 
% % % figure;
% % % stem(1:length(downsampled_signal_input), real(downsampled_signal_input), 'r', 'LineWidth', 1.5);
% % % hold on;
% % % stem(1:length(downsampled_signal_input), imag(downsampled_signal_input), 'b', 'LineWidth', 1.5);
% % % title('Downsampled Signal -1 (downsampled\_signal\_input)');
% % % xlabel('Time');
% % % ylabel('Amplitude');
% % % legend('Real Part', 'Imaginary Part');
% % % grid on;
% % % 
% % % 
% % % % Plot sig_plus_noise and downsampled_signal_input together
% % % figure;
% % % subplot(2,1,1);
% % % stem(1:length(sig_plus_noise), real(sig_plus_noise), 'r', 'LineWidth', 1.5);
% % % hold on;
% % % stem(1:length(sig_plus_noise), imag(sig_plus_noise), 'b', 'LineWidth', 1.5);
% % % title('Original Signal (sig\_plus\_noise)');
% % % xlabel('Time');
% % % ylabel('Amplitude');
% % % legend('Real Part', 'Imaginary Part');
% % % grid on;
% % % 
% % % subplot(2,1,2);
% % % stem(1:length(downsampled_signal_input), real(downsampled_signal_input), 'r', 'LineWidth', 1.5);
% % % hold on;
% % % stem(1:length(downsampled_signal_input), imag(downsampled_signal_input), 'b', 'LineWidth', 1.5);
% % % title('Downsampled Signal (downsampled\_signal\_input)');
% % % xlabel('Time');
% % % ylabel('Amplitude');
% % % legend('Real Part', 'Imaginary Part');
% % % grid on;
% % % 
% % % % Reverse and flip the transmit filter to create the matched filter
% % % received_filter = fliplr(transmit_filter);
% % % 
% % % % Filter using the matched filter
% % % received_output = conv(sig_plus_noise, received_filter, 'same');
% % % 
% % % downsampled_signal = downsample(received_output, 4);
% % % 
% % % 
% % % figure;
% % % subplot(2,1,1);
% % % stem(1:length(received_output), real(received_output), 'r', 'LineWidth', 1.5);
% % % hold on;
% % % stem(1:length(received_output), imag(received_output), 'b', 'LineWidth', 1.5);
% % % title('Output of Convolution with Matched Filter');
% % % xlabel('Sample Index');
% % % ylabel('Amplitude');
% % % legend('Real Part', 'Imaginary Part');
% % % grid on;
% % % 
% % % subplot(2,1,2);
% % % stem(1:length(downsampled_signal), real(downsampled_signal), 'r', 'LineWidth', 1.5);
% % % hold on;
% % % stem(1:length(downsampled_signal), imag(downsampled_signal), 'b', 'LineWidth', 1.5);
% % % xlabel('Sample Index');
% % % ylabel('Amplitude');
% % % title('Downsampled Signal');
% % % legend('Real Part', 'Imaginary Part');
% % % grid on;
% % % 
% % % [detected_bits_input, error_rate_input] = map16qam_decision_input(downsampled_signal_input,qam_symbols);
% % % 
% % % % Plot detected bits at input
% % % figure;
% % % subplot(2,1,1);
% % % stem(1:length(detected_bits_input), real(detected_bits_input), 'r', 'LineWidth', 1.5);
% % % hold on;
% % % stem(1:length(detected_bits_input), imag(detected_bits_input), 'b', 'LineWidth', 1.5);
% % % xlabel('Sample Index');
% % % ylabel('Amplitude');
% % % title('Detected bits - Input');
% % % legend('Real Part', 'Imaginary Part');
% % % fprintf('Error rate at the input: %.6f\n', error_rate_input);
% % % 
% % % % Call the map4qam_16_decision function
% % % [detected_bits_output, error_rate_output] = map16qam_decision_output( downsampled_signal,qam_symbols);
% % % 
% % % % Plot detected bits at output
% % % subplot(2,1,2);
% % % stem(1:length(detected_bits_output), real(detected_bits_output), 'r', 'LineWidth', 1.5);
% % % hold on;
% % % stem(1:length(detected_bits_output), imag(detected_bits_output), 'b', 'LineWidth', 1.5);
% % % xlabel('Sample Index');
% % % ylabel('Amplitude');
% % % title('Detected bits - Output');
% % % legend('Real Part', 'Imaginary Part');
% % % fprintf('Error rate at the output: %.6f\n', error_rate_output);
% % % 
% % % 
% % % % Plot constellation diagram of qam_16 symbols
% % % figure;
% % % plot(real(downsampled_signal), imag(downsampled_signal), 'o', 'MarkerSize', 8);
% % % title('Constellation Diagram of qam16 Symbols- OUTPUT');
% % % xlabel('In-phase');
% % % ylabel('Quadrature');
% % % % xlim([-2.5, 2.5]);
% % % axis square;
% % % grid on;
% % % 
% % % 
% % % figure;
% % % plot(real(downsampled_signal_input), imag(downsampled_signal_input), 'o', 'MarkerSize', 8);
% % % title('Constellation Diagram of qam16 Symbols- INPUT');
% % % xlabel('In-phase');
% % % ylabel('Quadrature');
% % % % xlim([-2.5, 2.5]);
% % % axis square;
% % % grid onc

% Determine an appropriate decision rule for 16-PAM modulation
[~,N] = size(binary_vector);
estimated_bits = zeros(N,1); % Initialize array to store estimated bits
binary_vector = binary_vector.';
wrong = 0;
for i = 1:4:N
    % Extract the real and imaginary parts of the decision statistic
    real_part = real(k((i+3)/4));
    imag_part = imag(k((i+3)/4));
    
    g = 15;
    % Assign bits based on the boundaries
    if real_part < -g % Left boundary
        estimated_bits(i:i+1) = [0, 0];
    elseif real_part < 0 && real_part >= -g % Between left and center boundary
        estimated_bits(i:i+1) = [0, 1];
    elseif real_part >= 0 && real_part < g % Between center and right boundary
        estimated_bits(i:i+1) = [1, 1];
    else % Right boundary
        estimated_bits(i:i+1) = [1, 0];
    end
    
    if imag_part < -g % Lower boundary
        estimated_bits(i+2:i+3) = [0, 0];
    elseif imag_part < 0 && imag_part >= -g % Between lower and center boundary
        estimated_bits(i+2:i+3) = [0, 1];
    elseif imag_part >= 0 && imag_part < g % Between center and upper boundary
        estimated_bits(i+2:i+3) = [1, 1];
    else % Upper boundary
        estimated_bits(i+2:i+3) = [1, 0];
    end

     if estimated_bits(i) ~= binary_vector(i) 
       wrong = wrong + 1;
    end
    if estimated_bits(i+1) ~= binary_vector(i+1) 
       wrong = wrong + 1;
    end
     if estimated_bits(i+2) ~= binary_vector(i+2) 
       wrong = wrong + 1;
    end
    if estimated_bits(i+3) ~= binary_vector(i+3) 
       wrong = wrong + 1;
    end

end

error_probability = wrong / N;

char_array = num2str(estimated_bits); % Convert numeric array to character array
char_array = strrep(char_array.', ' ', ''); % Remove any spaces


% --- Binary to Audio ---

% binary_matrix = reshape(char_array, [], 8); % Reshape back to matrix
% audio_integers = bin2dec(binary_matrix); % Convert binary to decimal
% audio_reconstructed = typecast(uint8(audio_integers), 'int8'); % Typecast to int16
% audio_reconstructed_normalized = double(audio_reconstructed) / 127; % Normalize to [-1, 1]

binary_matrix = reshape(char_array.', [], 16); % Reshape back to matrix
audio_integers = bin2dec(binary_matrix); % Convert binary to decimal
audio_reconstructed = typecast(uint16(audio_integers), 'int16'); % Typecast to int16
audio_reconstructed_normalized = double(audio_reconstructed) / 32767; % Normalize to [-1, 1]
% Save the reconstructed audio
audiowrite('reconstructed_rect_memory_a_0.9_SNR_25.wav', audio_reconstructed_normalized, Fs);
