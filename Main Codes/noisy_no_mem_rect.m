clc;
clearvars;
close all;

[x,Fs] = loadAudio("project.wav");

% 1. A/D CONVERTER

% x = [0.1;0.2;0.3];
% Fs = 48000;
audio_normalized = int16(x * 32767); % Normalize
audio_binary = dec2bin(typecast(audio_normalized(:), 'uint16'), 16); % Convert to binary

% audio_normalized = int8(x * 127); % Normalize
% audio_binary = dec2bin(typecast(audio_normalized(:), 'uint8'), 8); % Convert to binary

binary_vector = audio_binary(:)';

binary_vector= str2num(binary_vector(:))'; % don't change to double !! 


% binary_vector = [0, 0, 0, 1 ,1 , 1 , 0 ,0];


% 2 . ENCODER
qam_symbols = sixteenqammap_int(binary_vector);
% qam_symbols = sixteenqammap_int(binary_vector);




% 3 . ENCODER LINE - SRRC 
m = 10;
upsampled_16qam = upsample(qam_symbols, m); % adds zeroes in between


oversampling_factor = 10; % we need to sample at only one instant 
m = oversampling_factor;


length_fil = 16;
transmit_filter = get_rectangular(length_fil); % 1/2 1/2 -> 0.0087 

%NOISELESS MODULATED SIGNAL
tx_output = conv(upsampled_16qam,transmit_filter,"same");


% 4. MODULATION
m1 = real(tx_output);
m2 = imag(tx_output);

% parameters fc
[~,p] = size(tx_output);

f = 2e6; % carrier freq 

fs = 6e6; % sampling freq 

t = (0:p-1)/fs;

c1 = cos(2*pi*f*t);
c2 = sin(2*pi*f*t);

% plot(c1)
qam_mod = 2*m1.*c1 + 2*m2.*c2;


% 5 . CHANNEL NOISE  
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
EbN0_10_2 = interp1(Pb_unique, EbN0_dB_unique, 10^-2);

disp(['Eb/N0 (dB) corresponding to Pb = 10^-2: ', num2str(EbN0_10_2)]);


Es = E_B_n_2*norm(transmit_filter)^2; % 10 is because of mean square values 
Eb = Es/4;
N0 = Eb/(10^(EbN0_10_2 /10));

% sigma_squared = N0/2;
sigma_squared = 0.5;

noise_samples = sqrt(sigma_squared) * (randn(size(qam_mod)));

sig_plus_noise = noise_samples+qam_mod;

% sig_plus_noise = awgn(qam_mod,EbN0_10_2,'measured');



% 6 . DEMODULATION 

dem1 = sig_plus_noise.*c1;
dem2 = sig_plus_noise.*c2;


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

% Calculate the frequency response of baseband_signal_1
baseband_spectrum_1 = fftshift(fft(baseband_signal_1));

% Calculate the frequency response of baseband_signal_2
baseband_spectrum_2 = fftshift(fft(baseband_signal_2));

% Define the frequency axis
N = length(baseband_signal_1); % Length of the signal
Fs_baseband = fs; % Since the baseband signals have the same sampling rate as the original signals
frequencies = linspace(-Fs_baseband/2, Fs_baseband/2, N); % Frequency axis

 


% 7. LINE DECODING
matched_filter = fliplr(transmit_filter);

r1 = conv(baseband_signal_1, matched_filter, "same");
r2 = conv(baseband_signal_2, matched_filter, "same");
% disp(size(r1));
% disp(size(r2));

r1 = (downsample(r1,m));
r2 = (downsample(r2,m));

k = r1 + 1i*r2;

figure; 
plot(real(k) , imag(k) , ".");
title("Noise Constellation- Rect memoryless");
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
    
    g = 12;
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
 
% binary_matrix = reshape(char_array.', [], 8); % Reshape back to matrix
% audio_integers = bin2dec(binary_matrix); % Convert binary to decimal
% audio_reconstructed = typecast(uint8(audio_integers), 'int8'); % Typecast to int16
% audio_reconstructed_normalized = double(audio_reconstructed) / 127; % Normalize to [-1, 1]

binary_matrix = reshape(char_array.', [], 16); % Reshape back to matrix
audio_integers = bin2dec(binary_matrix); % Convert binary to decimal
audio_reconstructed = typecast(uint16(audio_integers), 'int16'); % Typecast to int16
audio_reconstructed_normalized = double(audio_reconstructed) / 32767; % Normalize to [-1, 1]
% Save the reconstructed audio
audiowrite('reconstructed_rect_no_memory_0.5 .wav', audio_reconstructed_normalized, Fs);
