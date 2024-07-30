clc;
% clearvars;
close all;


binary_vector = randi([0 1],1,80000);


% 2 . ENCODER
qam_symbols = sixteenqammap_int(binary_vector);
% qam_symbols = sixteenqammap_int(binary_vector);



% 3 . ENCODER LINE - SRRC 
m = 10;
upsampled_16qam = upsample(qam_symbols, m); % adds zeroes in between

oversampling_factor = 10; % we need to sample at only one instant 
m = oversampling_factor;
a = 1; % roll - off 
length_fil = 5;% (truncated outside [-length*T,length*T])

[~,len] = size(qam_symbols);
% [time_idx, transmit_filter] = rrc_pulse(len,10e6,a,1/1e6);
% [transmit_filter,~] = sqrt_raised_cosine(a,m,length_fil);
transmit_filter = get_rectangular(16);


tx_output = conv(upsampled_16qam,transmit_filter,"same");

% 4. MODULATION
m1 = real(tx_output);
m2 = imag(tx_output);

[~,p] = size(tx_output);

f = 2e6; % carrier freq 
% f = 100;
fs = 6e6; % sampling freq 
t = (0:p-1)/fs;

c1 = cos(2*pi*f*t);
c2 = sin(2*pi*f*t);

qam_mod = 2*m1.*c1 + 2*m2.*c2;



% 5 . CHANNEL NOISE  

% sig_plus_noise = noise_samples+qam_mod;
EbN0_dB_range = 0:1:20;

% Initialize arrays to store BER values
BER_mem_rect =  zeros(size(EbN0_dB_range));

% Loop over each Eb/N0 value
for idx = 1:length(EbN0_dB_range)
    
    EbN0_10_2 = EbN0_dB_range(idx);

    qam16_constellation = [-3-3i, -3-1i, -3+3i, -3+1i, ...
                            -1-3i, -1-1i, -1+3i, -1+1i, ...
                             3-3i,  3-1i,  3+3i,  3+1i, ...
                             1-3i,  1-1i,  1+3i,  1+1i
                             ];
    
    % Calculate the sum of squares of the constellation points
    sum_of_squares = sum(abs(qam16_constellation).^2);
    
    % Calculate average received energy per symbol
    E_B_n_2 = sum_of_squares/16; % 16qam case 
    
   
    
    Es = E_B_n_2*norm(transmit_filter)^2; % 10 is because of mean square values 
    Eb = Es/4;
    N0 = Eb/(10^(EbN0_10_2 /10));
    
    sigma_squared = N0/2;

    b_1 = 1;
    Tb = 1/(1e6);
    % delta = zeros(3,1);
    a_1 = 0.65;
    % delta(1) = a;
    % delta(b*Tb) = 1-a;
    t_1  = (0:20)/fs;

    delta = gen_mem_noise(a_1,b_1,Tb,t_1);

    qam_mod_with_memory = filter(delta,1,qam_mod); % convoluting 
     
    EbN0_dB = EbN0_dB_range(idx);


    noise_samples = sqrt(sigma_squared) * (randn(size(qam_mod_with_memory)));

    sig_plus_noise = noise_samples+qam_mod_with_memory;

    sig_plus_noise = filter(1,delta,sig_plus_noise);

    % sig_plus_noise = awgn(qam_mod,EbN0_dB,'measured');
    
    
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
    plot(real(k), imag(k), '.');
    title(['Constellation Diagram - Eb/N0 = ', num2str(EbN0_dB), ' dB']);
    xlabel('In-phase');
    ylabel('Quadrature');
    axis square;
    grid on;

    
    
    % Determine an appropriate decision rule for 16-PAM modulation
    [~,N] = size(binary_vector);
    estimated_bits = zeros(N,1); % Initialize array to store estimated bits
    % binary_vector = binary_vector.';
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
    
    BER_mem_rect(idx) = wrong / N;

end
% Calculate theoretical BER for 16-QAM
theory_BER = 3/4 * qfunc(sqrt(4/5 * 10.^(EbN0_dB_range/10))); % bit error 

% % Plot the BER versus SNR
% figure;
% plot(EbN0_dB_range, BER_no_mem, 'bo-', 'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', 'Simulated BER');
% hold on;
% plot(EbN0_dB_range, theory_BER, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Theoretical BER');
% title('Bit Error Rate (BER) vs. Signal-to-Noise Ratio (SNR)');
% xlabel('Eb/N0 (dB)');
% ylabel('BER');
% grid on;
% legend('Simulated', 'Theoritical');
% 
% 
% % Plot the BER versus SNR
% figure;
% semilogy(EbN0_dB_range, BER_no_mem, 'bo-', 'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', 'Simulated BER');
% hold on;
% semilogy(EbN0_dB_range, theory_BER, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Theoretical BER');
% title('Bit Error Rate (BER) vs. Signal-to-Noise Ratio (SNR)');
% xlabel('Eb/N0 (dB)');
% ylabel('BER');
% grid on;
% legend('Simulated', 'Theoritical');
% 
% % Plot the BER versus SNR
% figure;
% plot(EbN0_dB_range, BER_no_mem, 'bo-', 'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', 'Simulated BER');
% hold on;
% plot(EbN0_dB_range, BER_mem, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Theoretical BER');
% title('Bit Error Rate : No memory vs Memory - SRRC');
% xlabel('Eb/N0 (dB)');
% ylabel('BER');
% grid on;
% legend('No Memory', 'Memory');


% % Plot the BER versus SNR
figure;
plot(EbN0_dB_range, BER_no_mem_rect, 'bo-', 'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', 'Simulated BER');
hold on;
plot(EbN0_dB_range, BER_mem_rect, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Theoretical BER');
title('Bit Error Rate : No memory vs Memory - Rect');
xlabel('Eb/N0 (dB)');
ylabel('BER');
grid on;
legend('No Memory', 'Memory');