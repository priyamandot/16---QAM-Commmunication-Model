function [time_idx, h_rrc] = rrc_pulse(N, Fs, alpha, Ts)
    T_delta = 1 / Fs;
    time_idx = ((0:N-1) - N/2) * T_delta;
    sample_num = 0:N-1;
    h_rrc = zeros(1, N);

    for x = sample_num
        t = (x - N/2) * T_delta;
        if t == 0.0
            h_rrc(x+1) = 1.0 - alpha + (4*alpha/pi);
        elseif alpha ~= 0 && t == Ts/(4*alpha)
            h_rrc(x+1) = (alpha/sqrt(2)) * (((1+2/pi) * sin(pi/(4*alpha))) + ((1-2/pi) * cos(pi/(4*alpha))));
        elseif alpha ~= 0 && t == -Ts/(4*alpha)
            h_rrc(x+1) = (alpha/sqrt(2)) * (((1+2/pi) * sin(pi/(4*alpha))) + ((1-2/pi) * cos(pi/(4*alpha))));
        else
            h_rrc(x+1) = (sin(pi*t*(1-alpha)/Ts) + 4*alpha*(t/Ts)*cos(pi*t*(1+alpha)/Ts)) / (pi*t*(1-(4*alpha*t/Ts)*(4*alpha*t/Ts))/Ts);
        end
    end
end