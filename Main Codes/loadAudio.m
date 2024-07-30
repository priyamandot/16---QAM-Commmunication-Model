% 
% function [x, Fs] = loadAudio(input)
%     % Load the audio file
%     [x_stereo, Fs] = audioread(input);
% 
%     % Convert stereo to mono
%     x = mean(x_stereo, 2);
% end



function [x, Fs] = loadAudio(input)
    % Load the audio file
    Fs= 48000;
    samples = [1,2*Fs];
    % samples = [1,0.0005*Fs];
    [x_stereo, Fs] = audioread(input,samples);

    % Convert stereo to mono
    x = mean(x_stereo, 2);
end

