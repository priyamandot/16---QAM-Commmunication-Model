function [srrc,time_axis] = sqrt_raised_cosine(a,m,length) 
    length_os = floor(length*m); %number of samples on each side of peak
    %time vector (in units of symbol interval) on one side of the peak
    z = cumsum(ones(length_os,1))/m;
    A = 4*a*cos(pi*(1+a)*z);
    B = sin(pi*(1-a)*z)./(z);
    C = pi*(1-(4*a*z).^2);
    zerotest = m/(4*a);
    if (zerotest == floor(zerotest))
        A(zerotest) = 4*a*cos(pi*(1+a)*(z(zerotest)+0.001));
        % B(zerotest) = sin(pi*(1-a)*(z(zerotest)+0.001))./(z(zerotest)+0.001);
        C(zerotest) = pi*(1-(4*a*(z(zerotest)+0.001)).^2);
    end
    D = (A+B)./C;
    srrc = [flipud(D);1;D]; %add in peak and other side
    srrc(length_os+1) = 4*a/pi +1-a;
    time_axis = [flipud(-z);0;z];
end