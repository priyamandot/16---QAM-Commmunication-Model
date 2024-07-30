function h = gen_mem_noise(a, b,Tb, t)
    % Generate impulse response h(t) = a*delta(t) + (1-a)*delta(t - bTb)
    
    t = reshape(t, 1, []);
    
    h = zeros(size(t));
    
    % Find the indices where impulses occur
    ind_delta_1 = find(t == 0);
    ind_delta_2 = find(abs(t - b*Tb) < eps*10); % tolerance -> floating point 
    
    % Assign values for the impulses
    h(ind_delta_1) = a;
    h(ind_delta_2) = 1 - a;


end
